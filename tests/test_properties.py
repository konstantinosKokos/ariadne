"""Property-based tests over randomly generated DAG-shaped graphs."""
import asyncio
import itertools
import time

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import BaseModel

from ariadne import Graph, mk_node, NodeError, Error, dump_trace, load_trace


# ---------------------------------------------------------------------------
# Fixed message type pool — no fields, default-constructable
# ---------------------------------------------------------------------------

class M0(BaseModel, frozen=True): pass
class M1(BaseModel, frozen=True): pass
class M2(BaseModel, frozen=True): pass
class M3(BaseModel, frozen=True): pass
class M4(BaseModel, frozen=True): pass

MSG_POOL = [M0, M1, M2, M3, M4]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_raising_node(in_type, out_type_set):
    """Test node that always raises RuntimeError."""
    out_arg = next(iter(out_type_set)) if len(out_type_set) == 1 else out_type_set

    class RaisingNode(mk_node(in_type, out_arg)):  # type: ignore[misc]
        async def run(self, input):
            raise RuntimeError("deliberate test error")

    return RaisingNode()


def make_node(in_type, out_type_set):
    """Deterministic test node: always returns a default instance of the
    lexicographically first type in out_type_set."""
    first_out = min(out_type_set, key=lambda t: t.__name__)
    out_arg   = next(iter(out_type_set)) if len(out_type_set) == 1 else out_type_set

    class TestNode(mk_node(in_type, out_arg)):  # type: ignore[misc]
        async def run(self, input):
            return first_out()

    return TestNode()


# ---------------------------------------------------------------------------
# DAG strategy
# ---------------------------------------------------------------------------
#
# Nodes are labelled 0..n-1. Node 0 is always the initial node.
# Connectivity guarantee: each node j > 0 is given at least one predecessor
# i < j, ensuring every node is reachable from 0.
# Extra edges are added randomly (i < j only, preserving acyclicity).
# Type assignment: each node i gets an in_type drawn from MSG_POOL.
# out_type for non-sinks = union of in_types of successors (satisfies
# assert_type_alignment exactly). Sinks re-use their own in_type as out_type.

@st.composite
def graph_components_strategy(draw):
    """Returns (nodes, topology) without constructing Graph — lets tests vary on_error."""
    n        = draw(st.integers(min_value=2, max_value=7))
    topology = {i: [] for i in range(n)}

    for j in range(1, n):
        topology[draw(st.integers(min_value=0, max_value=j - 1))].append(j)

    for i in range(n):
        for j in range(i + 1, n):
            if j not in topology[i] and draw(st.booleans()):
                topology[i].append(j)

    in_types = {i: draw(st.sampled_from(MSG_POOL)) for i in range(n)}

    nodes = {}
    for i in range(n):
        succs        = topology[i]
        out_type_set = {in_types[j] for j in succs} if succs else {in_types[i]}
        nodes[i]     = make_node(in_types[i], out_type_set)

    return nodes, topology


@st.composite
def graph_strategy(draw):
    nodes, topology = draw(graph_components_strategy())
    return Graph(
        nodes      = nodes,
        topology   = topology,
        initial    = 0,
        id_factory = itertools.count().__next__,
    )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@given(graph_strategy())
def test_trace_ends_at_sink(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    assert trace[-1].successor_id is None


@given(graph_strategy())
def test_trace_successor_chain(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    for i in range(len(trace) - 1):
        assert trace[i].successor_id == trace[i + 1].node_id


@given(graph_strategy())
def test_trace_nodes_in_graph(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    assert all(e.node_id in graph.nodes for e in trace)


@given(graph_strategy())
def test_trace_input_types(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    assert all(isinstance(e.input, graph.nodes[e.node_id].in_type) for e in trace)


@given(graph_strategy())
def test_trace_output_types(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    assert all(type(e.output) in graph.nodes[e.node_id].out_type for e in trace)


@given(graph_strategy())
def test_resume_suffix_matches(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    for i, entry in enumerate(trace):
        resumed = asyncio.run(graph.resume(trace, entry.step_id))
        assert resumed[:i] == trace[:i]
        for orig, res in zip(trace[i:], resumed[i:]):
            assert orig.node_id      == res.node_id
            assert orig.input        == res.input
            assert orig.output       == res.output
            assert orig.successor_id == res.successor_id


# ---------------------------------------------------------------------------
# Logging roundtrip
# ---------------------------------------------------------------------------

@given(graph_strategy())
def test_dump_load_roundtrip(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    restored, last_step_id = load_trace(dump_trace(trace), graph)
    assert restored == trace
    assert last_step_id == trace[-1].step_id


@given(graph_strategy())
def test_resume_after_load_matches(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    restored, _ = load_trace(dump_trace(trace), graph)
    for i, entry in enumerate(trace):
        r1 = asyncio.run(graph.resume(trace,    entry.step_id))
        r2 = asyncio.run(graph.resume(restored, entry.step_id))
        for e1, e2 in zip(r1, r2):
            assert e1.node_id      == e2.node_id
            assert e1.input        == e2.input
            assert e1.output       == e2.output
            assert e1.successor_id == e2.successor_id


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

@st.composite
def dangling_successor_strategy(draw):
    n        = draw(st.integers(min_value=1, max_value=5))
    topology = {i: [] for i in range(n)}
    for j in range(1, n):
        topology[draw(st.integers(min_value=0, max_value=j - 1))].append(j)
    topology[draw(st.integers(min_value=0, max_value=n - 1))].append(n)  # n does not exist
    in_types = {i: draw(st.sampled_from(MSG_POOL)) for i in range(n)}
    nodes    = {i: make_node(in_types[i], {in_types[i]}) for i in range(n)}
    return nodes, topology


@given(dangling_successor_strategy())
def test_dangling_successor_raises(args):
    nodes, topology = args
    with pytest.raises(AssertionError, match="not present in the graph"):
        Graph(nodes=nodes, topology=topology, initial=0, id_factory=itertools.count().__next__)


@given(
    st.sampled_from(MSG_POOL),  # in_0
    st.sampled_from(MSG_POOL),  # out_0  (what node 0 produces)
    st.sampled_from(MSG_POOL),  # in_1   (what node 1 expects)
    st.sampled_from(MSG_POOL),  # out_1
)
def test_type_mismatch_cannot_produce_raises(in_0, out_0, in_1, out_1):
    assume(out_0 != in_1)
    nodes    = {0: make_node(in_0, {out_0}), 1: make_node(in_1, {out_1})}
    topology = {0: [1], 1: []}
    with pytest.raises(AssertionError, match="cannot produce"):
        Graph(nodes=nodes, topology=topology, initial=0, id_factory=itertools.count().__next__)


@given(
    st.sampled_from(MSG_POOL),  # in_0
    st.sampled_from(MSG_POOL),  # extra output type node 0 produces but no successor accepts
    st.sampled_from(MSG_POOL),  # in_1 (the only type a successor accepts)
    st.sampled_from(MSG_POOL),  # out_1
)
def test_type_mismatch_uncovered_output_raises(in_0, extra, in_1, out_1):
    assume(extra != in_1)
    nodes    = {0: make_node(in_0, {in_1, extra}), 1: make_node(in_1, {out_1})}
    topology = {0: [1], 1: []}
    with pytest.raises(AssertionError, match="not handled by any successor"):
        Graph(nodes=nodes, topology=topology, initial=0, id_factory=itertools.count().__next__)


# ---------------------------------------------------------------------------
# on_error
# ---------------------------------------------------------------------------

@given(graph_components_strategy())
def test_sink_local_construction(components):
    nodes, topology = components
    Graph(nodes=nodes, topology=topology, initial=0,
          id_factory=itertools.count().__next__, on_error='sink-local')


@given(graph_components_strategy())
def test_sink_global_construction(components):
    nodes, topology = components
    Graph(nodes=nodes, topology=topology, initial=0,
          id_factory=itertools.count().__next__, on_error='sink-global')


@given(st.sampled_from(MSG_POOL), st.sampled_from(MSG_POOL))
def test_on_error_raise_propagates(in_t, out_t):
    nodes    = {0: make_raising_node(in_t, {out_t}), 1: make_node(out_t, {out_t})}
    topology = {0: [1], 1: []}
    graph    = Graph(nodes=nodes, topology=topology, initial=0,
                     id_factory=itertools.count().__next__, on_error='raise')
    with pytest.raises(RuntimeError, match="deliberate test error"):
        asyncio.run(graph.execute(in_t()))


@given(st.sampled_from(MSG_POOL), st.sampled_from(MSG_POOL))
def test_sink_local_routes_error(in_t, out_t):
    nodes    = {0: make_raising_node(in_t, {out_t}), 1: make_node(out_t, {out_t})}
    topology = {0: [1], 1: []}
    graph    = Graph(nodes=nodes, topology=topology, initial=0,
                     id_factory=itertools.count().__next__, on_error='sink-local')
    trace = asyncio.run(graph.execute(in_t()))
    assert trace[-1].successor_id is None
    assert trace[-1].node_id == Error(node_id=0)
    assert isinstance(trace[-1].output, NodeError)
    assert trace[-1].output.exception_type == 'RuntimeError'
    assert trace[-1].output.message == 'deliberate test error'


@given(st.sampled_from(MSG_POOL), st.sampled_from(MSG_POOL))
def test_sink_global_routes_error(in_t, out_t):
    nodes    = {0: make_raising_node(in_t, {out_t}), 1: make_node(out_t, {out_t})}
    topology = {0: [1], 1: []}
    graph    = Graph(nodes=nodes, topology=topology, initial=0,
                     id_factory=itertools.count().__next__, on_error='sink-global')
    trace = asyncio.run(graph.execute(in_t()))
    assert trace[-1].successor_id is None
    assert trace[-1].node_id == Error(node_id=None)
    assert isinstance(trace[-1].output, NodeError)
    assert trace[-1].output.exception_type == 'RuntimeError'
    assert trace[-1].output.message == 'deliberate test error'


@given(st.sampled_from(MSG_POOL), st.sampled_from(MSG_POOL))
def test_explicit_handler_routes_error(in_t, out_t):
    class ErrorHandler(mk_node(NodeError, NodeError)):  # type: ignore[misc]
        async def run(self, input: NodeError) -> NodeError:
            return input

    nodes    = {0: make_raising_node(in_t, {out_t}), 1: make_node(out_t, {out_t}), 'handler': ErrorHandler()}
    topology = {0: [1], 1: [], 'handler': []}
    graph    = Graph(nodes=nodes, topology=topology, initial=0,
                     id_factory=itertools.count().__next__, on_error='handler')
    trace = asyncio.run(graph.execute(in_t()))
    assert trace[-1].successor_id is None
    assert trace[-1].node_id == 'handler'
    assert isinstance(trace[-1].output, NodeError)
    assert trace[-1].output.exception_type == 'RuntimeError'


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------

def test_concurrent_executions_interleave():
    """Two graph executions running concurrently should take ~1× node latency, not ~2×."""
    DELAY = 0.05

    class SlowNode(mk_node(M0, M0)):  # type: ignore[misc]
        async def run(self, input: M0) -> M0:
            await asyncio.sleep(DELAY)
            return M0()

    graph = Graph(
        nodes      = {0: SlowNode()},
        topology   = {0: []},
        initial    = 0,
        id_factory = itertools.count().__next__,
    )

    async def run():
        return await asyncio.gather(graph.execute(M0()), graph.execute(M0()))

    t0      = time.monotonic()
    results = asyncio.run(run())
    elapsed = time.monotonic() - t0

    assert len(results) == 2
    assert elapsed < 1.5 * DELAY  # concurrent: ~DELAY; sequential would be ~2*DELAY
