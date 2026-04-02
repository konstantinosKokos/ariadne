"""Property-based tests for ariadne: graphs, resumption, and parallel map."""
import asyncio
import functools
import itertools
import json
import time

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import BaseModel

from ariadne import Graph, AbstractNode, NodeError, Metadata, Error, MapNode, TraceEntry, dump_trace, load_trace, trace_list
from ariadne.core.parallel import _reduce_metadata


class M0(BaseModel, frozen=True): pass
class M1(BaseModel, frozen=True): pass
class M2(BaseModel, frozen=True): pass
class M3(BaseModel, frozen=True): pass
class M4(BaseModel, frozen=True): pass

MSG_POOL = [M0, M1, M2, M3, M4]


def _union(type_set):
    return functools.reduce(lambda a, b: a | b, type_set)


def make_raising_node(in_type, out_types_set):
    """Test node that always raises RuntimeError."""
    out_union = _union(out_types_set)

    class RaisingNode(AbstractNode[in_type, out_union]):  # type: ignore[misc]
        async def run(self, input):
            raise RuntimeError("deliberate test error")

    return RaisingNode()


def make_node(in_type, out_types_set):
    """Deterministic test node: always returns a default instance of the
    lexicographically first type in out_types_set."""
    first_out = min(out_types_set, key=lambda t: t.__name__)
    out_union = _union(out_types_set)

    class TestNode(AbstractNode[in_type, out_union]):  # type: ignore[misc]
        async def run(self, input):
            return first_out(), Metadata()

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
# out_types for non-sinks = union of in_types of successors (satisfies
# assert_type_alignment exactly). Sinks re-use their own in_type as out_types.

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
        out_types_set = {in_types[j] for j in succs} if succs else {in_types[i]}
        nodes[i]     = make_node(in_types[i], out_types_set)

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
    assert all(type(e.output) in graph.nodes[e.node_id].out_types for e in trace)


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


@given(graph_strategy())
def test_dump_load_roundtrip(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    restored, last_step_id = load_trace(dump_trace(trace, lambda x: x, lambda x: x), graph, lambda x: x, lambda x: x)
    assert restored == trace
    assert last_step_id == trace[-1].step_id


@given(graph_strategy())
def test_resume_after_load_matches(graph):
    trace = asyncio.run(graph.execute(graph.nodes[0].in_type()))
    restored, _ = load_trace(dump_trace(trace, lambda x: x, lambda x: x), graph, lambda x: x, lambda x: x)
    for i, entry in enumerate(trace):
        r1 = asyncio.run(graph.resume(trace,    entry.step_id))
        r2 = asyncio.run(graph.resume(restored, entry.step_id))
        for e1, e2 in zip(r1, r2):
            assert e1.node_id      == e2.node_id
            assert e1.input        == e2.input
            assert e1.output       == e2.output
            assert e1.successor_id == e2.successor_id


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
    class ErrorHandler(AbstractNode[NodeError, NodeError]):
        async def run(self, input: NodeError) -> tuple[NodeError, Metadata]:
            return input, Metadata()

    nodes    = {0: make_raising_node(in_t, {out_t}), 1: make_node(out_t, {out_t}), 'handler': ErrorHandler()}
    topology = {0: [1], 1: [], 'handler': []}
    graph    = Graph(nodes=nodes, topology=topology, initial=0,
                     id_factory=itertools.count().__next__, on_error='handler')
    trace = asyncio.run(graph.execute(in_t()))
    assert trace[-1].successor_id is None
    assert trace[-1].node_id == 'handler'
    assert isinstance(trace[-1].output, NodeError)
    assert trace[-1].output.exception_type == 'RuntimeError'


def test_concurrent_executions_interleave():
    """Two graph executions running concurrently should take ~1× node latency, not ~2×."""
    DELAY = 0.05

    class SlowNode(AbstractNode[M0, M0]):
        async def run(self, input: M0) -> tuple[M0, Metadata]:
            await asyncio.sleep(DELAY)
            return M0(), Metadata()

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


# ---------------------------------------------------------------------------
# MapNode and trace_list
# ---------------------------------------------------------------------------

class PA(BaseModel, frozen=True): value: int = 0
class PB(BaseModel, frozen=True): value: int = 0


class DoubleNode(AbstractNode[PA, PB]):
    async def run(self, input: PA) -> tuple[PB, Metadata]:
        return PB(value=input.value * 2), Metadata(tokens_input=1, tokens_output=1, cost_usd=0.001)


def make_double_graph():
    return Graph(
        nodes      = {'double': DoubleNode()},
        topology   = {'double': []},
        initial    = 'double',
        id_factory = itertools.count().__next__,
    )


pa_values = st.integers(min_value=0, max_value=100)


@st.composite
def pa_list(draw, min_size=0, max_size=8):
    vals = draw(st.lists(pa_values, min_size=min_size, max_size=max_size))
    return trace_list(PA)(items=[PA(value=v) for v in vals]), vals


@st.composite
def metadata_list(draw):
    n = draw(st.integers(min_value=0, max_value=6))
    return [
        Metadata(
            tokens_input  = draw(st.one_of(st.none(), st.integers(0, 100))),
            tokens_output = draw(st.one_of(st.none(), st.integers(0, 100))),
            cost_usd      = draw(st.one_of(st.none(), st.floats(0.0, 1.0, allow_nan=False))),
            duration_ms   = draw(st.one_of(st.none(), st.floats(0.0, 1000.0, allow_nan=False))),
            retries       = draw(st.integers(0, 5)),
        )
        for _ in range(n)
    ]


# trace_list singleton
@given(st.sampled_from(MSG_POOL))
def test_trace_list_singleton(t):
    assert trace_list(t) is trace_list(t)


# MapNode with plain inner node
@given(pa_list())
def test_map_node_plain_output_values(args):
    inp, vals = args
    mapper = MapNode(DoubleNode())
    output, _ = asyncio.run(mapper.run(inp))
    assert output.items == [PB(value=v * 2) for v in vals]  # type: ignore[attr-defined]


@given(pa_list())
def test_map_node_plain_no_sub_traces(args):
    inp, _ = args
    mapper = MapNode(DoubleNode())
    asyncio.run(mapper.run(inp))
    assert mapper.get_sub_traces() is None


# MapNode with Graph inner node
@given(pa_list(min_size=1))
def test_map_node_graph_sub_traces_length(args):
    inp, vals = args
    mapper = MapNode(make_double_graph())
    asyncio.run(mapper.run(inp))
    assert len(mapper.get_sub_traces()) == len(vals)  # type: ignore[arg-type]


@given(pa_list(min_size=1))
def test_map_node_graph_sub_traces_end_at_sink(args):
    inp, _ = args
    mapper = MapNode(make_double_graph())
    asyncio.run(mapper.run(inp))
    for sub_trace in mapper.get_sub_traces():  # type: ignore[union-attr]
        assert sub_trace[-1].successor_id is None


@given(pa_list(min_size=1))
def test_map_node_graph_sub_traces_outputs_match(args):
    inp, vals = args
    mapper = MapNode(make_double_graph())
    output, _ = asyncio.run(mapper.run(inp))
    for i, sub_trace in enumerate(mapper.get_sub_traces()):  # type: ignore[union-attr]
        assert sub_trace[-1].output == output.items[i]  # type: ignore[attr-defined]


# _reduce_metadata properties
@given(metadata_list())
def test_reduce_metadata_tokens_input_sum(metas):
    result   = _reduce_metadata(metas)
    nonnull  = [m.tokens_input for m in metas if m.tokens_input is not None]
    expected = sum(nonnull) if nonnull else None
    assert result.tokens_input == expected


@given(metadata_list())
def test_reduce_metadata_cost_sum(metas):
    result   = _reduce_metadata(metas)
    nonnull  = [m.cost_usd for m in metas if m.cost_usd is not None]
    expected = sum(nonnull) if nonnull else None
    assert (result.cost_usd is None) == (expected is None)
    if expected is not None:
        assert result.cost_usd == pytest.approx(expected)


@given(metadata_list())
def test_reduce_metadata_duration_max(metas):
    result   = _reduce_metadata(metas)
    nonnull  = [m.duration_ms for m in metas if m.duration_ms is not None]
    expected = max(nonnull) if nonnull else None
    assert result.duration_ms == expected


@given(metadata_list())
def test_reduce_metadata_retries_sum(metas):
    assert _reduce_metadata(metas).retries == sum(m.retries for m in metas)


# Serialization with sub_traces
@given(pa_list(min_size=1))
def test_dump_sub_traces_count_matches_input(args):
    inp, vals = args
    mapper = MapNode(make_double_graph())
    outer  = Graph(
        nodes      = {'map': mapper},
        topology   = {'map': []},
        initial    = 'map',
        id_factory = itertools.count().__next__,
    )
    t   = asyncio.run(outer.execute(inp))
    raw = json.loads(dump_trace(t, str, str))
    assert len(raw[0]['sub_traces']) == len(vals)
