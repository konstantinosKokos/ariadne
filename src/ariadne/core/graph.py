from __future__ import annotations

import traceback as tb
from typing import Callable, Literal

from pydantic import BaseModel

from .error import NodeError
from .node import AbstractNode, mk_node
from .trace import Trace, TraceEntry


# ---------------------------------------------------------------------------
# Error routing
# ---------------------------------------------------------------------------

class Error[NodeId](BaseModel, frozen=True):
    node_id: NodeId


class ErrorSink(mk_node(NodeError, NodeError)):  # type: ignore[misc]
    async def run(self, input: NodeError) -> NodeError:
        return input


def with_local_sinks[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> tuple[dict, dict]:
    error_nodes    = {Error(node_id=name): ErrorSink() for name in nodes}
    error_topology: dict[Error[NodeId], list[Error[NodeId]]] = {Error(node_id=name): [] for name in nodes}
    augmented      = {name: succs + [Error(node_id=name)] for name, succs in topology.items()}
    return {**nodes, **error_nodes}, {**augmented, **error_topology}


def with_global_sink[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> tuple[dict, dict]:
    key       = Error(node_id=None)
    augmented = {name: succs + [key] for name, succs in topology.items()}
    return {**nodes, key: ErrorSink()}, {**augmented, key: []}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def assert_initial_present[NodeId](topology: dict[NodeId, list[NodeId]], initial: NodeId) -> None:
    assert initial in topology, f"Initial node {initial!r} is not present in the graph"


def assert_no_dangling_successors[NodeId](topology: dict[NodeId, list[NodeId]]) -> None:
    (name, succ) = next(
        ((name, s) for name, succs in topology.items() for s in succs if s not in topology),
        (None, None),
    )
    assert name is None and succ is None, f"Successor {succ!r} of {name!r} is not present in the graph"


def assert_nodes_topology_consistent[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> None:
    extra_in_nodes    = set(nodes)    - set(topology)
    extra_in_topology = set(topology) - set(nodes)
    assert not extra_in_nodes,    f"Nodes {extra_in_nodes} have no entry in topology"
    assert not extra_in_topology, f"Topology names {extra_in_topology} have no implementation in nodes"


def assert_type_alignment[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> None:
    invalid = [name for name, node in nodes.items() if not isinstance(node, AbstractNode)]
    assert not invalid, f"Nodes {invalid} are not AbstractNode instances"

    non_sinks = [(name, succs) for name, succs in topology.items() if succs]

    (name, t) = next(
        (
            (name, t)
            for name, succs in non_sinks
            for t in {nodes[s].in_type for s in succs} - nodes[name].out_type
        ),
        (None, None),
    )
    assert name is None and t is None, f"Node {name!r}: successor expects {t} that {name!r} cannot produce"

    (name, t) = next(
        (
            (name, t)
            for name, succs in non_sinks
            for t in nodes[name].out_type - {nodes[s].in_type for s in succs}
        ),
        (None, None),
    )
    assert name is None and t is None, f"Node {name!r}: output type {t} is not handled by any successor"


def dispatch[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
    name:     NodeId,
    output:   BaseModel,
) -> NodeId | None:
    successors = topology[name]
    if not successors:
        return None
    matched = next((s for s in successors if isinstance(output, nodes[s].in_type)), None)
    assert matched is not None, f"No successor of {name!r} handles the produced output"
    return matched


async def run_from[StepId, NodeId](
    nodes:      dict[NodeId, AbstractNode],
    topology:   dict[NodeId, list[NodeId]],
    id_factory: Callable[[], StepId],
    name:       NodeId,
    input:      BaseModel,
) -> Trace[StepId, NodeId]:
    trace: list[TraceEntry[StepId, NodeId]] = []
    current_name  = name
    current_input = input

    while True:
        try:
            output = await nodes[current_name].run(current_input)
        except Exception as e:
            if not any(nodes[s].in_type is NodeError for s in topology[current_name]):
                raise
            output = NodeError(exception_type=type(e).__name__, message=str(e), traceback=tb.format_exc())

        successor = dispatch(nodes, topology, current_name, output)
        trace.append(TraceEntry(
            step_id      = id_factory(),
            node_id      = current_name,
            input        = current_input,
            output       = output,
            successor_id = successor,
        ))
        if successor is None:
            break
        current_name  = successor
        current_input = output

    return trace


async def resume[StepId, NodeId](
    nodes:      dict[NodeId, AbstractNode],
    topology:   dict[NodeId, list[NodeId]],
    id_factory: Callable[[], StepId],
    trace:      Trace[StepId, NodeId],
    step_id:    StepId,
) -> Trace[StepId, NodeId]:
    idx = next((i for i, e in enumerate(trace) if e.step_id == step_id), None)
    assert idx is not None, f"step_id {step_id!r} not found in trace"

    prefix = trace[:idx]
    entry  = trace[idx]

    return prefix + await run_from(nodes, topology, id_factory, entry.node_id, entry.input)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class Graph[I: BaseModel, StepId, NodeId](AbstractNode):
    """
    A typed state transition system.

    nodes      : mapping from NodeId to node implementations.
    topology   : adjacency list over NodeIds; sinks have empty successor lists.
    initial    : NodeId of the node to execute first.
    id_factory : called once per step to produce a fresh StepId.

    Graph itself is an AbstractNode, so it can be nested inside another Graph.
    In that context, run() executes the inner graph and returns the final output;
    the inner trace is independent.
    """

    def __init__(
        self,
        nodes:      dict[NodeId, AbstractNode],
        topology:   dict[NodeId, list[NodeId]],
        initial:    NodeId,
        id_factory: Callable[[], StepId],
        on_error:   Literal['raise', 'sink-local', 'sink-global'] | NodeId = 'raise',
    ) -> None:
        assert_initial_present(topology, initial)
        assert_no_dangling_successors(topology)
        assert_nodes_topology_consistent(nodes, topology)
        assert_type_alignment(nodes, topology)

        match on_error:
            case 'sink-local':
                nodes, topology = with_local_sinks(nodes, topology)
            case 'sink-global':
                nodes, topology = with_global_sink(nodes, topology)
            case 'raise':
                pass
            case node_id:
                match nodes.get(node_id, None):
                    case None:
                        raise ValueError(f"{on_error!r} is not a node identifier")
                    case AbstractNode() as node:
                        assert node.in_type is NodeError,  f"Error handler {node!r} must accept NodeError as input"
                        topology = {n: [*sucs, node_id] if sucs and n != node_id else sucs for n, sucs in topology.items()}
                    case _:
                        raise ValueError(f"{node_id!r} is not a valid node implementation")

        self.nodes      = nodes
        self.topology   = topology
        self.initial    = initial
        self.id_factory = id_factory

        self.in_type  = nodes[initial].in_type
        sinks         = [name for name, succs in topology.items() if not succs]
        self.out_type = frozenset(t for s in sinks for t in nodes[s].out_type)

    async def run(self, input: I) -> BaseModel:
        """Node interface — used when this Graph is nested inside another."""
        return (await run_from(self.nodes, self.topology, self.id_factory, self.initial, input))[-1].output

    async def execute(self, input: I) -> Trace[StepId, NodeId]:
        """Top-level execution — returns the full trace."""
        return await run_from(self.nodes, self.topology, self.id_factory, self.initial, input)

    async def resume(self, trace: Trace[StepId, NodeId], step_id: StepId) -> Trace[StepId, NodeId]:
        return await resume(self.nodes, self.topology, self.id_factory, trace, step_id)
