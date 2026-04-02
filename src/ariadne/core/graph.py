from __future__ import annotations

import time
import traceback as tb
from typing import Any, Callable, Literal, cast

from pydantic import BaseModel

from .error import NodeError, VisitLimitExceeded, StepLimitExceeded
from .metadata import Metadata
from .node import AbstractNode
from .trace import Trace, TraceEntry


class Error[NodeId](BaseModel, frozen=True):
    node_id: NodeId


class ErrorSink(AbstractNode[NodeError, NodeError]):
    async def run(self, input: NodeError) -> tuple[NodeError, Metadata]:
        return input, Metadata()


type LimitBreached = VisitLimitExceeded | StepLimitExceeded


class Limit[NodeId](BaseModel, frozen=True):
    node_id: NodeId


class LimitSink(AbstractNode[LimitBreached, LimitBreached]):
    async def run(self, input: LimitBreached) -> tuple[LimitBreached, Metadata]:
        return input, Metadata()


def with_sinks[NodeId, SinkKey](
    nodes:     dict[NodeId, AbstractNode],
    topology:  dict[NodeId, list[NodeId]],
    key_for:   Callable[[NodeId], SinkKey],
    make_sink: Callable[[], AbstractNode],
) -> tuple[dict[NodeId | SinkKey, AbstractNode], dict[NodeId | SinkKey, list[NodeId | SinkKey]]]:
    sink_keys     = {name: key_for(name) for name in nodes}
    unique_sinks  = {key: make_sink() for key in dict.fromkeys(sink_keys.values())}
    sink_topology: dict[SinkKey, list[NodeId | SinkKey]] = {key: [] for key in unique_sinks}
    augmented     = {name: succs + [sink_keys[name]] for name, succs in topology.items()}
    return (
        cast(dict[NodeId | SinkKey, AbstractNode], {**nodes, **unique_sinks}),
        cast(dict[NodeId | SinkKey, list[NodeId | SinkKey]], {**augmented, **sink_topology}),
    )


def with_local_sinks[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> tuple[dict[NodeId | Error, AbstractNode], dict[NodeId | Error, list[NodeId | Error]]]:
    return with_sinks(nodes, topology, lambda name: Error(node_id=name), ErrorSink)


def with_global_sink[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> tuple[dict[NodeId | Error, AbstractNode], dict[NodeId | Error, list[NodeId | Error]]]:
    return with_sinks(nodes, topology, lambda _: Error(node_id=None), ErrorSink)


def with_local_limit_sinks[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> tuple[dict[NodeId | Limit, AbstractNode], dict[NodeId | Limit, list[NodeId | Limit]]]:
    return with_sinks(nodes, topology, lambda name: Limit(node_id=name), LimitSink)


def with_global_limit_sink[NodeId](
    nodes:    dict[NodeId, AbstractNode],
    topology: dict[NodeId, list[NodeId]],
) -> tuple[dict[NodeId | Limit, AbstractNode], dict[NodeId | Limit, list[NodeId | Limit]]]:
    return with_sinks(nodes, topology, lambda _: Limit(node_id=None), LimitSink)


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
            for t in {nodes[s].in_type for s in succs} - nodes[name].out_types
        ),
        (None, None),
    )
    assert name is None and t is None, f"Node {name!r}: successor expects {t} that {name!r} cannot produce"

    (name, t) = next(
        (
            (name, t)
            for name, succs in non_sinks
            for t in nodes[name].out_types - {nodes[s].in_type for s in succs}
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
    max_visits: int | dict[NodeId, int] | None = None,
    max_steps:  int | None = None,
    acyclic:    bool = False,
) -> Trace[StepId, NodeId]:
    trace:        list[TraceEntry[StepId, NodeId]] = []
    current_name  = name
    current_input = input
    visit_counts: dict[NodeId, int] = {}
    seen_states:  set = set()
    step_count    = 0

    while True:
        if acyclic:
            state = (current_name, current_input)
            assert state not in seen_states, \
                f"Cycle detected: ({current_name!r}, ...) already visited"
            seen_states.add(state)

        node_limit = max_visits if isinstance(max_visits, int) else (max_visits or {}).get(current_name)
        visits     = visit_counts.get(current_name, 0)
        sub_traces = None

        if node_limit is not None and visits >= node_limit:
            output: BaseModel = VisitLimitExceeded()
            if not any(isinstance(output, nodes[s].in_type) for s in topology[current_name]):
                raise RuntimeError(f"Visit limit of {node_limit} exceeded for {current_name!r}")
            metadata = Metadata()
        elif max_steps is not None and step_count >= max_steps:
            output   = StepLimitExceeded()
            if not any(isinstance(output, nodes[s].in_type) for s in topology[current_name]):
                raise RuntimeError(f"Step limit of {max_steps} exceeded")
            metadata = Metadata()
        else:
            visit_counts[current_name] = visits + 1
            step_count += 1
            t0 = time.monotonic()
            try:
                output, metadata = await nodes[current_name].run(current_input)
                duration_ms = (time.monotonic() - t0) * 1000
                sub_traces  = nodes[current_name].get_sub_traces()
            except Exception as e:
                if not any(nodes[s].in_type is NodeError for s in topology[current_name]):
                    raise
                duration_ms = (time.monotonic() - t0) * 1000
                output      = NodeError(exception_type=type(e).__name__, message=str(e), traceback=tb.format_exc())
                metadata    = Metadata()
                sub_traces  = None
            metadata = metadata.model_copy(update={'duration_ms': duration_ms})

        successor = dispatch(nodes, topology, current_name, output)
        trace.append(TraceEntry(
            step_id      = id_factory(),
            node_id      = current_name,
            input        = current_input,
            output       = output,
            successor_id = successor,
            metadata     = metadata,
            sub_traces   = sub_traces,
        ))
        if successor is None:
            break
        current_name, current_input  = successor, output

    return trace


async def resume[StepId, NodeId](
    nodes:      dict[NodeId, AbstractNode],
    topology:   dict[NodeId, list[NodeId]],
    id_factory: Callable[[], StepId],
    trace:      Trace[StepId, NodeId],
    step_id:    StepId,
    max_visits: int | dict[NodeId, int] | None = None,
    max_steps:  int | None = None,
    acyclic:    bool = False,
) -> Trace[StepId, NodeId]:
    idx = next((i for i, e in enumerate(trace) if e.step_id == step_id), None)
    assert idx is not None, f"step_id {step_id!r} not found in trace"
    prefix = trace[:idx]
    entry  = trace[idx]
    return prefix + await run_from(nodes, topology, id_factory, entry.node_id, entry.input,
                                   max_visits=max_visits, max_steps=max_steps, acyclic=acyclic)


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
        max_visits: int | dict[NodeId, int] | None = None,
        max_steps:  int | None = None,
        on_limit:   Literal['raise', 'sink-local', 'sink-global'] | NodeId = 'raise',
        acyclic:    bool = False,
    ) -> None:
        assert_initial_present(topology, initial)
        assert_no_dangling_successors(topology)
        assert_nodes_topology_consistent(nodes, topology)
        assert_type_alignment(nodes, topology)

        match on_error:
            case 'sink-local':
                nodes, topology = with_local_sinks(nodes, topology)  # type: ignore[assignment]
            case 'sink-global':
                nodes, topology = with_global_sink(nodes, topology)  # type: ignore[assignment]
            case 'raise':
                pass
            case node_id:
                match nodes.get(node_id, None):
                    case None:
                        raise ValueError(f"{on_error!r} is not a node identifier")
                    case AbstractNode() as node:
                        assert node.in_type is NodeError, f"Error handler {node!r} must accept NodeError as input"
                        topology = {n: [*sucs, node_id] if sucs and n != node_id else sucs for n, sucs in topology.items()}
                    case _:
                        raise ValueError(f"{node_id!r} is not a valid node implementation")

        match on_limit:
            case 'sink-local':
                nodes, topology = with_local_limit_sinks(nodes, topology)  # type: ignore[assignment]
            case 'sink-global':
                nodes, topology = with_global_limit_sink(nodes, topology)  # type: ignore[assignment]
            case 'raise':
                pass
            case node_id:
                match nodes.get(node_id, None):
                    case None:
                        raise ValueError(f"{on_limit!r} is not a node identifier")
                    case AbstractNode() as node:
                        assert isinstance(VisitLimitExceeded(), node.in_type), \
                            f"Limit handler {node_id!r} must accept VisitLimitExceeded and StepLimitExceeded"
                        topology = {n: [*sucs, node_id] if sucs and n != node_id else sucs for n, sucs in topology.items()}
                    case _:
                        raise ValueError(f"{node_id!r} is not a valid node implementation")

        self.nodes      = nodes
        self.topology   = topology
        self.initial    = initial
        self.id_factory = id_factory
        self.max_visits = max_visits
        self.max_steps  = max_steps
        self.acyclic    = acyclic

        self.in_type   = nodes[initial].in_type
        sinks          = [name for name, succs in topology.items() if not succs]
        self.out_types = frozenset(t for s in sinks for t in nodes[s].out_types)

    async def run(self, input: I) -> tuple[BaseModel, Metadata]:
        """Node interface — used when this Graph is nested inside another."""
        last = (await run_from(self.nodes, self.topology, self.id_factory, self.initial, input,
                               max_visits=self.max_visits, max_steps=self.max_steps, acyclic=self.acyclic))[-1]
        return last.output, last.metadata

    async def execute(self, input: I) -> Trace[StepId, NodeId]:
        """Top-level execution — returns the full trace."""
        return await run_from(self.nodes, self.topology, self.id_factory, self.initial, input,
                              max_visits=self.max_visits, max_steps=self.max_steps, acyclic=self.acyclic)

    async def resume(self, trace: Trace[StepId, NodeId], step_id: StepId) -> Trace[StepId, NodeId]:
        return await resume(self.nodes, self.topology, self.id_factory, trace, step_id,
                            max_visits=self.max_visits, max_steps=self.max_steps, acyclic=self.acyclic)
