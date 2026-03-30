import json
from typing import Callable
from pydantic import BaseModel

from .graph import Graph
from .metadata import Metadata
from .trace import Trace, TraceEntry


def validates(t: type[BaseModel], data: dict) -> bool:
    try:
        t.model_validate(data)
        return True
    except Exception:
        return False


def dump_trace[StepId, NodeId, A, B](
    trace:             Trace[StepId, NodeId],
    serialize_step_id: Callable[[StepId], A],
    serialize_node_id: Callable[[NodeId], B],
) -> str:
    return json.dumps([
        {
            'step_id':      serialize_step_id(entry.step_id),
            'node_id':      serialize_node_id(entry.node_id),
            'input':        entry.input.model_dump(mode='json'),
            'output':       entry.output.model_dump(mode='json'),
            'successor_id': serialize_node_id(entry.successor_id) if entry.successor_id is not None else None,
            'metadata':     entry.metadata.model_dump(mode='json'),
        }
        for entry in trace
    ])


def load_trace[StepId, NodeId, A, B](
    data:                  str,
    graph:                 Graph,
    deserialize_step_id:   Callable[[A], StepId],
    deserialize_node_id:   Callable[[B], NodeId],
) -> tuple[Trace[StepId, NodeId], StepId]:
    """
    Returns the restored trace and the last step_id it contains.
    The caller should use the last step_id to advance their id_factory before resuming.
    """
    def output_type(node_id: NodeId, successor_id: NodeId | None, raw_output: dict):
        if successor_id is not None:
            return graph.nodes[successor_id].in_type
        return next(
            t for t in graph.nodes[node_id].out_types
            if validates(t, raw_output)
        )

    trace = []
    for raw in json.loads(data):
        node_id      = deserialize_node_id(raw['node_id'])
        successor_id = deserialize_node_id(raw['successor_id']) if raw['successor_id'] is not None else None
        trace.append(TraceEntry(
            step_id      = deserialize_step_id(raw['step_id']),
            node_id      = node_id,
            input        = graph.nodes[node_id].in_type.model_validate(raw['input']),
            output       = output_type(node_id, successor_id, raw['output']).model_validate(raw['output']),
            successor_id = successor_id,
            metadata     = Metadata.model_validate(raw['metadata']),
        ))

    return trace, trace[-1].step_id
