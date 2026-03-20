import json
from pydantic import BaseModel

from .graph import Graph
from .trace import Trace, TraceEntry


def validates(t: type[BaseModel], data: dict) -> bool:
    try:
        t.model_validate(data)
        return True
    except Exception:
        return False


def serialize_node_id(node_id) -> object:
    return node_id.model_dump() if isinstance(node_id, BaseModel) else node_id


def deserialize_node_id(raw_id, graph: Graph):
    for key in graph.nodes:
        if isinstance(key, BaseModel):
            if key.model_dump() == raw_id:
                return key
        elif key == raw_id:
            return key
    raise ValueError(f"Node id {raw_id!r} not found in graph")


def dump_trace(trace: Trace) -> str:
    return json.dumps([
        {
            'step_id':      entry.step_id,
            'node_id':      serialize_node_id(entry.node_id),
            'input':        entry.input.model_dump(mode='json'),
            'output':       entry.output.model_dump(mode='json'),
            'successor_id': serialize_node_id(entry.successor_id) if entry.successor_id is not None else None,
        }
        for entry in trace
    ])


def load_trace[StepId, NodeId](data: str, graph: Graph) -> tuple[Trace[StepId, NodeId], StepId]:
    """
    Returns the restored trace and the last step_id it contains.
    The caller should use the last step_id to advance their id_factory before resuming.
    """
    def output_type(node_id, successor_id):
        if successor_id is not None:
            return graph.nodes[successor_id].in_type
        return next(
            t for t in graph.nodes[node_id].out_type
            if validates(t, raw['output'])
        )

    trace = []
    for raw in json.loads(data):
        node_id      = deserialize_node_id(raw['node_id'],      graph)
        successor_id = deserialize_node_id(raw['successor_id'], graph) if raw['successor_id'] is not None else None
        trace.append(TraceEntry(
            step_id      = raw['step_id'],
            node_id      = node_id,
            input        = graph.nodes[node_id].in_type.model_validate(raw['input']),
            output       = output_type(node_id, successor_id).model_validate(raw['output']),
            successor_id = successor_id,
        ))

    return trace, trace[-1].step_id
