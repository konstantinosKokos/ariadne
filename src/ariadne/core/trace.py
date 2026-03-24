from dataclasses import dataclass
from pydantic import BaseModel
from typing import Optional as Maybe

from .metadata import Metadata


@dataclass(frozen=True)
class TraceEntry[StepId, NodeId]:
    """
    A single step in a trace.

    step_id      : unique within this trace; scheme determined by the graph's id_factory
    node_id      : identifies the node that executed; scheme determined by the graph's nodes dict
    input        : the input given to the node
    output       : the output produced by the node
    successor_id : node_id of the successor that was chosen, or None if the node was a sink
    metadata     : execution metadata produced by the node and the framework
    """
    step_id:      StepId
    node_id:      NodeId
    input:        BaseModel
    output:       BaseModel
    successor_id: Maybe[NodeId]
    metadata:     Metadata


type Trace[StepId, NodeId] = list[TraceEntry[StepId, NodeId]]
