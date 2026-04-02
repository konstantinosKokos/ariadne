from __future__ import annotations

from dataclasses import dataclass, field
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
    sub_traces   : for MapNode entries, one sub-trace per parallel item; None for all other nodes
    """
    step_id:      StepId
    node_id:      NodeId
    input:        BaseModel
    output:       BaseModel
    successor_id: Maybe[NodeId]
    metadata:     Metadata
    sub_traces:   Maybe[list[Trace]] = field(default=None, compare=False)


type Trace[StepId, NodeId] = list[TraceEntry[StepId, NodeId]]
