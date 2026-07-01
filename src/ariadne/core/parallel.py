from __future__ import annotations

import asyncio
from pydantic import BaseModel, create_model

from .metadata import Metadata, _reduce_metadata, _total_metadata
from .node import AbstractNode
from .trace import Trace


# ── TraceList singleton factory ────────────────────────────────────────────
_trace_list_cache: dict[type[BaseModel], type[BaseModel]] = {}


def trace_list(item_type: type[BaseModel]) -> type[BaseModel]:
    """
    Returns a Pydantic model class TraceList__<Name> with a single field
    `items: list[item_type]`.  Results are cached so that
    trace_list(A) is trace_list(A) always holds.

    Supports nesting: trace_list(trace_list(A)) is well-defined.
    """
    if item_type not in _trace_list_cache:
        _trace_list_cache[item_type] = create_model(
            f'TraceList__{item_type.__name__}',
            items=(list[item_type], ...),  # type: ignore[valid-type]
        )
    return _trace_list_cache[item_type]


# ── MapNode ────────────────────────────────────────────────────────────────
class MapNode(AbstractNode):
    """
    Applies an inner node to every element of a list in parallel via asyncio.gather.

    The inner node must have exactly one output type.

        inner: AbstractNode[A, B]
        map_node = MapNode(inner)
        # map_node: AbstractNode[TraceList[A], TraceList[B]]

    When the inner node is a Graph, the per-item inner traces are stored and
    exposed through get_sub_traces() so that the enclosing run_from records them
    in TraceEntry.sub_traces.
    """

    def __init__(self, inner: AbstractNode) -> None:
        if len(inner.out_types) != 1:
            raise ValueError("MapNode requires an inner node with exactly one output type")

        self.inner     = inner
        self.in_type   = trace_list(inner.in_type)
        self.out_types = frozenset({trace_list(next(iter(inner.out_types)))})

        self._sub_traces: list[Trace] | None = None

    def get_sub_traces(self) -> list[Trace] | None:
        return self._sub_traces

    async def run(self, input: BaseModel) -> tuple[BaseModel, Metadata]:
        from .graph import Graph

        items: list[BaseModel] = input.items  # type: ignore[attr-defined]

        if isinstance(self.inner, Graph):
            traces    = await asyncio.gather(*(self.inner.execute(item) for item in items))
            outputs   = [t[-1].output for t in traces]
            metadatas = [_total_metadata([e.metadata for e in t]) for t in traces]
            self._sub_traces = list(traces)
        else:
            pairs     = await asyncio.gather(*(self.inner.run(item) for item in items))
            outputs   = [p[0] for p in pairs]
            metadatas = [p[1] for p in pairs]
            self._sub_traces = None

        out_type = next(iter(self.out_types))
        return out_type(items=outputs), _reduce_metadata(metadatas)
