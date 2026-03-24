from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce
from typing import Callable as Fn

from .error import NodeError
from .trace import Trace, TraceEntry


def nullable_sum(a: float | None, b: float | None) -> float | None:
    match a, b:
        case (None, x) | (x, None): return x
        case _: return a + b


@dataclass
class Summary:
    steps:                    int          = 0
    errors:                   Counter[str] = field(default_factory=Counter)
    duration_ms:              float | None = None
    cost_usd:                 float | None = None
    tokens_input:             int | None   = None
    tokens_input_cached:      int | None   = None
    tokens_input_cache_write: int | None   = None
    tokens_output:            int | None   = None
    retries:                  int          = 0
    tools:                    Counter[str] = field(default_factory=Counter)
    finish_reasons:           Counter[str] = field(default_factory=Counter)
    models:                   Counter[str] = field(default_factory=Counter)

    def __add__(self, other: Summary) -> Summary:
        return Summary(
            steps                    = self.steps + other.steps,
            errors                   = self.errors + other.errors,
            duration_ms              = nullable_sum(self.duration_ms, other.duration_ms),
            cost_usd                 = nullable_sum(self.cost_usd, other.cost_usd),
            tokens_input             = nullable_sum(self.tokens_input, other.tokens_input),
            tokens_input_cached      = nullable_sum(self.tokens_input_cached, other.tokens_input_cached),
            tokens_input_cache_write = nullable_sum(self.tokens_input_cache_write, other.tokens_input_cache_write),
            tokens_output            = nullable_sum(self.tokens_output, other.tokens_output),
            retries                  = self.retries + other.retries,
            tools                    = self.tools + other.tools,
            finish_reasons           = self.finish_reasons + other.finish_reasons,
            models                   = self.models + other.models,
        )


def to_summary(entry: TraceEntry) -> Summary:
    m = entry.metadata
    return Summary(
        steps                    = 1,
        errors                   = Counter({entry.output.exception_type: 1}) if isinstance(entry.output, NodeError) else Counter(),
        duration_ms              = m.duration_ms,
        cost_usd                 = m.cost_usd,
        tokens_input             = m.tokens_input,
        tokens_input_cached      = m.tokens_input_cached,
        tokens_input_cache_write = m.tokens_input_cache_write,
        tokens_output            = m.tokens_output,
        retries                  = m.retries,
        tools                    = Counter(m.tools_used),
        finish_reasons           = Counter({m.finish_reason: 1}) if m.finish_reason is not None else Counter(),
        models                   = Counter({m.model: 1})         if m.model         is not None else Counter(),
    )


def summarize(trace: Trace) -> Summary:
    return reduce(lambda a, b: a + b, (to_summary(e) for e in trace), Summary())


def by_node[StepId, NodeId](trace: Trace[StepId, NodeId]) -> dict[NodeId, Trace[StepId, NodeId]]:
    return {
        node_id: [e for e in trace if e.node_id == node_id]
        for node_id in dict.fromkeys(e.node_id for e in trace)
    }


def by_model[StepId, NodeId](trace: Trace[StepId, NodeId]) -> dict[str, Trace[StepId, NodeId]]:
    return {
        model: [e for e in trace if e.metadata.model == model]
        for model in dict.fromkeys(e.metadata.model for e in trace)
    }


class Report[StepId, NodeId]:
    def __init__(self, trace: Trace[StepId, NodeId]) -> None:
        self.trace = trace

    def fmap[A](self, f: Fn[[Trace[StepId, NodeId]], dict[A, Trace[StepId, NodeId]]]) -> dict[A, Report[StepId, NodeId]]:
        return {key: Report(t) for key, t in f(self.trace).items()}

    def by_node(self) -> dict[NodeId, Report[StepId, NodeId]]:
        return self.fmap(by_node)

    def by_model(self) -> dict[str, Report[StepId, NodeId]]:
        return self.fmap(by_model)

    def summary(self) -> Summary:
        return summarize(self.trace)

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f'Report(steps={s.steps}, '
            f'duration_ms={s.duration_ms}, '
            f'cost_usd={s.cost_usd}, '
            f'tokens=({s.tokens_input}, {s.tokens_output}), '
            f'errors={sum(s.errors.values())})'
        )

    def __str__(self) -> str:
        s = self.summary()
        node_lines = [
            f'  {node_id}: {summarize(t)!r}'
            for node_id, t in by_node(self.trace).items()
        ]
        return '\n'.join([
            repr(self),
            f'  errors:         {dict(s.errors)}'         if s.errors         else '  errors:         none',
            f'  tools:          {dict(s.tools)}'          if s.tools          else '  tools:          none',
            f'  finish_reasons: {dict(s.finish_reasons)}' if s.finish_reasons else '  finish_reasons: none',
            f'  models:         {dict(s.models)}'         if s.models         else '  models:         none',
            'by node:',
            *node_lines,
        ])
