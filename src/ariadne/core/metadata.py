from typing import Callable
from pydantic import BaseModel


class Metadata(BaseModel, frozen=True):
    duration_ms:              float | None    = None
    model:                    str | None      = None
    tokens_input:             int | None      = None
    tokens_input_cached:      int | None      = None
    tokens_input_cache_write: int | None      = None
    tokens_output:            int | None      = None
    cost_usd:                 float | None    = None
    finish_reason:            str | None      = None
    tools_used:               tuple[str, ...] = ()
    retries:                  int             = 0


def _sum_or_none[A: (int, float)](*vals: A | None) -> A | None:
    nums = [v for v in vals if v is not None]
    return sum(nums) if nums else None  # type: ignore[return-value]


def _max_or_none(*vals: float | None) -> float | None:
    return max((v for v in vals if v is not None), default=None)


def _fold_metadata(metadatas: list[Metadata], duration: Callable[..., float | None]) -> Metadata:
    if not metadatas:
        return Metadata()
    return Metadata(
        duration_ms              = duration(*(m.duration_ms for m in metadatas)),
        model                    = None,
        tokens_input             = _sum_or_none(*(m.tokens_input             for m in metadatas)),
        tokens_input_cached      = _sum_or_none(*(m.tokens_input_cached      for m in metadatas)),
        tokens_input_cache_write = _sum_or_none(*(m.tokens_input_cache_write for m in metadatas)),
        tokens_output            = _sum_or_none(*(m.tokens_output            for m in metadatas)),
        cost_usd                 = _sum_or_none(*(m.cost_usd                 for m in metadatas)),
        finish_reason            = None,
        tools_used               = tuple(t for m in metadatas for t in m.tools_used),
        retries                  = sum(m.retries for m in metadatas),
    )


def _reduce_metadata(metadatas: list[Metadata]) -> Metadata:
    """
    Fold metadata across a parallel batch into one record.

    Numeric fields (tokens, cost, retries) are summed.
    duration_ms takes the max (wall-clock time of the parallel batch).
    Categorical fields (model, finish_reason) are set to None.
    tools_used is the concatenation of all per-item tuples.
    """
    return _fold_metadata(metadatas, _max_or_none)


def _total_metadata(metadatas: list[Metadata]) -> Metadata:
    """
    Fold metadata along a single sequential trace into one record.

    Identical to _reduce_metadata except duration_ms is summed: the steps run in
    sequence, so the wall-clock time is the sum of the steps'.
    """
    return _fold_metadata(metadatas, _sum_or_none)
