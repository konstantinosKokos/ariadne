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
