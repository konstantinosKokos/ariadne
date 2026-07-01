from typing import Literal
from pydantic import BaseModel


class NodeError(BaseModel, frozen=True):
    exception_type: str
    message:        str
    traceback:      str


class LimitExceeded(BaseModel, frozen=True):
    kind:  Literal['visits', 'steps']
    limit: int
