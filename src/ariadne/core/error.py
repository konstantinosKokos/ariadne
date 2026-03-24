from pydantic import BaseModel


class NodeError(BaseModel, frozen=True):
    exception_type: str
    message:        str
    traceback:      str


class VisitLimitExceeded(BaseModel, frozen=True): pass
class StepLimitExceeded(BaseModel, frozen=True):  pass
