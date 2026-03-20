from pydantic import BaseModel


class NodeError(BaseModel, frozen=True):
    exception_type: str
    message:        str
    traceback:      str
