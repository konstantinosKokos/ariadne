import types
import typing
from abc import ABC, abstractmethod
from typing import Union, cast, get_args, get_origin

from pydantic import BaseModel

from .metadata import Metadata
from .trace import Trace


def _out_types(t) -> frozenset[type[BaseModel]]:
    if isinstance(t, typing.TypeAliasType):
        t = t.__value__
    if isinstance(t, types.UnionType) or get_origin(t) is Union:
        return frozenset(get_args(t))
    return frozenset({cast(type[BaseModel], t)})


class AbstractNode[Input: BaseModel, Output: BaseModel](ABC):
    """
    Base class for all nodes in a typed state transition graph.

        class MyNode(AbstractNode[MyInput, MyOutput]):
            async def run(self, input: MyInput) -> tuple[MyOutput, Metadata]: ...

        class MyNode(AbstractNode[MyInput, OutputA | OutputB]):
            async def run(self, input: MyInput) -> tuple[OutputA | OutputB, Metadata]: ...

    in_type and out_types are derived automatically from the type parameters.
    """

    in_type:   type[BaseModel]
    out_types: frozenset[type[BaseModel]]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        base = next((b for b in getattr(cls, '__orig_bases__', ()) if get_origin(b) is AbstractNode), None)
        if base is None:
            return
        in_t, out_t   = get_args(base)
        cls.in_type   = cast(type[BaseModel], in_t)
        cls.out_types = _out_types(out_t)

    @abstractmethod
    async def run(self, input: Input) -> tuple[Output, Metadata]: ...

    def get_sub_traces(self) -> list[Trace] | None:
        return None
