from abc import ABC, abstractmethod
from pydantic import BaseModel


type Message = type[BaseModel]


class AbstractNode[Input: BaseModel, Output: BaseModel](ABC):
    in_type:  type[Input]
    out_type: frozenset[Message]

    @abstractmethod
    def run(self, input: Input) -> Output: ...


def mk_node[In: BaseModel, Out: BaseModel](
    input_type:  type[In],
    output_type: type[Out] | set[type[Out]],
) -> type[AbstractNode[In, Out]]:
    """
    Return a base class for a node with declared input type In and output type Out.

        class MyNode(mk_node(MyInput, MyOutput)):
            def run(self, input: MyInput) -> MyOutput:
                ...

        class MyNode(mk_node(MyInput, {MyOutputA, MyOutputB})):
            def run(self, input: MyInput) -> MyOutputA | MyOutputB:
                ...

    The generated class carries In and Out as plain class attributes (in_type, out_type),
    making them directly accessible without any runtime introspection.
    """
    class Node(AbstractNode[In, Out]):
        in_type  = input_type
        out_type = frozenset({output_type}) if isinstance(output_type, type) else frozenset(output_type)

        @abstractmethod
        def run(self, input: In) -> Out: ...

    return Node
