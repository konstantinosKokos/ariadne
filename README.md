# Ariadne

A project to help me (and maybe also you) navigate the maze that modern LLM pipelines are.

[![CI](https://github.com/konstantinosKokos/ariadne/actions/workflows/ci.yml/badge.svg)](https://github.com/konstantinosKokos/ariadne/actions/workflows/ci.yml)

**Ariadne** is a simple Python framework for defining, executing, and auditing typed state transition systems, the nodes of which may involve LLMs.
The core thesis is:

> Agents are internally stochastic. The workflow is externally deterministic.

The framework makes workflows transparent, inspectable, verifiable, and resumable by treating execution as a structured trace over a properly defined transition system.

> Why should I use this over LangChain / LangGraph?

You probably shouldn't, especially if you like your packages with their bells and whistles.
If you do decide to use it against your better judgement, remember: the primary use case for ariadne is to harness static LLM-based pipelines where correctness and auditability matter.
Ariadne attempts to satisfy these demands with construction-time type safety, and a typed trace as a first-class execution artifact.

You should *definitely* not use ariadne if you're expecting dynamic graph structure, or anything requiring parallel fan-out (a node dispatching to multiple successors simultaneously) or streaming; neither feature is supported or planned.

---

## Model

A pipeline is a directed graph over typed nodes.
Each node declares an input type and one or more output types; edges connect output types to input types.
The engine routes execution by dispatching on the runtime type of each node's output.

State is a pair `(node, input)` (*i.e.*, a continuation).
There's no global state.

Every execution produces a **trace**, *i.e.* a list of `TraceEntry` values recording, for each step: the node that ran, its input, its output, its successor, and a `Metadata` side-channel.
The trace is the canonical and faithful record of a run.

---

## Nodes

Subclass `AbstractNode[Input, Output]`.
Input and output types are derived from the type parameters at class creation.

```python
from pydantic import BaseModel
from ariadne import AbstractNode, Metadata

class MyInput(BaseModel, frozen=True): ...
class MyOutput(BaseModel, frozen=True): ...

class MyNode(AbstractNode[MyInput, MyOutput]):
    async def run(self, input: MyInput) -> tuple[MyOutput, Metadata]:
        ...
```

Nodes may also produce one of several output types.

```python
type Outcome = Success | Failure | NeedsRetry

class MyNode(AbstractNode[MyInput, Outcome]):
    async def run(self, input: MyInput) -> tuple[Outcome, Metadata]:
        ...
```

`Metadata` canonically carries standard LLM fields (`model`, `tokens_input`, `tokens_output`, `cost_usd`, `finish_reason`, `tools_used`, `retries`).
The framework fills `duration_ms` after the call; all other fields are the node's responsibility and default to `None`.
You can subclass it to include fields of your own choice.

---

## Graph

```python
import itertools
from ariadne import Graph

graph = Graph(
    nodes    = {'a': NodeA(), 'b': NodeB(), 'c': NodeC()},
    topology = {'a': ['b', 'c'], 'b': [], 'c': []},
    initial  = 'a',
    id_factory = itertools.count().__next__,
)
```

`Graph` validates at construction:

- every successor is present in the adjacency list
- types align at every edge: each output type of a non-sink node is handled by exactly one successor's input type
- the initial node is present

A type mismatch raises immediately, before your expensive pipeline runs.

```
AssertionError: Node 'a': output type Foo is not handled by any successor
```

`Graph` is itself an `AbstractNode`, so graphs compose freely.
You can nest graphs in graphs and do funny things like that.

---

## Execution

```python
import asyncio

trace = asyncio.run(graph.execute(my_input))
output = trace[-1].output
```

---

## Error routing

Unhandled exceptions in `run` are caught and wrapped in `NodeError(exception_type, message, traceback)`.
The `on_error` parameter controls what happens:

```python
Graph(..., on_error='raise')        # re-raise (default)
Graph(..., on_error='sink-local')   # distinct error sink for each node
Graph(..., on_error='sink-global')  # one shared error sink for the graph
Graph(..., on_error='handler_id')   # route to a named node accepting NodeError
```

Error sinks are wired as regular nodes and edges after construction-time validation so you don't have to think about them.

---

## Termination

Loops are permitted.
If you want to make sure you don't loop uncontrollably, you can say so.

```python
Graph(
    ...,
    max_visits = 5,            # per-node visit limit (int or dict[NodeId, int])
    max_steps  = 100,          # total step limit
    on_limit   = 'sink-local', # same options as on_error
    acyclic    = True,         # assert no (node, input) state recurs
)
```

`VisitLimitExceeded` and `StepLimitExceeded` are routed identically to `NodeError`.

---

## Resumability

```python
# Resume from a specific step — the prefix is preserved, the suffix is re-run
resumed = asyncio.run(graph.resume(trace, step_id))
```

### Serialisation

```python
from ariadne import dump_trace, load_trace

json_str = dump_trace(trace, serialize_step_id=str, serialize_node_id=str)

restored, last_step_id = load_trace(
    json_str, graph,
    deserialize_step_id = int,
    deserialize_node_id = str,
)
```

Serialisers for `StepId` and `NodeId` are caller-supplied.
`load_trace` returns the last `step_id` so the caller can advance their `id_factory` before resuming.

---

## Report

```python
from ariadne import Report

report  = Report(trace)
summary = report.summary()
# summary.steps, .duration_ms, .cost_usd, .tokens_input, .tokens_output,
# .models (Counter), .finish_reasons (Counter), .errors (Counter), ...

by_node  = report.by_node()   # dict[NodeId, Report]
by_model = report.by_model()  # dict[str, Report]
```

`Summary` is a typed monoid: `reduce(add, map(Report.summary, traces), Summary())` aggregates across many runs.

---

## Requirements

Python 3.13+, Pydantic 2.

```bash
pip install -e .
```

## Warranty

None. If this sets your cat on fire, it's your fault and your fault only.
