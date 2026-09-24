# Creating tasks

DedeuceRL's extension point is `TaskIR[State]`. It combines pure hidden-system
semantics with contracts that the runtime and all surfaces share.

The kernel remains small:

```python
class MyKernel:
    name = "mytask"
    version = "0.1"

    def initial_state(self, instance: TaskInstance) -> MyState: ...

    def call(
        self, instance: TaskInstance, state: MyState, tool_name: str, action: Any,
    ) -> KernelTransition[MyState]: ...
```

A kernel must keep instance-dependent state in `TaskInstance` and the explicit
state value, rather than mutable kernel object fields. This lets the runtime
isolate candidate execution and replay it deterministically.

`TaskIR` adds:

- `ToolActionSpace`: probe and diagnostic operations.
- `ObservationModel`: public task information, including the goal.
- `ObjectiveContract[State]`: submission tools and their grading.
- `ResourceModel`: costs and trap policy.
- `TaskGeneratorSpec`: deterministic, feasible instances.
- Optional `Renderer` objects: task-specific presentation.

Use `ToolActionContract` with `EnumSpace`, `ProductSpace`, `UnionSpace`,
`SequenceSpace`, `JsonSchemaSpace`, `MaskedSpace`, or `TypedSpace`. The runtime
canonicalizes arguments before calling the kernel. Kernels return transitions;
objectives judge submissions. `EpisodeRuntime` handles budget, traps, structured
errors, retryable submissions, event logs, and replay.

See [custom objectives](OBJECTIVES.md) for typed candidates, initial-state plan
execution, programmable feedback, and migration from hypothesis-only tasks.
`HypothesisObjective` adapts `ExactJSONContract` and
`FiniteTransducerIsomorphismContract` for reconstruction benchmarks.

See [generation and diversity](GENERATION.md) for explicit distribution invariants,
reusable graph checks, objective feasibility, and seed/version compatibility.

Surfaces compile the same task into prompts, provider-neutral schemas, datasets,
Verifiers environments, CLI evaluation/play, and MCP servers. Register a `TaskEntry`
for dataset and CLI discovery. `mealy` and `mealy_palindrome` demonstrate two
objectives over the same kernel. The runnable
[workflow example](../examples/custom_objective.py) demonstrates a different kernel.

Other domain design notes remain in `docs/skin-ideas/`: `protocol.md`, `apienv.md`,
and `exprpolicy.md`.
