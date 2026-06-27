# Creating Tasks

DedeuceRL's extension point is `TaskIR`. A task combines pure hidden-system
semantics with executable contracts that every surface can compile.

The minimum semantic kernel remains small:

```python
class MyKernel:
    name = "mytask"
    version = "0.1"

    def initial_state(self, instance): ...
    def call(self, instance, state, tool_name, action): ...
```

The `TaskIR` wraps that kernel with:

- a `ToolActionSpace`
- an `ObservationModel`
- a `HypothesisContract`
- a `ResourceModel`
- a `FeedbackModel`
- a `TaskGeneratorSpec`
- optional `Renderer` objects

Use `ToolActionContract` plus action spaces such as `EnumSpace`, `ProductSpace`,
`UnionSpace`, `SequenceSpace`, `JsonSchemaSpace`, and `MaskedSpace` to describe
tools. `EpisodeRuntime` canonicalizes raw tool arguments before calling kernels.
Use `KernelTransition` for probe/diagnostic results and `KernelJudgment` for
submissions. `EpisodeRuntime` handles budget, turns, traps, errors, event logs,
and replay.

Surfaces compile TaskIR into:

- Hugging Face datasets and split JSON
- Verifiers environments through `dedeucerl.vf_env`
- provider-neutral tool schemas
- prompts
- CLI evaluation and interactive play

`MealyKernel` plus the Mealy TaskIR is the reference implementation.

Former skin concepts are preserved as design anchors under `docs/skin-ideas/`:

- `protocol.md`
- `apienv.md`
- `exprpolicy.md`
