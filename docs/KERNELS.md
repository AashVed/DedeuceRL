# Creating Kernels

DedeuceRL's extension point is a pure `SystemKernel`, plus an optional sampler.
Kernels define hidden-system semantics only. They do not import Verifiers,
datasets, provider adapters, prompts, or CLI code.

The minimum kernel surface is:

```python
class MyKernel:
    name = "mykernel"
    version = "0.1"

    def initial_state(self, instance): ...
    def public_observation(self, instance): ...
    def tool_contracts(self, instance, state): ...
    def call(self, instance, state, tool_name, args): ...
```

Use `ToolContract` to describe tools, `KernelTransition` for probe/diagnostic
results, and `KernelJudgment` for submissions. `EpisodeRuntime` handles budget,
turns, traps, errors, event logs, and replay.

Surfaces compile kernels into:

- Hugging Face datasets and split JSON
- Verifiers environments through `dedeucerl.vf_env`
- provider-neutral tool schemas
- prompts
- CLI evaluation and interactive play

`MealyKernel` is the reference implementation.

Former skin concepts are preserved as design anchors under `docs/skin-ideas/`:

- `protocol.md`
- `apienv.md`
- `exprpolicy.md`
