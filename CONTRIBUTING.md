# Contributing

DedeuceRL is currently an alpha TaskIR/runtime/surface framework. Prefer clean
architecture over compatibility shims.

## Quality Bar

- Keep kernel code pure: no Verifiers, datasets, provider adapters, prompts, or CLI imports.
- Put executable task contracts in `dedeucerl.ir`, not in kernels or surfaces.
- Put episode mechanics in `dedeucerl.runtime`, not in kernels or CLIs.
- Put prompts, tool-schema conversion, datasets, Verifiers, and CLIs in `dedeucerl.surface` or CLI modules.
- Do not introduce dead compatibility code for the old skin architecture.
- Add focused tests for every public behavior change.

## Adding a Task

A new benchmark domain should define a pure `SystemKernel` and register it through
a `TaskIR`. The kernel should define only hidden-system semantics:

- `initial_state(instance)`
- `call(instance, state, tool_name, action)`

The `TaskIR` owns executable action spaces, tool action contracts, observations,
hypothesis judgment, resource policy, feedback policy, generators, and renderers.
Let `EpisodeRuntime` canonicalize raw tool arguments and handle budget, turns,
traps, errors, events, submissions, and replay from that IR.

## Before a PR

Run:

```bash
ruff check .
pytest -q
python -m build
```

For user-visible changes, update:

- `CHANGELOG.md`
- `pyproject.toml` version with the minimal appropriate bump
- README/docs/examples when public behavior changes

Benchmark result PRs should include exact commands, split path, model, feedback
flag, rollouts, temperature, effort, and output artifacts needed to reproduce
the result.
