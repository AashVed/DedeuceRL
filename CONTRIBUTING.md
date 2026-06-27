# Contributing

DedeuceRL is currently an alpha kernel/runtime/surface framework. Prefer clean
architecture over compatibility shims.

## Quality Bar

- Keep kernel code pure: no Verifiers, datasets, provider adapters, prompts, or CLI imports.
- Put episode mechanics in `dedeucerl.runtime`, not in kernels or CLIs.
- Put prompts, tool-schema conversion, datasets, Verifiers, and CLIs in `dedeucerl.surface` or CLI modules.
- Do not introduce dead compatibility code for the old skin architecture.
- Add focused tests for every public behavior change.

## Adding a Kernel

A new benchmark domain should implement `SystemKernel` and usually a sampler.
The kernel should define:

- `initial_state(instance)`
- `public_observation(instance)`
- `tool_contracts(instance, state)`
- `call(instance, state, tool_name, args)`

Use `ToolContract`, `KernelTransition`, and `KernelJudgment` from
`dedeucerl.kernel`. Let `EpisodeRuntime` handle budget, turns, traps, errors,
events, submissions, and replay.

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
