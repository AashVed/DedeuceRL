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
typed objectives, resource policy, objective feedback, generators, and renderers.
Let `EpisodeRuntime` canonicalize raw tool arguments and handle budget, turns,
traps, errors, events, submissions, and replay from that IR.
See [custom objectives](docs/OBJECTIVES.md) for the typed authoring contract and
initial-state evaluation semantics.

## Before a PR

Run:

```bash
git diff --cached --name-status
git ls-files --cached --ignored --exclude-from=.gitignore
ruff check .
mypy
pytest -q
python -m build
```

For user-visible changes, update:

- `CHANGELOG.md`
- `pyproject.toml` version with the minimal appropriate bump
- README/docs/examples when public behavior changes

Keep agent plans, review notes, scratch analyses, model transcripts, and generated
run artifacts under the ignored `.dedeucerl/` directory or outside the checkout.
Report review findings and benchmark scorecards in the PR description. Update
existing user documentation when behavior changes; do not add session reports to
the published documentation.

The tracked-file check above must print nothing. CI applies the root ignore policy
even if a nested `.gitignore` attempts to re-include generated files. Inspect the
staged file list before every commit; do not force-add working artifacts or add
ignore exceptions to publish them. Intentional public datasets or fixtures need
an explicit scope decision, a documented purpose, and a reproducible source.

For benchmark results, record exact commands, task/generator version, split hash,
model, feedback, rollouts, temperature, effort, outcomes, and infrastructure limits
in the PR. Preserve raw evidence locally; publish a separate artifact only when
explicitly requested. Keep that evidence separate from the library source and
package contents.
