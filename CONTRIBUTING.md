# Contributing to DedeuceRL

Thanks for your interest in contributing! DedeuceRL is a benchmark + environment framework for **active system identification** with an emphasis on:
- reproducibility (seeded task splits),
- clear interfaces (schema-first skins),
- and easy-to-share results (JSONL + aggregation).

## Quick links
- Bugs / feature requests: open a GitHub issue
- Security issues: please do **not** open a public issue; contact the maintainers privately

## Development setup

```bash
git clone https://github.com/AashVed/DedeuceRL.git
cd DedeuceRL
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## Running checks locally

```bash
ruff check .
pytest -q
dedeucerl-selfcheck --verbose
```

## Contribution types

### 1) Bug fixes
Please include:
- a minimal reproduction (commands + seed/split),
- expected vs actual behavior,
- and a test if it’s feasible.

### 2) New skins (new benchmark domains)
Skins are the main way to extend DedeuceRL. A good skin should:
- implement the `HiddenSystemEnv` contract,
- define a single source of truth via `domain_spec()` (tools + enums + hypothesis schema + observation fields),
- and provide deterministic `generate_system_static(seed=...)`.

Suggested checklist for a PR:
- add the skin to `dedeucerl/skins/__init__.py` registry
- add at least one focused test in `tests/` (and ensure `tests/test_universal_skins.py` passes)
- document the skin briefly in `README.md`

### 3) Adapters (new model providers)
Adapters should translate DedeuceRL’s canonical transcript:
- assistant tool calls (`tool_calls`) and
- tool results (`role="tool"`, `tool_call_id`, `content`)

into the provider’s expected request/response format.

Please include:
- a small unit test that validates transcript round-trip behavior (no network calls),
- clear environment variables for auth/config.

### 4) Submitting benchmark results
We welcome result PRs, but they must be reproducible and comparable.

Include in your PR description:
- exact command(s) used (including skin, split path, subset, feedback flag, rollouts, temperature)
- model identifier (and provider, e.g. `openai:gpt-4o` / `anthropic:...` / `gemini:...` / `openrouter:...`)
- date run + region (if relevant)
- package versions (at least `dedeucerl`, `verifiers`, `datasets`)

Recommended workflow:
```bash
# 1) Use an existing split, or generate one
dedeucerl-generate --skin mealy --seeds 0-99 --budget 100 --n-states 4 --no-trap -o seeds/mealy_n4_b100_test.json

# 2) Run eval
dedeucerl-eval --skin mealy --split seeds/mealy_n4_b100_test.json --model openai:gpt-4o --out results/mealy_gpt4o.jsonl

# 3) Aggregate
dedeucerl-aggregate results/mealy_gpt4o.jsonl --format markdown
```

Notes:
- Prefer `temperature 0.0` for determinism unless you’re explicitly studying sampling effects.
- If you use `openrouter:*`, include the base URL and any routing configuration.
- Don’t include API keys or secrets in logs/artifacts.

## Style guidelines
- Run `ruff check .` before pushing.
- Keep PRs focused; avoid refactors unrelated to your change.

