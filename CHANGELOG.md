# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] - 2026-02-01

### Added
- Verifiers-compatible environment entrypoint (`dedeucerl.vf_env`) to train any skin via a single `env.id`.
- Seed-based dataset generation for training without requiring split files.
- Sample `vf-rl` configs under `configs/vf-rl/`.
- `dedeucerl-train` CLI to generate and optionally run `vf-rl` configs.
- Training-friendly `reward_mode="train_dense"` rubric (generic, skin-agnostic).

### Changed

## [1.0.4] - 2026-01-31

### Added
- `dedeucerl-eval-parallel`: shard-parallel evaluation runner that merges per-shard JSONL outputs.
- Episode selection + sharding for `dedeucerl-eval` via `--episodes` and `--shard`.
- Split-aware resuming for `dedeucerl-eval` via `--resume` (and `--append` for explicit appends).
- Episode utilities: `parse_index_spec`, `parse_shard`, `apply_shard`, `compute_split_hash`.
- CLI tests covering selection, sharding, resume, and parallel parity.

### Changed
- `dedeucerl-eval` now streams results to JSONL (flush per episode/rollout) instead of buffering all results in memory.
- Updated smoke splits (`seeds/*_smoke.json`) to use a `dev` subset key and expanded episode items.

## [1.0.3] - 2026-01-29

### Added
- GitHub Actions CI workflow and PyPI publish workflow.
- `cliGame`: interactive CLI game for playing/exploring environments.
- `CONTRIBUTING.md` and skin documentation (`docs/SKINS.md`).

### Changed
- Documentation improvements and prompt clarity tweaks.

## [1.0.2] - 2026-01-22

### Added
- New `ExprPolicyEnv` skin (`exprpolicy`) modeling typed policy/DSL debugging with compiler + test feedback.
- Universal, schema-first skin contract test suite (`tests/test_universal_skins.py`) to centralize skin validation.

### Changed
- `TaskGenerator` no longer forwards `None` values into `domain_spec()` / `build_observation()` to prevent `null` fields in observations.
- Standardized submit tool payloads to always include `counterexample` (set to `null` when not applicable).

## [1.0.1] - 2026-01-17

### Changed
- Improve split generation docs and OpenRouter support.
