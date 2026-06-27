# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-06-27

### Changed
- Rebuilt DedeuceRL around Kernel, Runtime, and Surface layers.
- Replaced skin-based extension with pure `SystemKernel` contracts, generic `EpisodeRuntime`, and surface compilers for datasets, prompts, tools, CLI eval, CLI game, and Verifiers.
- Rebuilt Mealy as the reference `MealyKernel`.

### Removed
- Removed the old `HiddenSystemEnv`, `TaskGenerator`, `DomainSpec`, and built-in skin environment classes.
- Removed Protocol, APIEnv, and ExprPolicy example skins until they can be remade on the new kernel architecture.

## [1.0.9] - 2026-04-22

### Changed
- DedeuceRL is now Prime-first for RL workflows. Hosted training uses `prime rl run`, and self-managed training uses `prime-rl`, both through `dedeucerl.vf_env` instead of the removed legacy trainer wrapper.
- Package dependency floors now target the current stable Prime ecosystem: `verifiers>=0.1.12,<0.2` and `datasets>=3.0,<4.7.0`.
- Training documentation now reflects the current Prime workflow for both hosted training and self-managed Prime-compatible development.

### Removed
- The `dedeucerl-train` CLI and its checked-in legacy training templates.
- The `dedeucerl[rl]` extra. Prime should be installed separately via `uv tool install prime`.

## [1.0.8] - 2026-04-05

### Added
- `dedeucerl-aggregate` now emits additive benchmark summaries including `eval_config_hash`, `max_complete_k`, `pass_at_1`, and `pass_at_3` when sufficient rollouts are available.

### Fixed
- `dedeucerl-eval-parallel --resume` now reconstructs shard-local resume state from the merged output, so completed work is not rerun after shard part files have been deleted.
- `dedeucerl-eval` now derives episode turn limits per problem, allowing mixed-difficulty split files to run correctly.

### Changed
- Eval result rows now include normalized evaluation provenance via `eval_config` and `eval_config_hash`, and resume/summary filtering respects that identity.

## [1.0.7] - 2026-04-03

### Changed
- `dedeucerl-aggregate` now groups results by model + skin + split hash, and reports both run count (`n_runs`) and unique episode count (`n_episodes`).
- Documentation now clarifies that `dedeucerl-eval` reports the benchmark reward, while RL training may use `reward_mode="train_dense"` or other training-oriented rubric settings.

## [1.0.6] - 2026-02-01

### Added
- `dedeucerl-eval --effort <level>`: pass provider reasoning/thinking effort for supported models.
- `dedeucerl-eval --no-effort-probe`: skip the cheap probe call used to validate `--effort`.

### Changed
- Gemini adapter now supports only the official `google-genai` SDK (no legacy `google-generativeai` fallback).
- `--effort` validation is now provider-driven: a cheap probe call catches unsupported settings early (instead of hardcoded model tables).

## [1.0.5] - 2026-02-01

### Added
- Verifiers-compatible environment entrypoint (`dedeucerl.vf_env`) to train any skin via a single `env.id`.
- Seed-based dataset generation for training without requiring split files.
- Sample legacy training configs.
- Legacy training CLI for config generation and launch.
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
