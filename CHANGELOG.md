# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
