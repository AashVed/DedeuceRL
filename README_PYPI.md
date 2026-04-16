# DedeuceRL

Benchmark LLMs on active system identification: probe hidden systems, form hypotheses, and verify correctness.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml/badge.svg)](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dedeucerl.svg)](https://pypi.org/project/dedeucerl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-DedeuceRL-yellow)](https://huggingface.co/datasets/comfortably-dumb/DedeuceRL)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18280315.svg)](https://doi.org/10.5281/zenodo.18280315)

Full repository README, benchmark visuals, and documentation:

- GitHub: https://github.com/AashVed/DedeuceRL
- Dataset: https://huggingface.co/datasets/comfortably-dumb/DedeuceRL

## Installation

```bash
pip install dedeucerl
pip install "dedeucerl[openai]"
pip install "dedeucerl[all]"
pip install "dedeucerl[rl]"
```

Requirements: Python 3.10+, `verifiers>=0.1.9`, `datasets>=2.0`.

## Quickstart

```bash
dedeucerl-generate --skin mealy --seeds 0-9 --budget 25 --n-states 3 -o tasks.json
dedeucerl-eval --skin mealy --split tasks.json --model heuristic:none --out results.jsonl
dedeucerl-aggregate results.jsonl --format markdown
```

## Built-in Skins

- `mealy`: hidden Mealy machine identification
- `protocol`: stateful REST API reverse engineering
- `apienv`: SaaS API workflow identification
- `exprpolicy`: typed policy DSL debugging

## Main CLI Commands

- `dedeucerl-generate`
- `dedeucerl-eval`
- `dedeucerl-eval-parallel`
- `dedeucerl-aggregate`
- `dedeucerl-selfcheck`
- `dedeucerl-train`

## Development

```bash
git clone https://github.com/AashVed/DedeuceRL.git
cd DedeuceRL
pip install -e ".[dev]"
pytest -q
```

License: MIT.
