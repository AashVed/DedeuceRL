# DedeuceRL

**Benchmark LLMs on Active System Identification** — probe hidden systems, form hypotheses, verify correctness.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml/badge.svg)](https://github.com/AashVed/DedeuceRL/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dedeucerl.svg)](https://pypi.org/project/dedeucerl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-DedeuceRL-yellow)](https://huggingface.co/datasets/comfortably-dumb/DedeuceRL)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18280315.svg)](https://doi.org/10.5281/zenodo.18280315)

```bash
pip install dedeucerl
dedeucerl-generate --skin mealy --seeds 0-4 --budget 25 --n-states 3 -o tasks.json
dedeucerl-eval --skin mealy --split tasks.json --model heuristic:none --out results.jsonl
dedeucerl-eval-parallel --jobs 4 --out results.jsonl --skin mealy --split tasks.json --model heuristic:none  # merged output
```

---

## Why DedeuceRL?

Modern LLMs excel at knowledge retrieval and static reasoning, but struggle with **active exploration** — systematically probing unknown systems and deducing their structure from observations.

DedeuceRL benchmarks this capability by requiring agents to:

| Capability | What We Test |
|------------|--------------|
| **Systematic Exploration** | Strategically select probes to maximize information gain |
| **Hypothesis Formation** | Build mental models of hidden system dynamics |
| **Efficient Verification** | Minimize queries while ensuring correctness |
| **Safety Awareness** | Avoid dangerous "trap" states that penalize reward |

> **Research Context**: Active system identification builds on Angluin's L* algorithm for active automata learning, conformance testing (W-method), and query-based learning theory. See [Angluin (1987)](https://doi.org/10.1016/0890-5401(87)90052-6), [Vaandrager (2017)](https://doi.org/10.1007/978-3-319-57288-8_1).

---

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Available Skins](#available-skins)
- [Interactive Game](#interactive-game)
- [Training with RL](#training-with-rl)
- [CLI Reference](#cli-reference)
- [Creating New Skins](#creating-new-skins)
- [Metrics](#metrics)
- [Citation](#citation)
- [Contributing](CONTRIBUTING.md)

---

## Installation

```bash
pip install dedeucerl                   # Core
pip install "dedeucerl[openai]"         # + OpenAI adapter
pip install "dedeucerl[all]"            # All providers
```

<details>
<summary><strong>Development installation</strong></summary>

```bash
git clone https://github.com/AashVed/DedeuceRL.git
cd DedeuceRL
pip install -e ".[dev]"
```

</details>

**Requirements:** Python 3.10+ · `verifiers>=0.1.9` · `datasets>=2.0`

---

## Quickstart

### 1. Generate a task split

```bash
dedeucerl-generate --skin mealy --seeds 0-9 --budget 25 --n-states 3 -o tasks.json
```

### 2. Evaluate a model

```bash
export OPENAI_API_KEY="sk-..."
dedeucerl-eval --skin mealy --split tasks.json --model openai:gpt-4o --out results.jsonl
```

### 3. View results

```bash
dedeucerl-aggregate results.jsonl --format markdown
```

**Output:**
```
| Model         | Episodes | Success Rate | Trap Rate | Avg Queries | Avg Reward |
|---------------|----------|--------------|-----------|-------------|------------|
| openai:gpt-4o | 10       | 40.0%        | 20.0%     | 18.2        | 0.318      |
```


## Available Skins

DedeuceRL ships with multiple "skins" — domain-specific instantiations of the active identification paradigm:

| Skin | Domain | What the Agent Must Identify |
|------|--------|------------------------------|
| **`mealy`** | Automata Theory | Hidden Mealy machine (state × input → output) |
| **`protocol`** | API Testing | REST API state-dependent behavior |
| **`apienv`** | SaaS Systems | API with methods, endpoints, variants, response schemas |
| **`exprpolicy`** | DSL Debugging | Typed policy expression (compile + test + submit) |

<details>
<summary><strong>Skin details</strong></summary>

### Mealy (Reference Skin)

The agent identifies a hidden Mealy machine (finite-state transducer).

- **Tools:** `act(symbol)` → probe, `submit_table(json)` → submit hypothesis
- **Features:** Isomorphism checking, counterexample feedback, trap transitions
- **Guarantees:** Generated machines are minimal and fully reachable

### Protocol

Reverse-engineer a stateful REST API.

- **Tools:** `api_call(method, endpoint)` → probe, `submit_spec(json)` → submit
- **Features:** State-dependent HTTP responses, behavioral equivalence

### APIEnv

Realistic SaaS API identification with variants and response schemas.

- **Tools:** `api_call(method, endpoint, variant)` → probe, `submit_spec(json)` → submit
- **Features:** Complex multi-dimensional action space

### ExprPolicy

Debug a typed policy DSL using compiler feedback and test suites.

- **Tools:** `type_check(expr)`, `run_tests(expr, suite)`, `submit(expr)`
- **Features:** Hidden tests, counterexample feedback, token constraints

</details>

---

## Interactive Game

Play any skin as a human agent to understand the challenge:

> Note: `cliGame` is a repo-only helper and is not installed via `pip install dedeucerl`.

```bash
python -m cliGame
```

```
🎮 DedeuceRL Interactive Game
Available skins: mealy, protocol, apienv, exprpolicy

Select skin [1-4]: 1
Enter seed (int): 42

=== SYSTEM PROMPT ===
You are identifying a hidden Mealy machine...

=== YOUR TURN ===
> act A
{"output": 1, "budget_left": 24, "trap_hit": false}

> act B
{"output": 2, "budget_left": 23, "trap_hit": false}

> submit_table {"n":3,"start":0,"trans":{...}}
{"ok": true}
```

**Commands:** `:help` `:tools` `:prompt` `:state` `:quit`

---

## Generating Tasks

<details>
<summary><strong>CLI Generator (recommended)</strong></summary>

```bash
# Show available parameters for a skin
dedeucerl-generate --skin mealy --show-skin-params --seeds 0 --budget 25

# Generate 100-episode Mealy test split
dedeucerl-generate \
  --skin mealy \
  --seeds 0-99 \
  --subset test \
  --budget 100 \
  --n-states 4 \
  --no-trap \
  -o seeds/mealy_test.json

# Generate Protocol split
dedeucerl-generate \
  --skin protocol \
  --seeds 0-99 \
  --budget 120 \
  --n-endpoints 5 \
  --n-states 4 \
  -o seeds/protocol_test.json
```

</details>

<details>
<summary><strong>Python API</strong></summary>

```python
from dedeucerl.skins import MealyEnv
from dedeucerl.core import TaskGenerator

gen = TaskGenerator(MealyEnv)
split = gen.generate_split(
    seeds=list(range(100)),
    budget=25,
    subset_name="test",
    n_states=5,
    trap=True,
)
gen.save_split(split, "seeds/mealy_test.json")

# Build HuggingFace Dataset
dataset = gen.build_dataset("seeds/mealy_test.json", "test", feedback=True)
```

</details>

**Pre-built splits:** [🤗 comfortably-dumb/DedeuceRL](https://huggingface.co/datasets/comfortably-dumb/DedeuceRL)

---

## Guide: Running Evaluations

### Method 1: CLI (Recommended)

```bash
# Basic evaluation
dedeucerl-eval \
  --skin mealy \
  --split seeds/mealy_smoke.json \
  --model openai:gpt-4o \
  --out results.jsonl

# With all options
dedeucerl-eval \
  --skin apienv \
  --split seeds/apienv_smoke.json \
  --model anthropic:claude-3-opus-20240229 \
  --rollouts 3 \
  --feedback \
  --temperature 0.0 \
  --verbose \
  --out results/apienv_claude.jsonl
```

### Supported Model Specs

| Provider | Format | Examples |
|----------|--------|----------|
| OpenAI | `openai:<model>` | `openai:gpt-4o`, `openai:gpt-4-turbo` |
| Anthropic | `anthropic:<model>` | `anthropic:claude-3-opus-20240229` |
| Gemini | `gemini:<model>` | `gemini:gemini-1.5-pro` |
| OpenRouter | `openrouter:<model>` | `openrouter:meta-llama/llama-3-70b` |

### Method 2: Python API

```python
from dedeucerl.skins import MealyEnv
from dedeucerl.core import TaskGenerator, make_rubric
from dedeucerl.adapters import get_adapter

# Setup
generator = TaskGenerator(MealyEnv)
dataset = generator.build_dataset("seeds/mealy_smoke.json", "dev", feedback=True)
rubric = make_rubric()
env = MealyEnv(dataset=dataset, rubric=rubric, feedback=True, max_turns=30)

# Get adapter
adapter = get_adapter("openai:gpt-4o", temperature=0.0)

# Run episode manually
item = dataset[0]
state = {"prompt": item["prompt"], "answer": item["answer"]}
# ... custom evaluation loop
```

### Aggregating Results

```bash
# CSV (for spreadsheets)
dedeucerl-aggregate results.jsonl --format csv > leaderboard.csv

# Markdown (for README/reports)
dedeucerl-aggregate results.jsonl --format markdown

# JSON (for programmatic use)
dedeucerl-aggregate results.jsonl --format json -o summary.json

# Multiple files
dedeucerl-aggregate results/*.jsonl --format markdown
```

Output columns: `model`, `n_episodes`, `success_rate`, `trap_rate`, `avg_queries`, `avg_reward`

---

## Hugging Face Dataset

Public task splits (MIT-licensed) are available at:
- [🤗 comfortably-dumb/DedeuceRL](https://huggingface.co/datasets/comfortably-dumb/DedeuceRL)

---

## Training with RL

DedeuceRL environments inherit from [`verifiers.StatefulToolEnv`](https://github.com/PrimeIntellect-ai/verifiers), making them directly compatible with RL training frameworks.

### Quick Start with vf.RLTrainer

```bash
# Install verifiers with RL support
uv add 'verifiers[rl]'

# Run training (create your own config based on verifiers docs)
uv run vf-rl @ your-config.toml
```

### Example Configuration

```toml
# your-config.toml (example)
model = "Qwen/Qwen3-4B-Instruct"

[env]
path = "./your_env_module"

[env.args]
max_turns = 30

[trainer.args]
run_name = "dedeucerl-mealy"
micro_batch_size = 4
rollouts_per_example = 16
batch_size = 1024
max_steps = 500
```

### Creating the Environment Module

```python
# your_env_module.py
import verifiers as vf
from dedeucerl.skins import MealyEnv
from dedeucerl.core import TaskGenerator, make_rubric

def load_environment(split_path: str = "your_split.json") -> vf.Environment:
    gen = TaskGenerator(MealyEnv)
    dataset = gen.build_dataset(split_path, "train", feedback=True)
    rubric = make_rubric()
    return MealyEnv(dataset=dataset, rubric=rubric, feedback=True, max_turns=30)
```

### Alternative Training Frameworks

DedeuceRL is also compatible with:
- **[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)** — Async RL at scale with FSDP2 + vLLM
- **[SkyRL](https://github.com/NovaSky-AI/SkyRL)** — [Verifiers integration](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/integrations/verifiers)
- **[Tinker](https://github.com/thinking-machines-lab/tinker-cookbook)** — [Verifiers recipes](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/verifiers_rl)

<details>
<summary><strong>Custom reward functions</strong></summary>

```python
from verifiers import Rubric, Parser

def efficiency_reward(completion, answer, state, parser):
    """Reward efficiency: fewer queries = higher reward."""
    if not state.get("ok", False):
        return 0.0
    
    queries = state.get("queries_used", 0)
    budget = state.get("budget_init", 25)
    efficiency = 1.0 - (queries / budget)
    trap_penalty = 0.5 if state.get("trap_hit", False) else 0.0
    
    return efficiency - trap_penalty

custom_rubric = Rubric(
    funcs=[efficiency_reward],
    weights=[1.0],
    parser=Parser(extract_fn=lambda s: s),
)

env = MealyEnv(dataset=dataset, rubric=custom_rubric, feedback=True, max_turns=30)
```

</details>

See [verifiers training docs](https://docs.primeintellect.ai/verifiers/training) for complete setup instructions.

---

## CLI Reference

### `dedeucerl-eval`

Run evaluations on a skin.

```bash
dedeucerl-eval \
  --skin mealy \
  --split seeds/mealy_smoke.json \
  --model openai:gpt-4o \
  --rollouts 1 \
  --out results.jsonl \
  --feedback \
  --temperature 0.0 \
  --verbose
```

**Supported model specs:** `openai:gpt-4o` · `anthropic:claude-3-opus-20240229` · `gemini:gemini-1.5-pro` · `openrouter:<model>`

**Episode selection + sharding:**

```bash
# Run only specific episodes
dedeucerl-eval --skin mealy --split seeds/mealy_smoke.json --episodes 0-4,9

# Run shard 1 of 4 (0-based shard index)
dedeucerl-eval --skin mealy --split seeds/mealy_smoke.json --shard 1/4
```

**Resume runs (split-aware):**

```bash
dedeucerl-eval --skin mealy --split seeds/mealy_smoke.json --resume --out results.jsonl
```

Resume is safe across restarts because each result line includes a `split_hash` derived from the split file + subset.

### `dedeucerl-eval-parallel`

Run shard-parallel evals and merge results.

```bash
dedeucerl-eval-parallel \
  --jobs 4 \
  --out results.jsonl \
  --skin mealy \
  --split seeds/mealy_smoke.json \
  --model openai:gpt-4o
```

This writes a single merged JSONL to `--out`. Per-shard part files are deleted by default (use `--keep-parts` to keep them). You can then run `dedeucerl-aggregate results.jsonl` as usual.

### `dedeucerl-aggregate`

Aggregate results into a leaderboard.

```bash
dedeucerl-aggregate results.jsonl --format csv > leaderboard.csv
dedeucerl-aggregate results.jsonl --format markdown
dedeucerl-aggregate results.jsonl --format json -o results_summary.json
```

### `dedeucerl-selfcheck`

Validate installation.

```bash
dedeucerl-selfcheck --verbose
```

## Creating New Skins

For detailed implementation guide, see **[docs/SKINS.md](docs/SKINS.md)**.

<details>
<summary><strong>Quick reference</strong></summary>

```python
# dedeucerl/skins/myskin.py
from dedeucerl.core.env import HiddenSystemEnv
from dedeucerl.core.config import SkinConfig

class MySkinEnv(HiddenSystemEnv):
    config = SkinConfig(skin_name="myskin", default_budget=30)
    
    def _configure_from_metadata(self, meta): ...  # Parse ground truth
    def _get_start_state(self): ...                # Initial state  
    def _get_tools(self): ...                      # [probe, submit]
    
    @staticmethod
    def generate_system_static(seed, **params): ...  # Deterministic generation
    
    @classmethod
    def domain_spec(cls, **params): ...  # Tool/observation schemas
```

Register in `dedeucerl/skins/__init__.py` and run `dedeucerl-selfcheck --verbose`.

</details>

---

## Metrics

| Metric | Description |
|--------|-------------|
| `success` | 1 if correct submission without trap hit, else 0 |
| `queries_used` | Total probe + submit calls consumed |
| `trap_hit` | 1 if dangerous state triggered |
| `budget_remaining` | Queries left at episode end |
| `reward` | `1.0 - 0.01 * queries_used` if successful, else 0 |

<details>
<summary><strong>Project structure</strong></summary>

```
DedeuceRL/
├── dedeucerl/
│   ├── core/       # HiddenSystemEnv, TaskGenerator, rubric
│   ├── skins/      # MealyEnv, ProtocolEnv, APIEnv, ExprPolicyEnv
│   ├── adapters/   # OpenAI, Anthropic, Gemini
│   ├── cli/        # dedeucerl-eval, dedeucerl-generate, etc.
│   └── utils/      # RNG utilities
├── seeds/          # Pre-built evaluation splits
└── tests/          # pytest suite
```

</details>

---

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=dedeucerl
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI models |
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible APIs (e.g., OpenRouter) |
| `ANTHROPIC_API_KEY` | API key for Anthropic models |
| `GOOGLE_API_KEY` | API key for Google Gemini models |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{dedeucerl2026,
  title = {DedeuceRL: A Modular Framework for Active System Identification Benchmarks},
  author = {Vedansh},
  year = {2026},
  url = {https://github.com/AashVed/DedeuceRL}
}
```

See [`CITATION.cff`](CITATION.cff) for full metadata.

---

## Acknowledgments

Built on: [verifiers](https://github.com/PrimeIntellect-ai/verifiers) · [Angluin's L* algorithm](https://doi.org/10.1016/0890-5401(87)90052-6) · DedeuceBench
