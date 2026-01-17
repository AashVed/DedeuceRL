# DedeuceRL

**A Modular Framework for Active System Identification Benchmarks**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18280315.svg)](https://doi.org/10.5281/zenodo.18280315)

---

## Overview

DedeuceRL is a modular, extensible framework for benchmarking LLM agents on **active system identification** tasks. The core paradigm:

1. **Hidden System**: An unknown system with hidden dynamics (e.g., a Mealy machine or API state machine)
2. **Probe Actions**: The agent queries the system to gather observations (each probe costs 1 budget unit)
3. **Hypothesis Submission**: The agent submits a hypothesis (each submission costs 1 budget unit; correct ends the episode)
4. **Verification**: The hypothesis is checked for correctness (possibly up to isomorphism)

This framework enables researchers to:
- **Evaluate** LLM agents on active system identification tasks
- **Train** agents via RL on these environments (compatible with `verifiers` library)
- **Create new skins** (domains) without reimplementing core infrastructure

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DedeuceRL                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Evaluation Harness                    │   │
│  │  • CLI (dedeucerl-eval, dedeucerl-aggregate)            │   │
│  │  • Model Adapters (OpenAI, Anthropic, Gemini)           │   │
│  │  • Result Aggregation & Leaderboard                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────┐     │
│  │                    Core Abstraction                    │     │
│  │  • HiddenSystemEnv (base class)                        │     │
│  │  • SkinConfig (per-skin settings)                      │     │
│  │  • TaskGenerator (dataset building)                    │     │
│  │  • Metrics (success, efficiency, safety)               │     │
│  └───────────────────────────────────────────────────────┘     │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────┐     │
│  │                        Skins                           │     │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐             │     │
│  │  │ Mealy   │  │Protocol │  │ APIEnv  │  ...         │     │
│  │  │  (v1)   │  │  (v1)   │  │  (v1)   │             │     │
│  │  └─────────┘  └─────────┘  └─────────┘             │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### From Source

```bash
git clone https://github.com/AashVed/DedeuceRL.git
cd DedeuceRL

# Core install (no provider SDKs)
pip install -e .

# Provider adapters (optional extras)
pip install -e ".[openai]"      # OpenAI / OpenAI-compatible APIs
pip install -e ".[anthropic]"   # Anthropic Claude
pip install -e ".[gemini]"      # Google Gemini
# Or all providers at once:
pip install -e ".[all]"
```

### Dependencies

- Python 3.10+
- Core: `verifiers>=0.1.9`, `datasets>=2.0`
- Optional provider extras: `openai`, `anthropic`, `google-genai` / `google-generativeai`

### Development Installation

```bash
pip install -e ".[dev]"
```

---

## Quick Start

```bash
# 1. Generate tasks
python -c "
from dedeucerl.skins import MealyEnv
from dedeucerl.core import TaskGenerator

gen = TaskGenerator(MealyEnv)
split = gen.generate_split(seeds=range(10), budget=25, subset_name='dev', n_states=3)
gen.save_split(split, 'seeds/my_split.json')
"

# 2. Run evaluation
dedeucerl-eval --skin mealy --split seeds/my_split.json --model openai:gpt-4o --out results.jsonl

# 3. View results
dedeucerl-aggregate results.jsonl --format markdown
```

---

## Guide: Generating Tasks

DedeuceRL uses a **TaskGenerator** to create reproducible evaluation splits from any skin.

### Step 1: Choose a Skin

```python
from dedeucerl.skins import MealyEnv, ProtocolEnv
from dedeucerl.core import TaskGenerator

# Pick your skin
skin = MealyEnv        # Mealy machine identification
# skin = ProtocolEnv   # API reverse engineering

generator = TaskGenerator(skin)
```

### Step 2: Configure Generation Parameters

Each skin has specific parameters:

```python
# MealyEnv parameters
split = generator.generate_split(
    seeds=list(range(100)),   # 100 unique episodes
    budget=25,                 # Queries allowed per episode
    subset_name="test",        # Name this subset
    n_states=5,                # Mealy: number of states
    trap=True,                 # Include trap transitions
)

# ProtocolEnv parameters
split = generator.generate_split(
    seeds=list(range(50)),
    budget=30,
    subset_name="dev",
    n_endpoints=4,             # Protocol: number of API endpoints
    n_states=3,                # Protocol: number of API states
)
```

### Step 3: Save and Load Splits

```python
# Save to JSON (for reproducibility)
generator.save_split(split, "seeds/mealy_test.json")

# Build HuggingFace Dataset from saved split
dataset = generator.build_dataset(
    split_path="seeds/mealy_test.json",
    subset="test",
    feedback=True,   # Enable counterexample feedback
)

print(f"Generated {len(dataset)} episodes")
print(f"Example prompt: {dataset[0]['prompt'][0]['content'][:100]}...")
```

### Difficulty Scaling

| Parameter | Effect on Difficulty |
|-----------|---------------------|
| `n_states` | More states = harder identification |
| `budget` | Lower budget = less room for error |
| `trap=True` | Safety constraints add complexity |
| `feedback=False` | No counterexamples = much harder |

---

## Guide: Running Evaluations

### Method 1: CLI (Recommended)

```bash
# Basic evaluation
dedeucerl-eval \
  --skin mealy \
  --split seeds/mealy_dev.json \
  --subset dev \
  --model openai:gpt-4o \
  --out results.jsonl

# With all options
dedeucerl-eval \
  --skin protocol \
  --split seeds/protocol_dev.json \
  --model anthropic:claude-3-opus-20240229 \
  --rollouts 3 \
  --feedback \
  --temperature 0.0 \
  --verbose \
  --out results/protocol_claude.jsonl
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
dataset = generator.build_dataset("seeds/mealy_dev.json", "dev", feedback=True)
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

## Guide: Training with Verifiers

## Logging and Debugging

DedeuceRL includes a small structured logging utility intended for *library users* embedding the environments in their own training/eval loops.

- Logger helpers live in `dedeucerl/utils/logging.py` (e.g., `configure_logging()`, `log_episode_start()`).
- The CLI tools intentionally use `print()` for predictable, copy-pastable terminal output.

## Optional Type Helpers

The dataclasses in `dedeucerl/core/types.py` (`EpisodeState`, `ProbeResult`, `SubmitResult`) are optional, exported helpers for users who want structured types in custom loops. The built-in skins operate on `Dict[str, Any]` state for compatibility with `verifiers`.

DedeuceRL integrates directly with [PrimeIntellect's verifiers library](https://github.com/PrimeIntellect-ai/verifiers) for RL training.

### Basic Setup

```python
from dedeucerl.skins import MealyEnv
from dedeucerl.core import TaskGenerator, make_rubric

# Generate training data
generator = TaskGenerator(MealyEnv)
train_split = generator.generate_split(
    seeds=list(range(1000)),  # 1000 training episodes
    budget=25,
    subset_name="train",
    n_states=4,
)
generator.save_split(train_split, "seeds/mealy_train.json")

# Build dataset
train_dataset = generator.build_dataset("seeds/mealy_train.json", "train", feedback=True)

# Create environment (inherits from verifiers.StatefulToolEnv)
rubric = make_rubric()
env = MealyEnv(
    dataset=train_dataset,
    rubric=rubric,
    feedback=True,
    max_turns=30,
)
```

### Integration with GRPO/verl

```python
# The env is directly compatible with verifiers training loops
from verifiers.trainers import GRPOTrainer  # Example

trainer = GRPOTrainer(
    env=env,
    model="your-model",
    # ... other training config
)
trainer.train()
```

### Custom Reward Functions

```python
from verifiers import Rubric, Parser

def my_reward(completion, answer, state, parser):
    """Custom reward: bonus for efficiency."""
    if not state.get("ok", False):
        return 0.0
    
    queries = state.get("queries_used", 0)
    budget = 25
    efficiency = 1.0 - (queries / budget)
    trap_penalty = 0.5 if state.get("trap_hit", False) else 0.0
    
    return efficiency - trap_penalty

custom_rubric = Rubric(
    funcs=[my_reward],
    weights=[1.0],
    parser=Parser(extract_fn=lambda s: s),
)

env = MealyEnv(dataset=dataset, rubric=custom_rubric, feedback=True, max_turns=30)
```

---

## CLI Reference

### `dedeucerl-eval`

Run evaluations on a skin.

```bash
dedeucerl-eval \
  --skin mealy \              # Skin to use
  --split seeds/dev.json \    # Path to split JSON
  --subset dev \              # Subset name within split
  --model openai:gpt-4o \     # Model spec (provider:model)
  --rollouts 1 \              # Rollouts per episode
  --out results.jsonl \       # Output file
  --feedback \                # Enable counterexample feedback
  --temperature 0.0 \         # Sampling temperature
  --verbose                   # Verbose output
```

**Supported model specs:**
- `openai:gpt-4o`, `openai:gpt-4-turbo`, `openai:gpt-3.5-turbo`
- `anthropic:claude-3-opus-20240229`, `anthropic:claude-3-sonnet-20240229`
- `gemini:gemini-1.5-pro`, `gemini:gemini-1.5-flash`
- `openrouter:<any-model>` (requires `OPENAI_BASE_URL` set)

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

---

## Skins

### MealySkin (Mealy Machine Identification)

The reference skin for active system identification. The agent must identify a hidden Mealy machine (finite-state transducer).

**Hidden System:** Transition table `(state × symbol → next_state × output)`

**Tools:**
- `act(symbol: str)` - Execute symbol ('A', 'B', or 'C'), returns output and state info
- `submit_table(table_json: str)` - Submit hypothesis as JSON

**Hypothesis Schema:**
```json
{
  "n": 3,
  "start": 0,
  "trans": {
    "0": {"A": [1, 0], "B": [0, 1], "C": [2, 2]},
    "1": {"A": [2, 1], "B": [1, 0], "C": [0, 2]},
    "2": {"A": [0, 2], "B": [2, 1], "C": [1, 0]}
  }
}
```

**Features:**
- Isomorphism checking (accepts equivalent state relabelings)
- Counterexample generation (BFS-based shortest distinguishing trace)
- Trap transitions (safety-critical states)
- Guaranteed reachability and minimality in generated machines

### ProtocolSkin (API Reverse Engineering)

The agent must reverse-engineer a hidden REST API's state-dependent behavior.

**Hidden System:** State machine representing valid API call sequences

**Tools:**
- `api_call(method: str, endpoint: str)` - Make API call, returns status and response
- `submit_spec(spec_json: str)` - Submit API specification as JSON

**Hypothesis Schema:**
```json
{
  "n_states": 3,
  "start": 0,
  "transitions": {
    "0": {
      "/users": {"GET": [0, 200], "POST": [1, 201]},
      "/items": {"GET": [1, 200]}
    },
    "1": {
      "/users": {"GET": [1, 404], "POST": [0, 201]},
      "/items": {"GET": [0, 200]}
    }
  }
}
```

**Features:**
- State-dependent endpoint behavior
- HTTP status code simulation
- Behavioral equivalence checking

### APIEnv (Stateful SaaS API Reverse Engineering)

A more realistic API identification task: operations are `(method, endpoint, variant)` and outputs include both an HTTP-like status code and a response schema tag.

**Tools:**
- `api_call(method: str, endpoint: str, variant: str)` - Probe the API
- `submit_spec(spec_json: str)` - Submit a full state-dependent spec

**Hypothesis Schema:**
```json
{
  "n_states": 6,
  "start": 0,
  "transitions": {
    "0": {
      "/login": {"POST": {"valid": [1, 200, "AuthOk"], "invalid": [0, 401, "AuthFail"]}}
    }
  }
}
```

## Guide: Designing New Skins

Creating a new skin involves 6 steps. Use `MealyEnv` as a reference implementation.

### Step 1: Define Your Domain

Decide on:
- **Hidden System**: What structure is the agent trying to identify?
- **Probe Action**: How does the agent query the system?
- **Hypothesis Format**: What does a complete submission look like?
- **Equivalence Check**: When are two hypotheses considered equivalent?

| Skin | Hidden System | Probe | Hypothesis |
|------|---------------|-------|------------|
| Mealy | Transition table | Execute symbol | Full transition table |
| Protocol | API state machine | HTTP request | State-dependent transition spec |
| APIEnv | SaaS API workflow | HTTP request + variant | State-dependent transition spec |

### Step 2: Create the Skin File

```python
# dedeucerl/skins/myskin.py
from __future__ import annotations
import json
from typing import Any, Dict, List
from dedeucerl.core.env import HiddenSystemEnv
from dedeucerl.core.config import SkinConfig
from dedeucerl.utils.rng import get_rng

# Configure skin behavior
MY_CONFIG = SkinConfig(
    isomorphism_check=True,    # Enable equivalence checking
    trap_enabled=True,          # Include safety traps
    default_budget=30,          # Default query budget
    max_turns=64,               # Max episode turns
    skin_name="myskin",
    skin_version="1.0",
)

class MySkinEnv(HiddenSystemEnv):
    """My custom skin for [domain description]."""
    
    config = MY_CONFIG
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize skin-specific state
        self._ground_truth = None
        self._trap_pairs = set()
```

### Step 3: Implement Required Abstract Methods

```python
    # REQUIRED: Parse episode metadata
    def _configure_from_metadata(self, meta: Dict[str, Any]) -> None:
        """Extract ground truth from episode metadata."""
        self._ground_truth = meta.get("system")
        self._trap_pairs = set(tuple(x) for x in meta.get("traps", []))
    
    # REQUIRED: Return starting state
    def _get_start_state(self) -> Any:
        """Return the initial state (usually 0)."""
        return 0
    
    # REQUIRED: Return tool methods
    def _get_tools(self) -> List:
        """Return list of tool methods exposed to the agent."""
        return [self.probe, self.submit]
```

### Step 4: Implement Tools

Tools are methods that the LLM agent calls. They must:
- Return JSON strings
- Update internal state
- Track budget and trap hits

```python
    def probe(self, action: str) -> str:
        """Execute a probe action on the hidden system."""
        state = self._state()  # Get episode state dict
        
        if state["done"]:
            return self._episode_finished()

        # Each tool call should consume budget (ends episode at 0).
        if not self._consume_budget(1):
            return self._budget_exhausted()
        
        # Execute action on hidden system
        result = self._execute_on_hidden_system(action)
        
        # Check for traps
        if self._is_trap(state["cs"], action):
            state["trap_hit"] = True
        
        return json.dumps({
            "result": result,
            "budget_left": state["budget"],
            "trap_hit": state["trap_hit"],
            "queries_used": state["queries_used"],
        })
    
    def submit(self, hypothesis_json: str) -> str:
        """Submit a hypothesis about the hidden system."""
        state = self._state()
        
        if state["done"]:
            return self._episode_finished()

        # Every submission attempt consumes budget.
        if not self._consume_budget(self.config.submission_cost):
            return self._budget_exhausted()

        try:
            hypothesis = json.loads(hypothesis_json)
        except Exception:
            return self._tool_error(error_invalid_json("hypothesis_json"), extra={"ok": False})

        # Optional: pre-validate against your DomainSpec.hypothesis_schema.
        # This provides consistent MALFORMED_HYPOTHESIS errors.
        # (Build the schema from your skin's DomainSpec parameters)
        schema = self.__class__.domain_spec(budget=state["budget_init"], trap=True).hypothesis_schema
        validation_err = self._prevalidate_hypothesis(hypothesis, schema)
        if validation_err is not None:
            return self._tool_error(validation_err, extra={"ok": False})

        # Check if hypothesis matches ground truth
        ok = self._check_equivalence(hypothesis)

        if ok:
            state["ok"] = not state["trap_hit"]
            state["done"] = True
        # If incorrect, the episode continues (unless budget is exhausted).        
        payload = {"ok": ok, "budget_left": state["budget"]}
        if not ok and self.feedback_enabled:
            payload["counterexample"] = self._generate_counterexample(hypothesis)
        
        return json.dumps(payload)
```

### Step 5: Implement Static Generation

```python
    @staticmethod
    def generate_system_static(
        seed: int,
        my_param: int = 5,  # Skin-specific parameters
        trap: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a random hidden system.
        
        This must be deterministic given the seed.
        """
        rng = get_rng(seed)  # Seeded RNG for reproducibility
        
        # Generate your hidden system...
        system = generate_random_system(rng, my_param)
        
        # Generate traps if enabled
        traps = generate_traps(rng, system) if trap else []
        
        return {
            "system": system,
            "traps": traps,
        }
    
    @classmethod
    def get_prompt_template(
        cls,
        obs: Dict[str, Any],
        *,
        feedback: bool = False,
    ) -> List[Dict[str, str]]:
        """Build system and user prompts for an episode."""
        return [
            {
                "role": "system",
                "content": "You are an agent identifying a hidden [system type]...\n"
                           "Tools:\n- probe(action) -> result\n- submit(hypothesis) -> ok\n"
            },
            {
                "role": "user", 
                "content": f"OBSERVATION: {json.dumps(obs)}\nRespond only with tool calls."
            },
        ]

    @classmethod
    def domain_spec(cls, my_param: int = 5, budget: int = 25, trap: bool = True) -> "DomainSpec":
        """Single source of truth for tools/observations.

        `TaskGenerator` uses `domain_spec()` to build the observation shown to agents.
        If you populate `params`, `dedeucerl-generate` can expose these without
        hardcoding CLI flags.
        """
        # See `dedeucerl/core/domain_spec.py` for DomainSpec/ParamSpec
        ...
```

### Step 6: Register and Test

```python
# dedeucerl/skins/__init__.py
from .myskin import MySkinEnv

SKIN_REGISTRY = {
    "mealy": MealyEnv,
    "protocol": ProtocolEnv,
    "myskin": MySkinEnv,  # Add your skin
}

__all__ = [..., "MySkinEnv"]
```

Test your skin:

```bash
# Quick test
python -c "
from dedeucerl.skins import MySkinEnv
print(MySkinEnv.generate_system_static(seed=42, my_param=5))
"

# Full test
python -c "
from dedeucerl.skins import MySkinEnv
from dedeucerl.core import TaskGenerator

gen = TaskGenerator(MySkinEnv)
split = gen.generate_split(seeds=[0,1,2], budget=20, subset_name='test', my_param=5)
print(f'Generated {len(split[\"test\"][\"items\"])} episodes')
"
```

---

## Metrics

DedeuceRL tracks the following metrics:

| Metric | Description |
|--------|-------------|
| `success` | Binary: 1 if correct submission without trap, 0 otherwise |
| `queries_used` | Total probe queries consumed |
| `trap_hit` | Binary: 1 if a trap transition was triggered |
| `budget_remaining` | Queries left at episode end |
| `reward` | `1.0 - 0.01 * queries_used` if successful, else 0 |

---

## Split JSON Format

Evaluation splits are JSON files with the following structure:

```json
{
  "dev": {
    "budget": 25,
    "n_states": 3,
    "trap": true,
    "items": [
      {
        "seed": 0,
        "system": {
          "table": {"n": 3, "start": 0, "trans": {...}},
          "trap_pairs": [[0, "A"]]
        }
      },
      {"seed": 1, "system": {...}},
      ...
    ]
  },
  "test": {
    "budget": 40,
    "n_states": 5,
    "trap": true,
    "items": [...]
  }
}
```

---

## Integration with verifiers

DedeuceRL is built on top of [PrimeIntellect's verifiers library](https://github.com/PrimeIntellect-ai/verifiers). The `HiddenSystemEnv` class inherits from `StatefulToolEnv`, enabling:

- Direct compatibility with `verifiers` evaluation pipelines
- Integration with RL training frameworks (GRPO, verl)
- Async episode execution

```python
from verifiers import Rubric
from dedeucerl.skins import MealyEnv

# Use with verifiers evaluation
env = MealyEnv(dataset=dataset, rubric=rubric)
# ... standard verifiers workflow
```

---

## Project Structure

```
DedeuceRL/
├── dedeucerl/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── env.py              # HiddenSystemEnv base class
│   │   ├── config.py           # SkinConfig dataclass
│   │   ├── types.py            # ProbeResult, SubmitResult, EpisodeState
│   │   ├── rubric.py           # Scoring functions
│   │   └── task_generator.py   # TaskGenerator for datasets
│   ├── skins/
│   │   ├── __init__.py         # SKIN_REGISTRY
│   │   └── mealy.py            # MealyEnv implementation
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseAdapter, ModelReply
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   └── gemini.py
│   ├── cli/
│   │   ├── eval.py             # dedeucerl-eval
│   │   ├── aggregate.py        # dedeucerl-aggregate
│   │   └── selfcheck.py        # dedeucerl-selfcheck
│   └── utils/
│       └── rng.py              # Seeded RNG utilities
├── seeds/                      # Evaluation splits
│   ├── mealy_dev.json
│   └── mealy_test.json
├── tests/
│   ├── test_core.py
│   └── test_mealy.py
├── examples/
│   └── basic_eval.py
├── pyproject.toml
└── README.md
```

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

Zenodo DOIs:
- Concept DOI (all versions): `10.5281/zenodo.18280315`
- Version DOI (v1.0.1): `10.5281/zenodo.18280316`

If you use DedeuceRL in your research, please cite:

```bibtex
@software{dedeucerl2026,
  title = {DedeuceRL: A Modular Framework for Active System Identification Benchmarks},
  author = {Vedansh},
  year = {2026},
  doi = {10.5281/zenodo.18280315},
  url = {https://github.com/AashVed/DedeuceRL}
}
```

---

## Acknowledgments

DedeuceRL builds on concepts from:
- DedeuceBench - Original Mealy machine identification benchmark
- [verifiers](https://github.com/PrimeIntellect-ai/verifiers) - PrimeIntellect's RL verification library
- Active automata learning (Angluin's L* algorithm)

---

## Developer TODO Notes

- **SUL adapter:** `core/automata.py` already supports the needed adapter pattern via `get_transition(state, action) -> (next_state, output)` callables. A formal SUL interface is optional; add only if multiple skins start duplicating glue.
- **Conformance testing:** W/Wp-method style conformance testing typically assumes a `reset()` capability (or equivalent) so tests can be run from a known start state.
- **Reset tool:** Not planned for v0 because models already struggle on basic Mealy identification; adding reset/actions expands the action space and can make learning harder. Revisit if the repo sees broad adoption.
