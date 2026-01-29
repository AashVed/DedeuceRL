# Creating New Skins

This guide covers how to create custom skins for DedeuceRL. Use [`MealyEnv`](../dedeucerl/skins/mealy.py) as the reference implementation.

## Overview

Skins are domain-specific implementations of the active identification paradigm. Each skin defines:

1. **Hidden System** — The structure the agent must identify
2. **Probe Actions** — How the agent queries the system
3. **Hypothesis Format** — What a complete submission looks like
4. **Equivalence Check** — When two hypotheses are considered equivalent

| Skin | Hidden System | Probe | Hypothesis |
|------|---------------|-------|------------|
| Mealy | Transition table | Execute symbol | Full transition table |
| Protocol | API state machine | HTTP request | State-dependent spec |
| APIEnv | SaaS API workflow | HTTP + variant | State-dependent spec |
| ExprPolicy | Typed policy DSL | Compile + tests | Corrected expression |

---

## Step 1: Create the Skin File

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
    """Custom skin for [domain description]."""
    
    config = MY_CONFIG
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ground_truth = None
        self._trap_pairs = set()
```

---

## Step 2: Implement Required Methods

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

---

## Step 3: Implement Tools

Tools are methods the LLM agent calls. They must:
- Return JSON strings
- Update internal state
- Track budget and trap hits

```python
def probe(self, action: str) -> str:
    """Execute a probe action on the hidden system."""
    state = self._state()
    
    if state["done"]:
        return self._episode_finished()

    if not self._consume_budget(1):
        return self._budget_exhausted()
    
    result = self._execute_on_hidden_system(action)
    
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

    if not self._consume_budget(self.config.submission_cost):
        return self._budget_exhausted()

    try:
        hypothesis = json.loads(hypothesis_json)
    except Exception:
        return self._tool_error(error_invalid_json("hypothesis_json"), extra={"ok": False})

    # Pre-validate hypothesis schema
    schema = self.__class__.domain_spec(budget=state["budget_init"], trap=True).hypothesis_schema
    validation_err = self._prevalidate_hypothesis(hypothesis, schema)
    if validation_err is not None:
        return self._tool_error(validation_err, extra={"ok": False})

    ok = self._check_equivalence(hypothesis)

    if ok:
        state["ok"] = not state["trap_hit"]
        state["done"] = True
        
    payload = {"ok": ok, "budget_left": state["budget"]}
    if not ok and self.feedback_enabled:
        payload["counterexample"] = self._generate_counterexample(hypothesis)
    
    return json.dumps(payload)
```

---

## Step 4: Implement Static Generation

```python
@staticmethod
def generate_system_static(
    seed: int,
    my_param: int = 5,
    trap: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Generate a random hidden system (deterministic given seed)."""
    rng = get_rng(seed)
    
    system = generate_random_system(rng, my_param)
    traps = generate_traps(rng, system) if trap else []
    
    return {"system": system, "traps": traps}

@classmethod
def get_prompt_template(
    cls,
    obs: Dict[str, Any],
    *,
    feedback: bool = False,
) -> List[Dict[str, str]]:
    """Build system and user prompts for an episode."""
    return [
        {"role": "system", "content": "You are identifying a hidden [system]..."},
        {"role": "user", "content": f"OBSERVATION: {json.dumps(obs)}"},
    ]

@classmethod
def domain_spec(cls, my_param: int = 5, budget: int = 25, trap: bool = True) -> "DomainSpec":
    """Single source of truth for tools/observations.
    
    TaskGenerator uses domain_spec() to build observations.
    Populate `params` to expose skin-specific CLI flags.
    """
    # See dedeucerl/core/domain_spec.py for DomainSpec/ParamSpec
    ...
```

---

## Step 5: Register and Test

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

# Validate installation
dedeucerl-selfcheck --verbose
```

---

## Checklist

- [ ] Inherits from `HiddenSystemEnv`
- [ ] Has `config = SkinConfig(...)`
- [ ] Implements `_configure_from_metadata`, `_get_start_state`, `_get_tools`
- [ ] Implements probe tool(s) that consume budget
- [ ] Implements submit tool with equivalence checking
- [ ] Implements `generate_system_static` (deterministic)
- [ ] Implements `get_prompt_template`
- [ ] Implements `domain_spec`
- [ ] Registered in `SKIN_REGISTRY`
- [ ] Passes `dedeucerl-selfcheck`
