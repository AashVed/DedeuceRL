"""Core abstractions for DedeuceRL.

This module provides the foundational components for active system identification:

1. **Environment Base Class**: `HiddenSystemEnv` - the abstract base for all skins
2. **Configuration**: `SkinConfig` - per-skin settings
3. **Types**: `ProbeResult`, `SubmitResult`, `EpisodeState` - structured data types
4. **Rubric**: Scoring functions for RL training
5. **Task Generation**: `TaskGenerator` for reproducible dataset creation
6. **Domain Specification**: Schema-first tool and observation definitions
7. **Automata Algorithms**: Reachability, minimality, isomorphism, counterexamples

The automata module provides domain-agnostic algorithms that skins can leverage
instead of reimplementing from scratch.
"""

from .env import HiddenSystemEnv
from .types import ProbeResult, SubmitResult, EpisodeState
from .config import SkinConfig
from .rubric import (
    make_rubric,
    reward_identification,
    metric_success,
    metric_queries,
    metric_trap,
    metric_budget_remaining,
)
from .task_generator import TaskGenerator
from .domain_spec import (
    DomainSpec,
    ToolSchema,
    ArgSchema,
    ReturnField,
    ObservationField,
    ParamSpec,
)
from .automata import (
    TransitionSystem,
    CounterexampleTrace,
    compute_reachable_states,
    is_fully_reachable,
    compute_state_signatures,
    is_minimal,
    check_behavioral_equivalence,
    check_isomorphism_with_signatures,
    find_counterexample,
    generate_random_traps,
    verify_trap_free_path_exists,
    create_reachability_backbone,
    apply_backbone,
)

__all__ = [
    # Environment
    "HiddenSystemEnv",
    # Types
    "ProbeResult",
    "SubmitResult",
    "EpisodeState",
    # Configuration
    "SkinConfig",
    # Rubric
    "make_rubric",
    "reward_identification",
    "metric_success",
    "metric_queries",
    "metric_trap",
    "metric_budget_remaining",
    # Task Generation
    "TaskGenerator",
    # Domain Specification
    "DomainSpec",
    "ToolSchema",
    "ArgSchema",
    "ReturnField",
    "ObservationField",
    "ParamSpec",
    # Automata Algorithms
    "TransitionSystem",
    "CounterexampleTrace",
    "compute_reachable_states",
    "is_fully_reachable",
    "compute_state_signatures",
    "is_minimal",
    "check_behavioral_equivalence",
    "check_isomorphism_with_signatures",
    "find_counterexample",
    "generate_random_traps",
    "verify_trap_free_path_exists",
    "create_reachability_backbone",
    "apply_backbone",
]
