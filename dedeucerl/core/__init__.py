"""Shared utilities for DedeuceRL.

The primary extension API now lives in `dedeucerl.kernel`, `dedeucerl.runtime`,
and `dedeucerl.surface`. This module keeps pure automata helpers and scoring
rubrics available from their historical location.
"""

from .automata import (
    CounterexampleTrace,
    TransitionSystem,
    apply_backbone,
    check_behavioral_equivalence,
    check_isomorphism_with_signatures,
    compute_reachable_states,
    compute_state_signatures,
    create_reachability_backbone,
    find_counterexample,
    generate_random_traps,
    is_fully_reachable,
    is_minimal,
    verify_trap_free_path_exists,
)
from .rubric import (
    make_rubric,
    make_train_rubric,
    metric_budget_remaining,
    metric_queries,
    metric_success,
    metric_trap,
    reward_identification,
    reward_train_dense,
)
from .transducers import (
    normalize_transducer_table,
    parse_transducer_transitions,
    transducer_counterexample,
    transducer_table_schema,
    transducer_tables_are_isomorphic,
)

__all__ = [
    "CounterexampleTrace",
    "TransitionSystem",
    "apply_backbone",
    "check_behavioral_equivalence",
    "check_isomorphism_with_signatures",
    "compute_reachable_states",
    "compute_state_signatures",
    "create_reachability_backbone",
    "find_counterexample",
    "generate_random_traps",
    "is_fully_reachable",
    "is_minimal",
    "make_rubric",
    "make_train_rubric",
    "metric_budget_remaining",
    "metric_queries",
    "metric_success",
    "metric_trap",
    "normalize_transducer_table",
    "parse_transducer_transitions",
    "reward_identification",
    "reward_train_dense",
    "transducer_counterexample",
    "transducer_table_schema",
    "transducer_tables_are_isomorphic",
    "verify_trap_free_path_exists",
]
