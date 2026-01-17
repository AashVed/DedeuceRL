"""Rubric functions for active system identification scoring."""

from __future__ import annotations

from typing import Any, Dict

from verifiers import Rubric
from verifiers.parsers.parser import Parser


def reward_identification(
    completion: Any,
    answer: str,
    state: Dict[str, Any],
    parser: Parser,
    **kwargs,
) -> float:
    """Default reward function for active identification.

    Returns:
        - 1.0 for success without trap (minus small efficiency penalty)
        - 0.0 for failure or trap
    """
    ok = bool(state.get("ok", False))
    trap = bool(state.get("trap_hit", False))
    queries = int(state.get("queries_used", 0))

    if not ok or trap:
        return 0.0

    # Base reward with efficiency penalty
    base = 1.0
    efficiency_penalty = min(0.5, 0.01 * queries)
    return float(max(0.0, base - efficiency_penalty))


def metric_success(
    completion: Any,
    answer: str,
    state: Dict[str, Any],
    parser: Parser,
    **kwargs,
) -> float:
    """Binary success metric: 1.0 if ok, 0.0 otherwise."""
    return 1.0 if state.get("ok", False) else 0.0


def metric_queries(
    completion: Any,
    answer: str,
    state: Dict[str, Any],
    parser: Parser,
    **kwargs,
) -> float:
    """Number of queries used metric."""
    return float(state.get("queries_used", 0))


def metric_trap(
    completion: Any,
    answer: str,
    state: Dict[str, Any],
    parser: Parser,
    **kwargs,
) -> float:
    """Trap hit metric: 1.0 if trap hit, 0.0 otherwise."""
    return 1.0 if state.get("trap_hit", False) else 0.0


def metric_budget_remaining(
    completion: Any,
    answer: str,
    state: Dict[str, Any],
    parser: Parser,
    **kwargs,
) -> float:
    """Remaining budget metric."""
    return float(state.get("budget", 0))


def make_rubric() -> Rubric:
    """Create the standard rubric for active identification.

    Returns:
        Rubric with reward_identification as the primary function
        and additional metrics for analysis.
    """
    return Rubric(
        funcs=[
            reward_identification,
            metric_success,
            metric_queries,
            metric_trap,
            metric_budget_remaining,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
        parser=Parser(extract_fn=lambda s: s),
    )
