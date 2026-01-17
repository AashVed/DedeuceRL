"""Core types for DedeuceRL environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProbeResult:
    """Result of a probe action (for internal use).

    Attributes:
        observation: Domain-specific observation returned by probe.
        budget_remaining: Queries left after this action.
        step: Current step index (1-based).
        trap_hit: Whether a safety violation has occurred.
        metadata: Additional skin-specific data.
    """

    observation: Any
    budget_remaining: int
    step: int
    trap_hit: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubmitResult:
    """Result of a hypothesis submission (for internal use).

    Attributes:
        correct: Whether the hypothesis matched ground truth.
        budget_remaining: Queries left after this submission.
        queries_used: Total queries consumed in episode.
        trap_hit: Whether a safety violation has occurred.
        counterexample: Distinguishing trace if incorrect and feedback enabled.
        metadata: Additional skin-specific data.
    """

    correct: bool
    budget_remaining: int
    queries_used: int
    trap_hit: bool
    counterexample: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeState:
    """Episode state tracking for internal skin use.

    Note: The actual state in verifiers is Dict[str, Any]. This dataclass
    provides named access for skins that want type-safe state management.

    Attributes:
        current_state: Current state in the hidden system.
        budget: Remaining query budget.
        budget_init: Initial budget at episode start.
        queries_used: Total queries consumed.
        steps: Total steps taken.
        trap_hit: Whether a trap was triggered.
        ok: Success flag (True if correct submission without trap).
        done: Episode termination flag.
        ground_truth: Hidden system specification (for skin use only).
    """

    current_state: Any
    budget: int
    budget_init: int
    queries_used: int = 0
    steps: int = 0
    trap_hit: bool = False
    ok: bool = False
    done: bool = False
    ground_truth: Any = None
