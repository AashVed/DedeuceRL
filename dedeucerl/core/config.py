"""Per-skin configuration for DedeuceRL environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class SkinConfig:
    """Configuration for a specific skin.

    Each skin can override this to customize its behavior.

    Attributes:
        isomorphism_check: Whether to accept equivalent representations.
        isomorphism_fn: Custom isomorphism function (hypothesis, ground_truth) -> bool.
        feedback_enabled: Whether to provide counterexamples on failure.
        counterexample_fn: Custom counterexample generator (hypothesis, ground_truth) -> Any.
        trap_enabled: Whether trap states exist in this domain.
        trap_ends_episode: Whether hitting a trap immediately ends the episode.
        default_budget: Default query budget for episodes.
        submission_cost: Budget cost of any submission attempt.
        max_turns: Maximum tool calls per episode.
        skin_name: Identifier for this skin.
        skin_version: Version string for compatibility tracking.
    """

    # Correctness settings
    isomorphism_check: bool = True
    isomorphism_fn: Optional[Callable[[Any, Any], bool]] = None

    # Feedback settings
    feedback_enabled: bool = False
    counterexample_fn: Optional[Callable[[Any, Any], Any]] = None

    # Safety settings
    trap_enabled: bool = True
    trap_ends_episode: bool = False

    # Budget settings
    default_budget: int = 25
    submission_cost: int = 1

    # Episode limits
    max_turns: int = 64

    # Metadata
    skin_name: str = "unknown"
    skin_version: str = "1.0"
