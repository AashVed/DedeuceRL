"""Utility functions for seeded random number generation."""

from __future__ import annotations

import random


def get_rng(seed: int) -> random.Random:
    """Create a seeded random number generator.

    Args:
        seed: Integer seed for reproducibility.

    Returns:
        A random.Random instance seeded with the given value.
    """
    rng = random.Random()
    rng.seed(seed)
    return rng
