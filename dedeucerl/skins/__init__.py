"""Skin registry for DedeuceRL."""

from .mealy import MealyEnv
from .protocol import ProtocolEnv
from .apienv import APIEnv

# Registry mapping skin names to classes
SKIN_REGISTRY = {
    "mealy": MealyEnv,
    "protocol": ProtocolEnv,
    "apienv": APIEnv,
}

__all__ = ["SKIN_REGISTRY", "MealyEnv", "ProtocolEnv", "APIEnv"]
