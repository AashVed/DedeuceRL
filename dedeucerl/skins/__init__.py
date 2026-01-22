"""Skin registry for DedeuceRL."""

from .mealy import MealyEnv
from .protocol import ProtocolEnv
from .apienv import APIEnv
from .exprpolicy import ExprPolicyEnv

# Registry mapping skin names to classes
SKIN_REGISTRY = {
    "mealy": MealyEnv,
    "protocol": ProtocolEnv,
    "apienv": APIEnv,
    "exprpolicy": ExprPolicyEnv,
}

__all__ = ["SKIN_REGISTRY", "MealyEnv", "ProtocolEnv", "APIEnv", "ExprPolicyEnv"]
