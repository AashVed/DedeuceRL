"""Error types and helpers for DedeuceRL.

This module defines a small, stable set of machine-readable error codes.
Skins should prefer these codes for *framework-level* failures (budget, episode
state, JSON parsing), while still being free to include domain-specific fields
in tool payloads (e.g., HTTP-like status codes).

Important: Tool methods should return a *standard envelope* that includes:
- budget_left
- queries_used
- trap_hit
- error: {code, message, details?}

The envelope itself is assembled in `HiddenSystemEnv` so errors stay consistent
across all skins.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Standard error codes for DedeuceRL."""

    # Episode errors
    EPISODE_FINISHED = "E001"
    BUDGET_EXHAUSTED = "E002"
    STATE_NOT_INITIALIZED = "E003"

    # Tool / parsing errors
    INVALID_ARGUMENT = "E101"
    INVALID_JSON = "E102"
    UNKNOWN_TOOL = "E103"

    # Skin / domain errors (optional)
    INVALID_SYMBOL = "E201"
    INVALID_METHOD = "E203"
    INVALID_ENDPOINT = "E204"
    ENDPOINT_NOT_FOUND = "E205"
    METHOD_NOT_ALLOWED = "E206"

    # Submission errors
    MALFORMED_HYPOTHESIS = "E301"
    INCORRECT_HYPOTHESIS = "E302"

    # System errors
    SKIN_NOT_FOUND = "E401"
    SPLIT_NOT_FOUND = "E402"
    ADAPTER_ERROR = "E403"


@dataclass(frozen=True)
class DedeuceError:
    """Structured error object.

    This object is intentionally *not* responsible for including per-episode
    state fields (budget, trap flags, etc.). That is handled by the environment
    so the envelope stays uniform.
    """

    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": str(self.code.value), "message": self.message}
        if self.details is not None:
            out["details"] = self.details
        return out


# -----------------------------------------------------------------------------
# Error factories
# -----------------------------------------------------------------------------


def error_episode_finished() -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.EPISODE_FINISHED,
        message="Episode already finished. No more actions allowed.",
    )


def error_budget_exhausted() -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.BUDGET_EXHAUSTED,
        message="Budget exhausted. No queries remaining.",
    )


def error_invalid_symbol(symbol: str, valid: list) -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.INVALID_SYMBOL,
        message=f"Invalid symbol '{symbol}'. Must be one of: {valid}",
        details={"received": symbol, "valid": valid},
    )


def error_invalid_method(method: str, valid: list) -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.INVALID_METHOD,
        message=f"Invalid method '{method}'. Must be one of: {valid}",
        details={"received": method, "valid": valid},
    )


def error_endpoint_not_found(endpoint: str) -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.ENDPOINT_NOT_FOUND,
        message=f"Endpoint '{endpoint}' not found.",
        details={"endpoint": endpoint},
    )


def error_method_not_allowed(method: str, endpoint: str, allowed: list) -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.METHOD_NOT_ALLOWED,
        message=f"Method '{method}' not allowed for '{endpoint}'. Allowed: {allowed}",
        details={"method": method, "endpoint": endpoint, "allowed": allowed},
    )


def error_invalid_json(context: str = "payload") -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.INVALID_JSON,
        message=f"Failed to parse {context} as valid JSON.",
        details={"context": context},
    )


def error_unknown_tool(tool: str, available: Optional[list] = None) -> DedeuceError:
    details: Dict[str, Any] = {"tool": tool}
    if available is not None:
        details["available"] = list(available)
    return DedeuceError(
        code=ErrorCode.UNKNOWN_TOOL,
        message=f"Unknown tool '{tool}'.",
        details=details,
    )


def error_invalid_argument(
    message: str, *, details: Optional[Dict[str, Any]] = None
) -> DedeuceError:
    """Invalid tool arguments (framework-level error)."""
    return DedeuceError(code=ErrorCode.INVALID_ARGUMENT, message=message, details=details)


def error_malformed_hypothesis(reason: str) -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.MALFORMED_HYPOTHESIS,
        message=f"Malformed hypothesis: {reason}",
        details={"reason": reason},
    )


def error_skin_not_found(skin: str, available: list) -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.SKIN_NOT_FOUND,
        message=f"Skin '{skin}' not found. Available: {available}",
        details={"skin": skin, "available": available},
    )


def error_split_not_found(path: str) -> DedeuceError:
    return DedeuceError(
        code=ErrorCode.SPLIT_NOT_FOUND,
        message=f"Split file not found: {path}",
        details={"path": path},
    )
