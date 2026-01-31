"""Utility functions for DedeuceRL."""

from .rng import get_rng
from .logging import (
    get_logger,
    configure_logging,
    log_episode_start,
    log_episode_end,
    log_tool_call,
    log_error,
    log_warning,
)
from .errors import (
    ErrorCode,
    DedeuceError,
    error_episode_finished,
    error_budget_exhausted,
    error_invalid_symbol,
    error_invalid_method,
    error_endpoint_not_found,
    error_method_not_allowed,
    error_invalid_json,
    error_unknown_tool,
    error_invalid_argument,
    error_malformed_hypothesis,
    error_skin_not_found,
    error_split_not_found,
)
from .episodes import parse_index_spec, parse_shard, apply_shard, compute_split_hash

__all__ = [
    # RNG
    "get_rng",
    # Logging
    "get_logger",
    "configure_logging",
    "log_episode_start",
    "log_episode_end",
    "log_tool_call",
    "log_error",
    "log_warning",
    # Errors
    "ErrorCode",
    "DedeuceError",
    "error_episode_finished",
    "error_budget_exhausted",
    "error_invalid_symbol",
    "error_invalid_method",
    "error_endpoint_not_found",
    "error_method_not_allowed",
    "error_invalid_json",
    "error_unknown_tool",
    "error_invalid_argument",
    "error_malformed_hypothesis",
    "error_skin_not_found",
    "error_split_not_found",
    # Episodes
    "parse_index_spec",
    "parse_shard",
    "apply_shard",
    "compute_split_hash",
]
