"""Structured logging for DedeuceRL."""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional


# Module-level logger
_logger: Optional[logging.Logger] = None


def get_logger(name: str = "dedeucerl") -> logging.Logger:
    """Get or create the DedeuceRL logger.

    Args:
        name: Logger name (default: 'dedeucerl').

    Returns:
        Configured logger instance.
    """
    global _logger

    if _logger is None or _logger.name != name:
        _logger = logging.getLogger(name)

        if not _logger.handlers:
            # Default handler: stderr with structured format
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(DedeuceFormatter())
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)

    return _logger


def configure_logging(
    level: str = "INFO",
    format_style: str = "structured",
    include_timestamp: bool = True,
) -> None:
    """Configure DedeuceRL logging.

    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        format_style: 'structured' for key=value or 'simple' for plain text.
        include_timestamp: Whether to include timestamps.

    Example:
        >>> configure_logging(level="DEBUG", format_style="simple")
    """
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Replace handlers with new formatter
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    if format_style == "simple":
        handler.setFormatter(
            logging.Formatter(
                "%(levelname)s: %(message)s"
                if not include_timestamp
                else "%(asctime)s %(levelname)s: %(message)s"
            )
        )
    else:
        handler.setFormatter(DedeuceFormatter(include_timestamp=include_timestamp))

    logger.addHandler(handler)


class DedeuceFormatter(logging.Formatter):
    """Structured log formatter with key=value pairs."""

    def __init__(self, include_timestamp: bool = True):
        self.include_timestamp = include_timestamp
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        parts = []

        if self.include_timestamp:
            parts.append(self.formatTime(record, "%Y-%m-%d %H:%M:%S"))

        parts.append(f"level={record.levelname.lower()}")
        parts.append(f"module={record.module}")

        # Main message
        msg = record.getMessage()
        parts.append(f'msg="{msg}"')

        # Extra fields (structured data)
        extra_keys = set(record.__dict__.keys()) - {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "message",
            "asctime",
        }
        for key in sorted(extra_keys):
            value = getattr(record, key)
            if isinstance(value, str):
                parts.append(f'{key}="{value}"')
            else:
                parts.append(f"{key}={value}")

        return " ".join(parts)


# Convenience functions for structured logging
def log_episode_start(skin: str, seed: int, budget: int, **extra) -> None:
    """Log episode start."""
    logger = get_logger()
    logger.info("Episode started", extra={"skin": skin, "seed": seed, "budget": budget, **extra})


def log_episode_end(
    skin: str, seed: int, success: bool, queries_used: int, trap_hit: bool, **extra
) -> None:
    """Log episode completion."""
    logger = get_logger()
    logger.info(
        "Episode completed",
        extra={
            "skin": skin,
            "seed": seed,
            "success": success,
            "queries_used": queries_used,
            "trap_hit": trap_hit,
            **extra,
        },
    )


def log_tool_call(tool_name: str, args: Dict[str, Any], **extra) -> None:
    """Log a tool call."""
    logger = get_logger()
    logger.debug(f"Tool call: {tool_name}", extra={"tool": tool_name, "args": str(args), **extra})


def log_error(message: str, error_code: str = "UNKNOWN", **extra) -> None:
    """Log an error with structured context."""
    logger = get_logger()
    logger.error(message, extra={"error_code": error_code, **extra})


def log_warning(message: str, **extra) -> None:
    """Log a warning with structured context."""
    logger = get_logger()
    logger.warning(message, extra=extra)
