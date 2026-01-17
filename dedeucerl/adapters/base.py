"""Base adapter for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelReply:
    """Response from an LLM model.

    Attributes:
        content: Raw text content of the reply.
        tool_calls: List of tool call dicts if the model requested tool use.
        finish_reason: Why the model stopped (e.g., 'stop', 'tool_calls').
        usage: Token usage information if available.
    """

    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None


class BaseAdapter(ABC):
    """Abstract base class for LLM adapters.

    Each adapter translates between the DedeuceRL message format
    and the specific API requirements of an LLM provider.
    """

    def __init__(self, model_id: str, **kwargs):
        """Initialize the adapter.

        Args:
            model_id: Model identifier (e.g., 'gpt-4o', 'claude-3-opus').
            **kwargs: Additional provider-specific options.
        """
        self.model_id = model_id
        self.options = kwargs

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        """Send a chat request to the model.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional list of tool schemas for function calling.
            **kwargs: Additional request options.

        Returns:
            ModelReply with the model's response.
        """
        pass

    def reset_conversation(self) -> None:
        """Reset any internal conversation state (optional).

        Some providers (notably the OpenAI Responses API) can maintain
        server-side state across requests. This hook lets the caller ensure
        a fresh conversation between episodes.
        """
        return None

    @property
    def supports_tools(self) -> bool:
        """Whether this adapter supports tool/function calling."""
        return True


def decompose_model_spec(spec: str) -> tuple[str, str]:
    """Decompose a model specification into provider and model ID.

    Args:
        spec: Model spec like 'openai:gpt-4o' or 'anthropic:claude-3-opus'.

    Returns:
        Tuple of (provider, model_id).
    """
    if ":" in spec:
        provider, model_id = spec.split(":", 1)
    else:
        # Default to openai if no provider specified
        provider, model_id = "openai", spec
    return provider, model_id
