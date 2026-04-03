"""Model adapters for DedeuceRL."""

from __future__ import annotations

import os

from .base import BaseAdapter, ModelReply, decompose_model_spec
from .openai import OpenAIAdapter
from .openrouter import OpenRouterAdapter
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter
from .heuristic import HeuristicAdapter

# Registry mapping provider names to adapter classes
ADAPTER_REGISTRY = {
    "openai": OpenAIAdapter,
    "openrouter": OpenRouterAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "google": GeminiAdapter,
    "heuristic": HeuristicAdapter,
    "dummy": HeuristicAdapter,
}


def get_adapter(model_spec: str, **kwargs) -> BaseAdapter:
    """Get an adapter for a model specification.

    Args:
        model_spec: Model spec like 'openai:gpt-4o' or 'anthropic:claude-3-opus'.
        **kwargs: Additional options passed to the adapter.

    Returns:
        Configured adapter instance.

    Raises:
        ValueError: If the provider is not supported.
    """
    provider, model_id = decompose_model_spec(model_spec)

    if provider not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {list(ADAPTER_REGISTRY.keys())}"
        )

    # Provider-specific configuration (avoid OpenAI/OpenRouter env interference).
    if provider == "openai":
        if "api_key" not in kwargs:
            # The OpenAI SDK defaults to OPENAI_API_KEY; we pass explicitly for clarity.
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                kwargs["api_key"] = env_key
        if "base_url" not in kwargs:
            env_base_url = os.getenv("OPENAI_BASE_URL")
            if env_base_url:
                kwargs["base_url"] = env_base_url

    if provider == "openrouter":
        if "api_key" not in kwargs:
            # Prefer OPENROUTER_API_KEY; fall back to OPENAI_API_KEY for convenience.
            kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

        if "base_url" not in kwargs:
            kwargs["base_url"] = os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"

        if "default_headers" not in kwargs:
            headers = {}
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            title = os.getenv("OPENROUTER_X_TITLE")
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
            if headers:
                kwargs["default_headers"] = headers

    adapter_cls = ADAPTER_REGISTRY[provider]
    return adapter_cls(model_id, **kwargs)


__all__ = [
    "BaseAdapter",
    "ModelReply",
    "OpenAIAdapter",
    "OpenRouterAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "HeuristicAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
    "decompose_model_spec",
]
