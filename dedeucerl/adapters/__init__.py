"""Model adapters for DedeuceRL."""

from .base import BaseAdapter, ModelReply, decompose_model_spec
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter
from .heuristic import HeuristicAdapter

# Registry mapping provider names to adapter classes
ADAPTER_REGISTRY = {
    "openai": OpenAIAdapter,
    "openrouter": OpenAIAdapter,  # OpenRouter is OpenAI-compatible
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

    adapter_cls = ADAPTER_REGISTRY[provider]
    return adapter_cls(model_id, **kwargs)


__all__ = [
    "BaseAdapter",
    "ModelReply",
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "HeuristicAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
    "decompose_model_spec",
]
