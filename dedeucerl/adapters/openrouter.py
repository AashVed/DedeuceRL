"""OpenRouter adapter for DedeuceRL.

OpenRouter provides an OpenAI-compatible API surface, but we keep a dedicated
adapter to avoid OpenAI-specific behavior (e.g., the Responses API) leaking into
OpenRouter runs and to support OpenRouter-specific headers/config cleanly.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from .base import BaseAdapter, ModelReply


class OpenRouterAdapter(BaseAdapter):
    """Adapter for OpenRouter's OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self._client = None

    @property
    def client(self):
        """Lazy-load the OpenAI client (configured to point at OpenRouter)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "OpenRouterAdapter requires the `openai` package. Install with: "
                    "pip install 'dedeucerl[openai]'"
                ) from e

            kwargs: Dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.default_headers:
                kwargs["default_headers"] = dict(self.default_headers)

            self._client = OpenAI(**kwargs)
        return self._client

    def _prepare_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build a chat.completions request dict (pure; easy to unit test)."""
        request: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            **self.options,
            **kwargs,
        }

        # OpenRouter accepts provider-agnostic fields (e.g., reasoning) via the
        # OpenAI SDK's `extra_body` passthrough, since the SDK can reject unknown
        # top-level kwargs.
        extra_body: Dict[str, Any] = {}
        if "extra_body" in request and isinstance(request["extra_body"], dict):
            extra_body.update(request["extra_body"])

        reasoning = request.pop("reasoning", None)
        if reasoning is not None:
            extra_body["reasoning"] = reasoning

        if extra_body:
            request["extra_body"] = extra_body

        if tools:
            request["tools"] = self._format_tools(tools)
            if "tool_choice" not in request:
                request["tool_choice"] = "required"

        return request

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        request = self._prepare_request(messages, tools=tools, **kwargs)

        response: Any = None
        last_exc: Exception | None = None

        for _ in range(3):
            try:
                response = self.client.chat.completions.create(**request)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                msg = str(e)

                # Some backends/models use `max_completion_tokens` instead of `max_tokens`.
                if (
                    "max_tokens" in request
                    and "max_tokens" in msg
                    and "max_completion_tokens" in msg
                ):
                    request["max_completion_tokens"] = request.pop("max_tokens")
                    continue

                # Some models restrict or forbid temperature.
                if (
                    "temperature" in request
                    and "temperature" in msg
                    and (
                        "Only the default" in msg
                        or "not supported" in msg
                        or "Unsupported parameter" in msg
                    )
                ):
                    request.pop("temperature", None)
                    continue

                raise

        if last_exc is not None:
            raise last_exc

        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if getattr(message, "tool_calls", None):
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ModelReply(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
        )

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI Chat Completions style."""
        return [{"type": "function", "function": tool} for tool in tools]
