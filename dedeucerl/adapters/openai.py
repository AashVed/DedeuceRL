"""OpenAI adapter for DedeuceRL."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import BaseAdapter, ModelReply


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI and OpenAI-compatible APIs (e.g., OpenRouter)."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize OpenAI adapter.

        Args:
            model_id: Model name (e.g., 'gpt-4o', 'gpt-4-turbo').
            api_key: API key. If None, uses OPENAI_API_KEY env var.
            base_url: Optional base URL for OpenAI-compatible APIs.
            **kwargs: Additional options passed to client.
        """
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

        # Stateful Responses API bookkeeping (best-effort).
        self._responses_prev_id: Optional[str] = None
        self._responses_cursor: int = 0
        self._responses_fingerprint: Optional[str] = None

    def reset_conversation(self) -> None:
        # Always reset any server-side conversation linkage.
        self._responses_prev_id = None
        self._responses_cursor = 0
        self._responses_fingerprint = None

    @property
    def client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "OpenAIAdapter requires the `openai` package. Install with: pip install 'dedeucerl[openai]'"
                ) from e

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        """Send a chat/tool request to OpenAI."""
        # GPT-5 family is most reliable via the Responses API.
        if self.model_id.startswith("gpt-5"):
            return self._chat_responses(messages, tools=tools, **kwargs)

        request = {
            "model": self.model_id,
            "messages": messages,
            **self.options,
            **kwargs,
        }

        if tools:
            request["tools"] = self._format_tools(tools)

            # Default to requiring tool calls when tools are provided.
            # This makes benchmark runs much more reliable (models won't "chat" instead of acting).
            if "tool_choice" not in request:
                request["tool_choice"] = "required"

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

                # Some newer OpenAI models use `max_completion_tokens` instead of `max_tokens`.
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

        # Extract tool calls
        tool_calls = None
        if message.tool_calls:
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

        # Extract usage
        usage = None
        if response.usage:
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

    def _to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert canonical messages to Responses API input items."""
        items: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role in ("system", "developer", "user", "assistant"):
                content = msg.get("content")
                if content is None:
                    content = ""

                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content,
                    }
                )

                # Preserve past tool calls so tool outputs have a matching call_id.
                if role == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        fn = (tc or {}).get("function", {})
                        fc_id = (tc or {}).get("id") or ""
                        call_id = (tc or {}).get("call_id") or fc_id
                        items.append(
                            {
                                "type": "function_call",
                                "id": fc_id,
                                "call_id": call_id,
                                "name": fn.get("name") or "",
                                "arguments": fn.get("arguments") or "{}",
                                "status": "completed",
                            }
                        )
                continue

            if role == "tool":
                call_id = msg.get("tool_call_id") or ""
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": msg.get("content") or "",
                        "status": "completed",
                    }
                )

        return items

    def _to_responses_input_delta(
        self, messages: List[Dict[str, Any]], start_idx: int
    ) -> List[Dict[str, Any]]:
        """Build only the *new* input items since `start_idx`.

        When using `previous_response_id`, we should not replay assistant
        tool-call items (they require provider-specific reasoning items).
        """
        items: List[Dict[str, Any]] = []
        saw_tool_output = False
        for msg in messages[start_idx:]:
            role = msg.get("role")
            if role in ("developer", "user"):
                items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": msg.get("content") or "",
                    }
                )
            elif role == "tool":
                call_id = msg.get("tool_call_id") or ""
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": msg.get("content") or "",
                        "status": "completed",
                    }
                )
                saw_tool_output = True

        # Some Responses API backends behave better if tool outputs are
        # followed by an explicit user "continue".
        if saw_tool_output:
            items.append({"type": "message", "role": "user", "content": "continue"})

        return items

    def _format_tools_responses(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Responses API."""
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "strict": False,
            }
            for tool in tools
        ]

    def _chat_responses(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        """Send a request via the OpenAI Responses API."""
        params: Dict[str, Any] = {
            **self.options,
            **kwargs,
        }

        # Normalize token limit options.
        max_output = params.pop("max_output_tokens", None)
        max_completion = params.pop("max_completion_tokens", None)
        max_tokens = params.pop("max_tokens", None)
        if max_output is None:
            if max_completion is not None:
                max_output = max_completion
            elif max_tokens is not None:
                max_output = max_tokens
        if max_output is not None:
            params["max_output_tokens"] = max_output

        fingerprint = json.dumps(
            [
                (
                    messages[0].get("role"),
                    messages[0].get("content"),
                )
                if len(messages) > 0
                else None,
                (
                    messages[1].get("role"),
                    messages[1].get("content"),
                )
                if len(messages) > 1
                else None,
            ],
            ensure_ascii=False,
        )

        if fingerprint != self._responses_fingerprint:
            self._responses_prev_id = None
            self._responses_cursor = 0
            self._responses_fingerprint = fingerprint

        input_items = self._to_responses_input_delta(messages, self._responses_cursor)

        request: Dict[str, Any] = {
            "model": self.model_id,
            "input": input_items,
            **params,
        }

        # Prefer `instructions` for the system prompt in Responses API.
        if (
            not self._responses_prev_id
            and len(messages) > 0
            and messages[0].get("role") == "system"
        ):
            request["instructions"] = messages[0].get("content") or ""

        if self._responses_prev_id:
            request["previous_response_id"] = self._responses_prev_id

        if tools:
            request["tools"] = self._format_tools_responses(tools)
            if "tool_choice" not in request:
                request["tool_choice"] = "required"

        response: Any = None
        last_exc: Exception | None = None

        for _ in range(3):
            try:
                response = self.client.responses.create(**request)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                msg = str(e)

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

        # Update conversation cursor/id for next request.
        self._responses_cursor = len(messages)
        resp_id = getattr(response, "id", None)
        if isinstance(resp_id, str) and resp_id:
            self._responses_prev_id = resp_id

        # Extract tool calls
        tool_calls: Optional[List[Dict[str, Any]]] = None
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": getattr(item, "id", ""),
                        "call_id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(item, "name", ""),
                            "arguments": getattr(item, "arguments", "{}"),
                        },
                    }
                )

        # Extract text content
        content: Optional[str] = None
        out_text = getattr(response, "output_text", None)
        if callable(out_text):
            val = out_text()
            if isinstance(val, str):
                content = val
            elif val is not None:
                content = str(val)
        elif isinstance(out_text, str):
            content = out_text

        usage = None
        resp_usage = getattr(response, "usage", None)
        if resp_usage is not None:
            try:
                usage = {
                    "prompt_tokens": int(getattr(resp_usage, "input_tokens", 0)),
                    "completion_tokens": int(getattr(resp_usage, "output_tokens", 0)),
                    "total_tokens": int(getattr(resp_usage, "total_tokens", 0)),
                }
            except Exception:
                usage = None

        return ModelReply(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(response, "status", "stop") or "stop",
            usage=usage,
        )

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for OpenAI Chat Completions API."""
        return [
            {
                "type": "function",
                "function": tool,
            }
            for tool in tools
        ]
