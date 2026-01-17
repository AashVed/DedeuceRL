"""Anthropic adapter for DedeuceRL."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAdapter, ModelReply


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude API.

    Notes on tool calling
    ---------------------
    DedeuceRL keeps a provider-agnostic transcript where:
    - Assistant tool requests are stored as OpenAI-style `tool_calls` on an assistant message.
    - Tool results are stored as messages with `role="tool"`, `tool_call_id`, and `content`.

    Anthropic expects tool round-trips encoded as:
    - Assistant message content blocks of type `tool_use`
    - User message content blocks of type `tool_result`

    This adapter translates between these formats so multi-turn tool episodes work.
    """

    def __init__(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Anthropic adapter.

        Args:
            model_id: Model name (e.g., 'claude-3-opus-20240229').
            api_key: API key. If None, uses ANTHROPIC_API_KEY env var.
            **kwargs: Additional options.
        """
        super().__init__(model_id, **kwargs)
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "AnthropicAdapter requires the `anthropic` package. Install with: pip install 'dedeucerl[anthropic]'"
                ) from e

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = Anthropic(**kwargs)
        return self._client

    def _to_anthropic_conversation(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Translate canonical DedeuceRL messages to Anthropic message format."""
        system_content = ""
        conversation: List[Dict[str, Any]] = []

        # Map tool_call_id -> tool name so we can attach tool_result blocks.
        tool_name_by_id: Dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                system_content = msg.get("content") or ""
                continue

            if role == "user":
                conversation.append({"role": "user", "content": msg.get("content") or ""})
                continue

            if role == "assistant":
                blocks: List[Dict[str, Any]] = []

                text = msg.get("content")
                if isinstance(text, str) and text:
                    blocks.append({"type": "text", "text": text})

                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    fn = (tc or {}).get("function", {})
                    name = fn.get("name")
                    raw_args = fn.get("arguments", "{}")

                    try:
                        inp = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        inp = {"_raw_arguments": raw_args}

                    tc_id = (tc or {}).get("id") or ""
                    if not tc_id:
                        tc_id = f"call_{len(tool_name_by_id) + 1}"

                    if name:
                        tool_name_by_id[tc_id] = name
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": name,
                                "input": inp if isinstance(inp, dict) else {"input": inp},
                            }
                        )

                # Skip empty assistant messages (Anthropic can reject them)
                if blocks:
                    conversation.append({"role": "assistant", "content": blocks})
                continue

            if role == "tool":
                tool_use_id = msg.get("tool_call_id") or msg.get("tool_use_id") or ""
                result_content = msg.get("content")
                if result_content is None:
                    result_content = ""

                tool_result_block: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                }
                tool_name = tool_name_by_id.get(tool_use_id)
                if tool_name:
                    tool_result_block["name"] = tool_name

                conversation.append({"role": "user", "content": [tool_result_block]})
                continue

        return system_content, conversation

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        """Send a chat request to Anthropic."""
        system_content, conversation = self._to_anthropic_conversation(messages)

        request = {
            "model": self.model_id,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "messages": conversation,
            **self.options,
            **kwargs,
        }

        if system_content:
            request["system"] = system_content

        if tools:
            request["tools"] = self._format_tools(tools)

        response = self.client.messages.create(**request)

        # Extract content and tool calls
        content = None
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                # Keep the last text block if multiple appear
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )

        # Extract usage
        usage = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return ModelReply(
            content=content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage=usage,
        )

    def _format_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format tools for Anthropic API."""
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            }
            for tool in tools
        ]
