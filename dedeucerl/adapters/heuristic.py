"""Offline heuristic adapter.

This adapter exists for two reasons:
1) Provide a zero-dependency baseline (`heuristic:none`) for first-time users.
2) Enable CI / local smoke tests without API keys.

It is deliberately simple: it reads the tool schemas (including enums) and
constructs valid-looking tool calls. It does not attempt to solve the task.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import BaseAdapter, ModelReply


class HeuristicAdapter(BaseAdapter):
    """A tiny baseline adapter that emits tool calls from schemas."""

    @property
    def supports_tools(self) -> bool:  # pragma: no cover
        return True

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        # Pick a "probe" tool if available; otherwise pick the first tool.
        tool_schemas = tools or []
        if not tool_schemas:
            return ModelReply(content=None, tool_calls=None, finish_reason="stop", usage=None)

        probe = None
        submit = None
        for t in tool_schemas:
            name = t.get("name", "")
            if name.startswith("submit"):
                submit = t
            else:
                probe = probe or t

        chosen = probe or submit or tool_schemas[0]

        # If we see many tool results already, try submitting to end sooner.
        tool_results_seen = sum(1 for m in messages if m.get("role") == "tool")
        if submit is not None and tool_results_seen >= 5:
            chosen = submit

        tc = {
            "id": f"call_{tool_results_seen + 1}",
            "type": "function",
            "function": {
                "name": chosen.get("name", ""),
                "arguments": json.dumps(
                    self._default_args_from_schema(chosen.get("parameters", {}))
                ),
            },
        }

        return ModelReply(content=None, tool_calls=[tc], finish_reason="tool_calls", usage=None)

    def _default_args_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not schema or schema.get("type") != "object":
            return {}

        out: Dict[str, Any] = {}
        props = schema.get("properties", {}) or {}
        for name, prop in props.items():
            if not isinstance(prop, dict):
                continue

            # Prefer enum first.
            enum = prop.get("enum")
            if isinstance(enum, list) and enum:
                out[name] = enum[0]
                continue

            t = prop.get("type", "string")
            if t == "integer":
                out[name] = 0
            elif t == "number":
                out[name] = 0.0
            elif t == "boolean":
                out[name] = False
            elif t == "array":
                out[name] = []
            elif t == "object":
                out[name] = {}
            else:
                out[name] = ""

        return out
