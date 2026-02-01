"""Google Gemini adapter for DedeuceRL (google-genai SDK only).

We intentionally support only the official, actively maintained Google Gen AI
Python SDK (`google-genai`, imported as `google.genai`).

The older `google-generativeai` package is EOL upstream and is not supported
here to avoid confusing warnings and brittle behavior for practitioners.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import BaseAdapter, ModelReply


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini tool calling APIs (google-genai)."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key

    def _import_backend(self) -> Any:
        """Import the supported Gemini client backend (google-genai)."""
        try:
            import google.genai as genai_new  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "GeminiAdapter requires the supported `google-genai` SDK (import: google.genai). "
                "Install with: pip install 'dedeucerl[gemini]'."
            ) from e
        return genai_new

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        genai = self._import_backend()
        return self._chat_genai(genai, messages, tools=tools, **kwargs)

    def _chat_genai(
        self,
        genai: Any,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        """Chat via `google.genai` (official Gemini SDK).

        Implementation notes:
        - Uses dict-based `contents` so we don't pin to a specific types API.
        - Preserves tool calls + tool results as function_call/function_response parts.
        """
        # Build google-genai contents (role + parts)
        system_instruction = ""
        contents: List[Dict[str, Any]] = []

        tool_name_by_id: Dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                system_instruction = msg.get("content") or ""
                continue

            if role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.get("content") or ""}]})
                continue

            if role == "assistant":
                parts: List[Dict[str, Any]] = []

                text = msg.get("content")
                if isinstance(text, str) and text:
                    parts.append({"text": text})

                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    fn = (tc or {}).get("function", {})
                    name = fn.get("name") or ""
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args_dict = {"_raw_arguments": raw_args}

                    tc_id = (tc or {}).get("id") or ""
                    if tc_id and name:
                        tool_name_by_id[tc_id] = name

                    if name:
                        if not isinstance(args_dict, dict):
                            args_dict = {"input": args_dict}
                        parts.append({"function_call": {"name": name, "args": args_dict}})

                if parts:
                    contents.append({"role": "model", "parts": parts})
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id") or ""
                tool_name = tool_name_by_id.get(tool_call_id, "tool")
                raw_content = msg.get("content") or ""

                try:
                    parsed = json.loads(raw_content)
                    response_obj = parsed if isinstance(parsed, dict) else {"value": parsed}
                except Exception:
                    response_obj = {"content": raw_content}

                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": tool_name,
                                    "response": response_obj,
                                }
                            }
                        ],
                    }
                )
                continue

        # Tool declarations
        tool_decls: Optional[List[Dict[str, Any]]] = None
        if tools:
            fdecls = []
            for tool in tools:
                fdecls.append(
                    {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                )
            tool_decls = [{"function_declarations": fdecls}]

        # Build config dict. The python-genai SDK supports a dict-like config.
        config: Dict[str, Any] = {}
        if system_instruction:
            config["system_instruction"] = system_instruction
        if tool_decls is not None:
            config["tools"] = tool_decls

        # Common generation controls
        if "temperature" in kwargs:
            config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            config["max_output_tokens"] = kwargs["max_tokens"]

        # DedeuceRL effort knob for Gemini 3 "thinking" models.
        # Validity is checked via a cheap probe in the CLI when --effort is provided.
        if "thinking_level" in kwargs and kwargs["thinking_level"] is not None:
            config["thinking_config"] = {"thinking_level": str(kwargs["thinking_level"])}

        # Create client
        try:
            client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        except Exception as e:
            raise RuntimeError("Failed to initialize google-genai Client") from e

        # Call model
        try:
            resp = client.models.generate_content(
                model=self.model_id, contents=contents, config=config
            )
        except TypeError:
            # Some versions use `generation_config` instead of `config`
            resp = client.models.generate_content(
                model=self.model_id, contents=contents, generation_config=config
            )
        except Exception as e:
            raise RuntimeError("google-genai generate_content failed") from e

        # Parse response
        content_text = getattr(resp, "text", None)
        tool_calls: Optional[List[Dict[str, Any]]] = None

        candidates = getattr(resp, "candidates", None)
        if candidates:
            try:
                parts = candidates[0].content.parts
            except Exception:
                parts = []

            for part in parts or []:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    content_text = text
                    continue

                fc = getattr(part, "function_call", None)
                if fc is not None:
                    if tool_calls is None:
                        tool_calls = []
                    name = getattr(fc, "name", None) or ""
                    args = getattr(fc, "args", {})
                    try:
                        args_dict = dict(args) if not isinstance(args, dict) else args
                    except Exception:
                        args_dict = {"_raw": str(args)}

                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args_dict)},
                        }
                    )

        usage = None
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {
                "prompt_tokens": int(getattr(um, "prompt_token_count", 0)),
                "completion_tokens": int(getattr(um, "candidates_token_count", 0)),
                "total_tokens": int(getattr(um, "total_token_count", 0)),
            }

        return ModelReply(
            content=content_text, tool_calls=tool_calls, finish_reason="stop", usage=usage
        )
