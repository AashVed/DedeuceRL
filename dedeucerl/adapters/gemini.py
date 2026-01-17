"""Google Gemini adapter for DedeuceRL.

This adapter targets *tool-calling* workloads.

Provider support strategy
-------------------------
Google's Python ecosystem has had multiple client libraries in active use.
To keep DedeuceRL usable for practitioners, we support:

- Preferred (newer): `google-genai` (imported as `google.genai`)
- Legacy: `google-generativeai` (imported as `google.generativeai`)

At runtime, the adapter will try the newer library first and fall back.

Tool round-trip semantics
------------------------
DedeuceRL uses a provider-agnostic transcript:
- Assistant tool requests are stored as OpenAI-style `tool_calls` on an assistant message.
- Tool results are stored as messages with `role="tool"`, `tool_call_id`, and `content`.

Gemini expects tool round-trips as function_call/function_response parts.
When the installed client library exposes those proto types we use them.
Otherwise, we fall back to embedding tool calls/results as text blocks.

The fallback is intentionally conservative: it preserves correctness of the
loop even if the provider-specific structured parts aren't available.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseAdapter, ModelReply


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini tool calling APIs."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.api_key = api_key

    def _import_backend(self) -> Tuple[str, Any]:
        """Import a Gemini client backend.

        Returns:
            (backend_name, module)

        Raises:
            ImportError if neither backend is installed.
        """
        try:
            import google.genai as genai_new  # type: ignore

            return "genai", genai_new
        except Exception:
            pass

        try:
            import google.generativeai as genai_legacy  # type: ignore

            return "generativeai", genai_legacy
        except Exception as e:
            raise ImportError(
                "GeminiAdapter requires either `google-genai` (google.genai) or "
                "`google-generativeai` (google.generativeai). Install with: pip install 'dedeucerl[gemini]'"
            ) from e

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        backend, genai = self._import_backend()

        if backend == "genai":
            return self._chat_genai(messages, tools=tools, **kwargs)

        return self._chat_generativeai(messages, tools=tools, **kwargs)

    # ---------------------------------------------------------------------
    # Legacy backend: google-generativeai
    # ---------------------------------------------------------------------

    def _chat_generativeai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        genai: Any = importlib.import_module("google.generativeai")

        if self.api_key:
            configure = getattr(genai, "configure", None)
            if callable(configure):
                configure(api_key=self.api_key)

        system_instruction, history, last_parts = self._to_generativeai_chat(messages)

        GenerationConfig = getattr(genai, "GenerationConfig", None)
        if callable(GenerationConfig):
            generation_config = GenerationConfig(**self.options.get("generation_config", {}))
        else:
            # Best-effort fallback: pass no generation config
            generation_config = None

        gemini_tools = self._format_tools_legacy(tools) if tools else None

        GenerativeModel = getattr(genai, "GenerativeModel", None)
        if not callable(GenerativeModel):
            raise RuntimeError(
                "google-generativeai is installed but does not expose GenerativeModel; "
                "please upgrade the package."
            )

        if system_instruction:
            model: Any = GenerativeModel(
                self.model_id,
                system_instruction=system_instruction,
                tools=gemini_tools,
            )
        else:
            model = GenerativeModel(self.model_id, tools=gemini_tools)

        start_chat = getattr(model, "start_chat", None)
        if not callable(start_chat):
            raise RuntimeError("Gemini model object does not support start_chat()")
        chat: Any = start_chat(history=history)

        send_message = getattr(chat, "send_message", None)
        if not callable(send_message):
            raise RuntimeError("Gemini chat object does not support send_message()")

        if generation_config is not None:
            response: Any = send_message(last_parts, generation_config=generation_config)
        else:
            response = send_message(last_parts)

        content = None
        tool_calls: Optional[List[Dict[str, Any]]] = None

        # Extract response parts (text + tool calls)
        for part in getattr(response, "parts", []) or []:
            if hasattr(part, "text") and part.text:
                content = part.text
            elif hasattr(part, "function_call"):
                if tool_calls is None:
                    tool_calls = []
                fc = part.function_call
                # Some SDK versions expose args as mapping-like
                args_obj = getattr(fc, "args", {})
                try:
                    args_dict = dict(args_obj)
                except Exception:
                    args_dict = {"_raw": str(args_obj)}

                tool_calls.append(
                    {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": getattr(fc, "name", None) or "",
                            "arguments": json.dumps(args_dict),
                        },
                    }
                )

        usage = None
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0),
                "completion_tokens": getattr(um, "candidates_token_count", 0),
                "total_tokens": getattr(um, "total_token_count", 0),
            }

        return ModelReply(content=content, tool_calls=tool_calls, finish_reason="stop", usage=usage)

    def _to_generativeai_chat(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]], Any]:
        """Convert canonical transcript into google-generativeai chat inputs.

        Returns:
            (system_instruction, history, last_parts)
        """
        system_instruction: Optional[str] = None
        history: List[Dict[str, Any]] = []

        # We try to use proto parts if available, otherwise we fall back to strings.
        try:
            genai: Any = importlib.import_module("google.generativeai")

            protos = getattr(genai, "protos", None)
        except Exception:
            protos = None

        # Track tool name for tool_result mapping.
        tool_name_by_id: Dict[str, str] = {}

        def _text_part(text: str) -> Any:
            return text

        def _function_call_part(name: str, args: Dict[str, Any]) -> Any:
            if protos is None:
                return _text_part(f"[TOOL_CALL] {name} {json.dumps(args)}")
            try:
                return protos.Part(function_call=protos.FunctionCall(name=name, args=args))
            except Exception:
                return _text_part(f"[TOOL_CALL] {name} {json.dumps(args)}")

        def _function_response_part(name: str, response_obj: Dict[str, Any]) -> Any:
            if protos is None:
                return _text_part(f"[TOOL_RESULT] {name} {json.dumps(response_obj)}")
            try:
                return protos.Part(
                    function_response=protos.FunctionResponse(name=name, response=response_obj)
                )
            except Exception:
                return _text_part(f"[TOOL_RESULT] {name} {json.dumps(response_obj)}")

        def _append(role: str, parts: List[Any]) -> None:
            # google-generativeai uses "user" and "model" roles.
            history.append({"role": role, "parts": parts})

        # Build a list of (role, parts) entries; we'll send the last one via send_message.
        entries: List[Tuple[str, List[Any]]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                system_instruction = msg.get("content") or ""
                continue

            if role == "user":
                entries.append(("user", [_text_part(msg.get("content") or "")]))
                continue

            if role == "assistant":
                parts: List[Any] = []
                text = msg.get("content")
                if isinstance(text, str) and text:
                    parts.append(_text_part(text))

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
                        if isinstance(args_dict, dict):
                            parts.append(_function_call_part(name, args_dict))
                        else:
                            parts.append(_function_call_part(name, {"input": args_dict}))

                if parts:
                    entries.append(("model", parts))
                continue

            if role == "tool":
                tool_use_id = msg.get("tool_call_id") or ""
                tool_name = tool_name_by_id.get(tool_use_id, "")
                content = msg.get("content") or ""

                # Gemini function_response expects an object; attempt to parse tool output JSON.
                response_obj: Dict[str, Any]
                try:
                    parsed = json.loads(content)
                    response_obj = parsed if isinstance(parsed, dict) else {"value": parsed}
                except Exception:
                    response_obj = {"content": content}

                if not tool_name:
                    # Best-effort: preserve id even if we can't resolve the tool name.
                    tool_name = "tool"
                    response_obj["tool_call_id"] = tool_use_id

                entries.append(("user", [_function_response_part(tool_name, response_obj)]))
                continue

        # If no entries, send empty.
        if not entries:
            return system_instruction, [], ""

        # Send last entry in send_message; put the rest in history.
        for r, p in entries[:-1]:
            _append(r, p)

        last_role, last_parts_list = entries[-1]
        # `send_message` expects the content/parts for the user turn. If the last entry
        # is a model entry, we send an empty user turn to continue.
        if last_role != "user":
            last_parts: Any = ""
            _append(last_role, last_parts_list)
        else:
            last_parts = last_parts_list

        return system_instruction, history, last_parts

    def _format_tools_legacy(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """Format tools for google-generativeai.

        We access proto types dynamically to avoid hard dependency on stub exports.
        """
        genai: Any = importlib.import_module("google.generativeai")
        protos = getattr(genai, "protos", None)
        if protos is None:
            raise RuntimeError("google-generativeai does not expose protos; please upgrade")

        FunctionDeclaration = getattr(protos, "FunctionDeclaration", None)
        Tool = getattr(protos, "Tool", None)
        if not callable(FunctionDeclaration) or not callable(Tool):
            raise RuntimeError("google-generativeai protos missing FunctionDeclaration/Tool")

        function_declarations = []
        for tool in tools:
            fd = FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=self._convert_parameters_legacy(tool.get("parameters", {})),
            )
            function_declarations.append(fd)

        return [Tool(function_declarations=function_declarations)]

    def _convert_parameters_legacy(self, params: Dict[str, Any]) -> Optional[Any]:
        """Convert JSON Schema parameters to google-generativeai schema.

        Important: preserve `enum` where supported so domain alphabets remain first-class.
        """
        genai: Any = importlib.import_module("google.generativeai")
        protos = getattr(genai, "protos", None)
        if protos is None:
            return None

        Schema = getattr(protos, "Schema", None)
        TypeEnum = getattr(protos, "Type", None)
        if not callable(Schema) or TypeEnum is None:
            return None

        if not params or params.get("type") != "object":
            return None

        properties: Dict[str, Any] = {}
        for name, prop in (params.get("properties") or {}).items():
            prop_type = str(prop.get("type", "string")).upper()
            if prop_type == "INTEGER":
                prop_type = "NUMBER"

            string_type = getattr(TypeEnum, "STRING", None)
            inferred_type = getattr(TypeEnum, prop_type, string_type)

            schema_kwargs: Dict[str, Any] = {
                "type": inferred_type,
                "description": prop.get("description", ""),
            }

            # Best-effort enum preservation.
            if "enum" in prop and isinstance(prop.get("enum"), list):
                schema_kwargs["enum"] = prop["enum"]

            try:
                properties[name] = Schema(**schema_kwargs)
            except TypeError:
                schema_kwargs.pop("enum", None)
                properties[name] = Schema(**schema_kwargs)

        try:
            object_type = getattr(TypeEnum, "OBJECT", None)
            return Schema(
                type=object_type,
                properties=properties,
                required=params.get("required", []),
            )
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # New backend: google-genai (best-effort)
    # ---------------------------------------------------------------------

    def _chat_genai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> ModelReply:
        """Chat via `google.genai` (preferred Gemini SDK).

        Implementation notes:
        - Uses dict-based `contents` so we don't pin to a specific types API.
        - Preserves tool calls + tool results as function_call/function_response parts.
        """
        try:
            import google.genai as genai_new  # type: ignore
        except Exception as e:
            raise ImportError("google-genai not installed") from e

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

        # Build config dict (google-genai uses GenerateContentConfig; dict works in practice)
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

        # Create client
        try:
            client = genai_new.Client(api_key=self.api_key) if self.api_key else genai_new.Client()
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
                        args_dict = dict(args)
                    except Exception:
                        args_dict = {"_raw": str(args)}

                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args_dict)},
                        }
                    )

        return ModelReply(
            content=content_text, tool_calls=tool_calls, finish_reason="stop", usage=None
        )
