"""Domain specification types for DedeuceRL skins.

This module defines the schema-first approach for exposing:
- Action/output vocabularies
- Tool interface schemas (with enum support)
- Hypothesis schemas (for pre-validation)
- Observation schemas

Each skin implements `domain_spec(params) -> DomainSpec` as the single
source of truth for its domain vocabulary and interface contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class ArgSchema:
    """Schema for a tool argument.

    Supports enum constraints for structured action spaces.
    """

    type: Literal["string", "integer", "number", "boolean", "object", "array"]
    description: str = ""
    enum: Optional[List[Any]] = None  # Allowed values (makes action space explicit)
    required: bool = True
    default: Optional[Any] = None

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSONSchema format for tool calling APIs."""
        schema: Dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ReturnField:
    """Schema for a return field."""

    type: str  # "int", "bool", "string", "list", etc.
    description: str = ""


@dataclass
class ToolSchema:
    """Complete schema for a skin tool.

    Used by:
    - Prompt generation (structured tool descriptions)
    - Eval CLI (proper function calling schemas)
    - Input validation (single source of truth)
    """

    name: str
    description: str
    args: Dict[str, ArgSchema]
    returns: Dict[str, ReturnField]

    def to_tool_dict(self) -> Dict[str, Any]:
        """Convert to provider-agnostic tool schema.

        This matches what DedeuceRL adapters expect:
        - OpenAI adapter wraps this into {"type":"function","function": ...}
        - Anthropic adapter maps {name, description, parameters} to input_schema
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for arg_name, arg_schema in self.args.items():
            properties[arg_name] = arg_schema.to_json_schema()
            if arg_schema.required:
                required.append(arg_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def format_for_prompt(self) -> str:
        """Format tool for inclusion in system prompt."""
        args_str = ", ".join(
            f"{name}: {s.type}" + (f" (one of {s.enum})" if s.enum else "")
            for name, s in self.args.items()
        )
        returns_str = ", ".join(f"{name}: {f.type}" for name, f in self.returns.items())
        return f"- {self.name}({args_str}) → {{{returns_str}}}"


@dataclass
class ObservationField:
    """Schema for an observation field shown to the agent."""

    type: str
    description: str
    example: Optional[Any] = None


ParamType = Literal["int", "float", "bool", "str", "json"]


@dataclass
class ParamSpec:
    """Schema for a skin configuration parameter.

    This is intended for practical usability:
    - Auto-documenting skins ("what knobs exist?")
    - Optional CLI auto-wiring (avoid hardcoded skin args)

    Notes:
    - Keep this lightweight; `dedeucerl-generate` also supports `--param KEY=VALUE`
      and `--skin-kwargs '{...}'` for power users.
    """

    type: ParamType
    description: str
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None


@dataclass
class DomainSpec:
    """Single source of truth for a skin's domain vocabulary.

    Every skin implements `domain_spec(params) -> DomainSpec` that returns
    the complete specification for that configuration. This enables:

    1. **Structured action spaces**: `actions` with explicit enum in tool schema
    2. **Pre-validation**: `hypothesis_schema` checked before equivalence
    3. **Prompt generation**: Tools + observation from spec (no hand-written duplication)
    4. **RL grounding**: Agents receive structured action vocabulary
    """

    # ─────────────────────────────────────────────────────────────
    # Domain vocabulary
    # ─────────────────────────────────────────────────────────────

    actions: List[str]
    """Allowed action tokens (e.g., ["A", "B", "C"] or ["N", "S", "E", "W"])."""

    outputs: List[Any]
    """Allowed output values (e.g., [0, 1, 2] or [200, 201, 400, 404])."""

    # ─────────────────────────────────────────────────────────────
    # Interface schemas
    # ─────────────────────────────────────────────────────────────

    tool_schemas: List[ToolSchema]
    """Complete schemas for all tools exposed to the agent."""

    hypothesis_schema: Dict[str, Any]
    """JSONSchema for hypothesis submission (used for pre-validation)."""

    observation_fields: Dict[str, ObservationField]
    """Fields shown in observation, with types and descriptions."""

    # ─────────────────────────────────────────────────────────────
    # Metadata (non-default fields must come before default fields)
    # ─────────────────────────────────────────────────────────────

    skin_name: str
    """Identifier for the skin (e.g., mealy, protocol)."""

    n_states: int
    """Number of states in the hidden system."""

    has_traps: bool
    """Whether trap transitions exist."""

    # ─────────────────────────────────────────────────────────────
    # Optional fields with defaults
    # ─────────────────────────────────────────────────────────────

    params: Dict[str, ParamSpec] = field(default_factory=dict)
    """Optional parameter registry for this skin configuration.

    This is used for usability and tooling (e.g., `dedeucerl-generate` can
    expose skin params without hardcoding).
    """

    # ─────────────────────────────────────────────────────────────
    # Helper methods
    # ─────────────────────────────────────────────────────────────

    def get_tool_schema(self, name: str) -> Optional[ToolSchema]:
        """Get schema for a specific tool by name."""
        for tool in self.tool_schemas:
            if tool.name == name:
                return tool
        return None

    def build_observation(self, **values) -> Dict[str, Any]:
        """Build observation dict from provided values.

        Falls back to examples if value not provided.
        """
        obs = {}
        for field_name, field_spec in self.observation_fields.items():
            if field_name in values:
                obs[field_name] = values[field_name]
            elif field_spec.example is not None:
                obs[field_name] = field_spec.example
        return obs

    def format_tools_for_prompt(self) -> str:
        """Format all tools for system prompt."""
        return "\n".join(tool.format_for_prompt() for tool in self.tool_schemas)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in adapter-friendly format."""
        return [tool.to_tool_dict() for tool in self.tool_schemas]

    def validate_action(self, action: str) -> bool:
        """Check if action is in allowed vocabulary."""
        return action in self.actions
