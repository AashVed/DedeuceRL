"""Action space for typed Python submissions with matching JSON schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generic, Iterable, TypeVar

from pydantic import TypeAdapter, ValidationError

from dedeucerl.ir.actions import ActionContext, ActionValidationError, NonEnumerableActionSpace

A = TypeVar("A")


@dataclass(frozen=True)
class TypedSpace(Generic[A]):
    """Strict JSON decoding; trusted Python instances follow Pydantic policy."""

    name: str
    model: type[A]

    def canonicalize(self, raw: Any) -> A:
        try:
            adapter: TypeAdapter[A] = TypeAdapter(self.model)
            if isinstance(raw, self.model):
                return adapter.validate_python(raw, strict=True)
            return adapter.validate_json(json.dumps(raw, allow_nan=False), strict=True)
        except (ValidationError, TypeError, ValueError) as error:
            raise ActionValidationError(str(error)) from error

    def contains(self, action: Any) -> bool:
        try:
            self.canonicalize(action)
        except ActionValidationError:
            return False
        return True

    def to_json_schema(self) -> dict[str, Any]:
        return TypeAdapter(self.model).json_schema()

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]:
        schema = self.to_json_schema()
        reference = schema.get("$ref", "")
        if reference.startswith("#/$defs/"):
            # Recursive object models use a root reference. Tool surfaces need
            # root properties; retain definitions for all nested references.
            key = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
            schema = {
                **schema["$defs"][key],
                **{key: value for key, value in schema.items() if key != "$ref"},
            }
        if schema.get("type") != "object":
            raise ValueError("typed tool arguments must be an object model")
        return {"name": name, "description": description, "parameters": schema}

    def mask(self, context: ActionContext) -> TypedSpace[A]:
        return self

    def enumerate(self, limit: int | None = None) -> Iterable[A]:
        raise NonEnumerableActionSpace(f"{self.name}: typed models are not enumerable")

    def sample(self, rng: Any) -> A:
        raise NonEnumerableActionSpace(f"{self.name}: provide a domain-specific sampler")
