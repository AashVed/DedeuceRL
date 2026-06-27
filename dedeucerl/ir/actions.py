"""Executable action-space algebra for TaskIR tasks."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Generic, Iterable, Literal, Mapping, Protocol, Sequence, TypeVar

from dedeucerl.kernel.types import TaskInstance
from dedeucerl.utils.schema import validate_jsonschema


A = TypeVar("A")
ToolKind = Literal["probe", "submit", "diagnostic"]


class ActionSpaceError(ValueError):
    """Base class for action-space validation and execution errors."""


class ActionValidationError(ActionSpaceError):
    """Raised when raw action input cannot be canonicalized into an action."""


class NonEnumerableActionSpace(ActionSpaceError):
    """Raised when an action space cannot be exhaustively enumerated."""


@dataclass(frozen=True)
class ActionContext:
    """Runtime context available to dynamic action masks."""

    instance: TaskInstance
    state: Any
    budget: int
    queries_used: int
    tool_calls: int
    done: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ActionSpace(Protocol, Generic[A]):
    """Validated and canonicalizable action vocabulary."""

    name: str

    def sample(self, rng: Any) -> A: ...

    def contains(self, action: Any) -> bool: ...

    def canonicalize(self, raw: Any) -> A: ...

    def to_json_schema(self) -> dict[str, Any]: ...

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]: ...

    def enumerate(self, limit: int | None = None) -> Iterable[A]: ...

    def mask(self, context: ActionContext) -> ActionSpace[A]: ...


@dataclass(frozen=True)
class EnumSpace(Generic[A]):
    """Finite enum action space."""

    name: str
    values: Sequence[A]
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def sample(self, rng: Any) -> A:
        if not self.values:
            raise ActionValidationError(f"{self.name}: cannot sample from empty enum")
        return rng.choice(list(self.values))

    def contains(self, action: Any) -> bool:
        try:
            self.canonicalize(action)
            return True
        except ActionValidationError:
            return False

    def canonicalize(self, raw: Any) -> A:
        if isinstance(raw, Mapping) and set(raw.keys()) == {"value"}:
            raw = raw["value"]
        for value in self.values:
            if _same_json_scalar(raw, value):
                return value
        raise ActionValidationError(f"{self.name}: expected one of {list(self.values)!r}")

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"enum": list(self.values)}
        if self.values and all(isinstance(v, str) for v in self.values):
            schema["type"] = "string"
        elif self.values and all(isinstance(v, int) and not isinstance(v, bool) for v in self.values):
            schema["type"] = "integer"
        elif self.values and all(isinstance(v, bool) for v in self.values):
            schema["type"] = "boolean"
        if self.description:
            schema["description"] = self.description
        return schema

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]:
        return _tool_schema(name, description, _wrapped_value_schema(self.to_json_schema()))

    def enumerate(self, limit: int | None = None) -> Iterable[A]:
        values = list(self.values)
        return iter(values if limit is None else values[: max(0, limit)])

    def mask(self, context: ActionContext) -> ActionSpace[A]:
        _ = context
        return self


@dataclass(frozen=True)
class ProductSpace:
    """Object action space whose fields are action spaces."""

    name: str
    fields: Mapping[str, ActionSpace[Any]]
    description: str | None = None
    allow_extra: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", dict(self.fields))

    def sample(self, rng: Any) -> dict[str, Any]:
        return {key: space.sample(rng) for key, space in self.fields.items()}

    def contains(self, action: Any) -> bool:
        try:
            self.canonicalize(action)
            return True
        except ActionValidationError:
            return False

    def canonicalize(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ActionValidationError(f"{self.name}: expected an object")

        raw_keys = set(raw.keys())
        field_keys = set(self.fields.keys())
        missing = sorted(field_keys - raw_keys)
        if missing:
            raise ActionValidationError(f"{self.name}: missing required field {missing[0]!r}")
        if not self.allow_extra:
            unknown = sorted(raw_keys - field_keys)
            if unknown:
                raise ActionValidationError(f"{self.name}: unknown field {unknown[0]!r}")

        canonical = {key: space.canonicalize(raw[key]) for key, space in self.fields.items()}
        if self.allow_extra:
            for key in sorted(raw_keys - field_keys):
                canonical[key] = _json_roundtrip(raw[key])
        return canonical

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {key: space.to_json_schema() for key, space in self.fields.items()},
            "required": list(self.fields.keys()),
            "additionalProperties": bool(self.allow_extra),
        }
        if self.description:
            schema["description"] = self.description
        return schema

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]:
        return _tool_schema(name, description, self.to_json_schema())

    def enumerate(self, limit: int | None = None) -> Iterable[dict[str, Any]]:
        if limit is not None and limit <= 0:
            return iter(())
        keys = list(self.fields.keys())
        value_lists = [list(self.fields[key].enumerate(limit=limit)) for key in keys]

        def _gen() -> Iterable[dict[str, Any]]:
            yielded = 0
            for combo in itertools.product(*value_lists):
                if limit is not None and yielded >= limit:
                    break
                yielded += 1
                yield dict(zip(keys, combo, strict=True))

        return _gen()

    def mask(self, context: ActionContext) -> ActionSpace[dict[str, Any]]:
        return replace(
            self,
            fields={key: space.mask(context) for key, space in self.fields.items()},
        )


@dataclass(frozen=True)
class UnionSpace:
    """Tagged union action space."""

    name: str
    variants: Mapping[str, ActionSpace[Any]]
    tag_field: str = "type"
    value_field: str = "value"
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "variants", dict(self.variants))

    def sample(self, rng: Any) -> dict[str, Any]:
        tags = list(self.variants.keys())
        if not tags:
            raise ActionValidationError(f"{self.name}: cannot sample from empty union")
        tag = rng.choice(tags)
        return {self.tag_field: tag, self.value_field: self.variants[tag].sample(rng)}

    def contains(self, action: Any) -> bool:
        try:
            self.canonicalize(action)
            return True
        except ActionValidationError:
            return False

    def canonicalize(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ActionValidationError(f"{self.name}: expected a tagged object")
        if self.tag_field not in raw:
            raise ActionValidationError(f"{self.name}: missing tag field {self.tag_field!r}")
        tag = str(raw[self.tag_field])
        if tag not in self.variants:
            raise ActionValidationError(f"{self.name}: unknown variant {tag!r}")
        if self.value_field not in raw:
            raise ActionValidationError(f"{self.name}: missing value field {self.value_field!r}")
        return {
            self.tag_field: tag,
            self.value_field: self.variants[tag].canonicalize(raw[self.value_field]),
        }

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                self.tag_field: {"type": "string", "enum": list(self.variants.keys())},
                self.value_field: {
                    "anyOf": [space.to_json_schema() for space in self.variants.values()]
                },
            },
            "required": [self.tag_field, self.value_field],
            "additionalProperties": False,
        }
        if self.description:
            schema["description"] = self.description
        return schema

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]:
        return _tool_schema(name, description, self.to_json_schema())

    def enumerate(self, limit: int | None = None) -> Iterable[dict[str, Any]]:
        if limit is not None and limit <= 0:
            return iter(())

        def _gen() -> Iterable[dict[str, Any]]:
            yielded = 0
            for tag, space in self.variants.items():
                for value in space.enumerate(limit=None if limit is None else limit - yielded):
                    if limit is not None and yielded >= limit:
                        return
                    yielded += 1
                    yield {self.tag_field: tag, self.value_field: value}

        return _gen()

    def mask(self, context: ActionContext) -> ActionSpace[dict[str, Any]]:
        return replace(
            self,
            variants={tag: space.mask(context) for tag, space in self.variants.items()},
        )


@dataclass(frozen=True)
class SequenceSpace:
    """Bounded sequence action space."""

    name: str
    item_space: ActionSpace[Any]
    max_len: int
    min_len: int = 0
    description: str | None = None

    def __post_init__(self) -> None:
        if self.min_len < 0:
            raise ValueError("min_len must be >= 0")
        if self.max_len < self.min_len:
            raise ValueError("max_len must be >= min_len")

    def sample(self, rng: Any) -> list[Any]:
        length = rng.randint(self.min_len, self.max_len)
        return [self.item_space.sample(rng) for _ in range(length)]

    def contains(self, action: Any) -> bool:
        try:
            self.canonicalize(action)
            return True
        except ActionValidationError:
            return False

    def canonicalize(self, raw: Any) -> list[Any]:
        if isinstance(raw, Mapping) and set(raw.keys()) == {"items"}:
            raw = raw["items"]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ActionValidationError(f"{self.name}: expected an array")
        values = list(raw)
        if len(values) < self.min_len:
            raise ActionValidationError(f"{self.name}: expected at least {self.min_len} items")
        if len(values) > self.max_len:
            raise ActionValidationError(f"{self.name}: expected at most {self.max_len} items")
        return [self.item_space.canonicalize(value) for value in values]

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "array",
            "items": self.item_space.to_json_schema(),
            "minItems": self.min_len,
            "maxItems": self.max_len,
        }
        if self.description:
            schema["description"] = self.description
        return schema

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]:
        return _tool_schema(
            name,
            description,
            {
                "type": "object",
                "properties": {"items": self.to_json_schema()},
                "required": ["items"],
                "additionalProperties": False,
            },
        )

    def enumerate(self, limit: int | None = None) -> Iterable[list[Any]]:
        if limit is not None and limit <= 0:
            return iter(())
        items = list(self.item_space.enumerate(limit=limit))

        def _gen() -> Iterable[list[Any]]:
            yielded = 0
            for length in range(self.min_len, self.max_len + 1):
                for combo in itertools.product(items, repeat=length):
                    if limit is not None and yielded >= limit:
                        return
                    yielded += 1
                    yield list(combo)

        return _gen()

    def mask(self, context: ActionContext) -> ActionSpace[list[Any]]:
        return replace(self, item_space=self.item_space.mask(context))


@dataclass(frozen=True)
class JsonSchemaSpace:
    """Action space backed by a JSONSchema-like schema."""

    name: str
    schema: Mapping[str, Any]
    description: str | None = None
    sampler: Callable[[Any], Any] | None = None
    enumerator: Callable[[int | None], Iterable[Any]] | None = None

    def sample(self, rng: Any) -> Any:
        if self.sampler is None:
            raise NonEnumerableActionSpace(f"{self.name}: no sampler configured")
        return self.canonicalize(self.sampler(rng))

    def contains(self, action: Any) -> bool:
        try:
            self.canonicalize(action)
            return True
        except ActionValidationError:
            return False

    def canonicalize(self, raw: Any) -> Any:
        if self.schema.get("type") != "object" and isinstance(raw, Mapping) and set(raw.keys()) == {
            "value"
        }:
            raw = raw["value"]
        canonical = _json_roundtrip(raw)
        reason = validate_jsonschema(canonical, dict(self.schema))
        if reason:
            raise ActionValidationError(f"{self.name}: {reason}")
        return canonical

    def to_json_schema(self) -> dict[str, Any]:
        schema = dict(self.schema)
        if self.description and "description" not in schema:
            schema["description"] = self.description
        return schema

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]:
        schema = self.to_json_schema()
        parameters = schema if schema.get("type") == "object" else _wrapped_value_schema(schema)
        return _tool_schema(name, description, parameters)

    def enumerate(self, limit: int | None = None) -> Iterable[Any]:
        if self.enumerator is None:
            raise NonEnumerableActionSpace(f"{self.name}: no finite enumerator configured")
        values = self.enumerator(limit)
        return (self.canonicalize(value) for value in values)

    def mask(self, context: ActionContext) -> ActionSpace[Any]:
        _ = context
        return self


MaskFn = Callable[[ActionContext, Any], bool]


@dataclass(frozen=True)
class MaskedSpace(Generic[A]):
    """Context-sensitive filter over another action space."""

    name: str
    base_space: ActionSpace[A]
    mask_fn: MaskFn
    context: ActionContext | None = None
    description: str | None = None

    def sample(self, rng: Any) -> A:
        if self.context is None:
            return self.base_space.sample(rng)
        try:
            values = list(self.enumerate(limit=None))
        except NonEnumerableActionSpace:
            values = []
        if values:
            return rng.choice(values)
        for _ in range(1000):
            action = self.base_space.sample(rng)
            if self._allowed(action):
                return action
        raise ActionValidationError(f"{self.name}: no unmasked action could be sampled")

    def contains(self, action: Any) -> bool:
        try:
            self.canonicalize(action)
            return True
        except ActionValidationError:
            return False

    def canonicalize(self, raw: Any) -> A:
        action = self.base_space.canonicalize(raw)
        if not self._allowed(action):
            raise ActionValidationError(f"{self.name}: action is masked out")
        return action

    def to_json_schema(self) -> dict[str, Any]:
        schema = self.base_space.to_json_schema()
        if self.context is not None:
            try:
                values = list(self.enumerate(limit=None))
            except NonEnumerableActionSpace:
                values = []
            if values and all(_is_json_scalar(value) for value in values):
                schema["enum"] = values
        if self.description and "description" not in schema:
            schema["description"] = self.description
        return schema

    def to_tool_schema(self, name: str, description: str) -> dict[str, Any]:
        schema = self.to_json_schema()
        parameters = schema if schema.get("type") == "object" else _wrapped_value_schema(schema)
        return _tool_schema(name, description, parameters)

    def enumerate(self, limit: int | None = None) -> Iterable[A]:
        if limit is not None and limit <= 0:
            return iter(())

        def _gen() -> Iterable[A]:
            yielded = 0
            for action in self.base_space.enumerate(limit=None):
                if self._allowed(action):
                    if limit is not None and yielded >= limit:
                        return
                    yielded += 1
                    yield action

        return _gen()

    def mask(self, context: ActionContext) -> ActionSpace[A]:
        return replace(self, context=context, base_space=self.base_space.mask(context))

    def _allowed(self, action: A) -> bool:
        return self.context is None or bool(self.mask_fn(self.context, action))


@dataclass(frozen=True)
class ToolActionContract(Generic[A]):
    """Named tool projection over an action space."""

    name: str
    kind: ToolKind
    description: str
    action_space: ActionSpace[A]
    return_schema: Mapping[str, Any]
    cost: int = 1

    def canonicalize(self, raw_args: Any) -> A:
        return self.action_space.canonicalize(raw_args)

    def mask(self, context: ActionContext) -> ToolActionContract[A]:
        return replace(self, action_space=self.action_space.mask(context))

    def to_tool_schema(self) -> dict[str, Any]:
        return self.action_space.to_tool_schema(self.name, self.description)


@dataclass(frozen=True)
class ToolActionSpace:
    """Task-level named projection space over tool contracts."""

    contracts: Sequence[ToolActionContract[Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contracts", tuple(self.contracts))
        names = [contract.name for contract in self.contracts]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tool action names: {names!r}")

    def names(self) -> list[str]:
        return [contract.name for contract in self.contracts]

    def contracts_for_context(self, context: ActionContext) -> list[ToolActionContract[Any]]:
        return [contract.mask(context) for contract in self.contracts]

    def get(
        self,
        name: str,
        context: ActionContext | None = None,
    ) -> ToolActionContract[Any] | None:
        for contract in self.contracts:
            if contract.name == name:
                return contract if context is None else contract.mask(context)
        return None

    def to_tool_schemas(self, context: ActionContext | None = None) -> list[dict[str, Any]]:
        contracts = self.contracts if context is None else self.contracts_for_context(context)
        return [contract.to_tool_schema() for contract in contracts]


def _tool_schema(name: str, description: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": dict(parameters),
    }


def _wrapped_value_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"value": dict(schema)},
        "required": ["value"],
        "additionalProperties": False,
    }


def _same_json_scalar(left: Any, right: Any) -> bool:
    if not _is_json_scalar(left) or not _is_json_scalar(right):
        return left == right
    return type(left) is type(right) and left == right


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _json_roundtrip(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except TypeError as e:
        raise ActionValidationError(f"value is not JSON-serializable: {e}") from e
