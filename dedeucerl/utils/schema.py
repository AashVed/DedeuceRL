"""Lightweight JSONSchema validation utilities.

DedeuceRL uses JSONSchema-like dicts (see `DomainSpec.hypothesis_schema`) to
optionally pre-validate hypothesis submissions.

We intentionally implement a small subset of JSONSchema to avoid adding a hard
runtime dependency. The supported keywords are chosen to cover the schemas used
in the built-in skins:

- type
- required
- properties
- patternProperties
- enum
- const
- minimum / maximum
- items (list or schema)
- minItems / maxItems

For unsupported keywords, validation is best-effort (ignored).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union


_JSONSchema = Dict[str, Any]


def validate_jsonschema(instance: Any, schema: _JSONSchema) -> Optional[str]:
    """Validate `instance` against a limited JSONSchema subset.

    Returns:
        None if valid, otherwise a short human-readable reason string.

    Notes:
        This is not a full JSONSchema implementation.
    """

    return _validate(instance, schema, path="$")


def _validate(instance: Any, schema: _JSONSchema, *, path: str) -> Optional[str]:
    if not isinstance(schema, dict):
        return None

    # const
    if "const" in schema:
        expected = schema.get("const")
        if instance != expected:
            return f"{path}: must equal {expected!r}"

    # enum
    if "enum" in schema:
        allowed = schema.get("enum")
        if isinstance(allowed, list) and instance not in allowed:
            return f"{path}: must be one of {allowed!r}"

    # type
    expected_type = schema.get("type")
    if expected_type is not None:
        ok, type_err = _check_type(instance, expected_type)
        if not ok:
            return f"{path}: {type_err}"

    # numeric bounds
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema:
            min_v = schema.get("minimum")
            if isinstance(min_v, (int, float)) and instance < min_v:
                return f"{path}: must be >= {min_v}"
        if "maximum" in schema:
            max_v = schema.get("maximum")
            if isinstance(max_v, (int, float)) and instance > max_v:
                return f"{path}: must be <= {max_v}"

    # object handling
    if expected_type == "object" and isinstance(instance, dict):
        req = schema.get("required")
        if isinstance(req, list):
            for k in req:
                if k not in instance:
                    return f"{path}: missing required key {k!r}"

        props = schema.get("properties")
        if isinstance(props, dict):
            for k, subschema in props.items():
                if k in instance:
                    err = _validate(instance[k], subschema, path=f"{path}.{k}")
                    if err:
                        return err

        pat_props = schema.get("patternProperties")
        if isinstance(pat_props, dict):
            compiled: List[Tuple[re.Pattern[str], _JSONSchema]] = []
            for pattern, subschema in pat_props.items():
                try:
                    compiled.append((re.compile(pattern), subschema))
                except re.error:
                    continue

            for key, value in instance.items():
                for regex, subschema in compiled:
                    if regex.search(str(key)):
                        err = _validate(value, subschema, path=f"{path}.{key}")
                        if err:
                            return err

        return None

    # array handling
    if expected_type == "array" and isinstance(instance, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(instance) < min_items:
            return f"{path}: must have at least {min_items} items"
        if isinstance(max_items, int) and len(instance) > max_items:
            return f"{path}: must have at most {max_items} items"

        items = schema.get("items")
        if isinstance(items, list):
            # tuple typing
            for i, subschema in enumerate(items):
                if i >= len(instance):
                    break
                err = _validate(instance[i], subschema, path=f"{path}[{i}]")
                if err:
                    return err
        elif isinstance(items, dict):
            for i, value in enumerate(instance):
                err = _validate(value, items, path=f"{path}[{i}]")
                if err:
                    return err

        return None

    return None


def _check_type(value: Any, expected: Union[str, List[str]]) -> Tuple[bool, str]:
    def _one(v: Any, t: str) -> bool:
        if t == "object":
            return isinstance(v, dict)
        if t == "array":
            return isinstance(v, list)
        if t == "string":
            return isinstance(v, str)
        if t == "boolean":
            return isinstance(v, bool)
        if t == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        if t == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        return True

    if isinstance(expected, list):
        if any(_one(value, t) for t in expected if isinstance(t, str)):
            return True, ""
        return False, f"expected type in {expected!r}"

    if isinstance(expected, str):
        if _one(value, expected):
            return True, ""
        return False, f"expected type '{expected}'"

    return True, ""
