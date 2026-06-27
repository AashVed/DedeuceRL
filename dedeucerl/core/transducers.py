"""Pure finite-transducer table utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dedeucerl.core.automata import find_counterexample
from dedeucerl.utils.schema import validate_jsonschema


def transducer_table_schema(
    alphabet: Sequence[str],
    *,
    outputs: Sequence[Any] | None = None,
    start_const: int | None = None,
) -> dict[str, Any]:
    """Return the JSONSchema-like table shape used by finite transducer contracts."""

    output_schema: dict[str, Any] = {}
    if outputs is not None:
        values = list(outputs)
        output_schema["enum"] = values
        output_type = _homogeneous_json_type(values)
        if output_type is not None:
            output_schema["type"] = output_type

    start_schema: dict[str, Any] = {"type": "integer"}
    if start_const is not None:
        start_schema["const"] = int(start_const)

    return {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1},
            "start": start_schema,
            "trans": {
                "type": "object",
                "patternProperties": {
                    "^[0-9]+$": {
                        "type": "object",
                        "properties": {
                            sym: {
                                "type": "array",
                                "items": [{"type": "integer"}, output_schema],
                                "minItems": 2,
                                "maxItems": 2,
                            }
                            for sym in alphabet
                        },
                        "required": list(alphabet),
                    }
                },
            },
        },
        "required": ["n", "start", "trans"],
    }


def normalize_transducer_table(
    table: Mapping[str, Any],
    alphabet: Sequence[str],
    *,
    outputs: Sequence[Any] | None = None,
    start_const: int | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize a complete finite-transducer table."""

    reason = validate_jsonschema(
        table,
        transducer_table_schema(alphabet, outputs=outputs, start_const=start_const),
    )
    if reason:
        raise ValueError(reason)

    n = _int_field(table, "n")
    start = _int_field(table, "start")
    if n < 1:
        raise ValueError("n must be >= 1")
    if not (0 <= start < n):
        raise ValueError("start must be a valid state")

    raw_trans = table.get("trans")
    if not isinstance(raw_trans, Mapping):
        raise ValueError("trans must be an object")

    allowed_outputs = None if outputs is None else set(outputs)
    trans: dict[str, dict[str, list[Any]]] = {}
    for state in range(n):
        row = raw_trans.get(str(state))
        if row is None:
            row = raw_trans.get(state)
        if not isinstance(row, Mapping):
            raise ValueError(f"missing transition row for state {state}")

        canonical_row: dict[str, list[Any]] = {}
        for symbol in alphabet:
            pair = row.get(symbol)
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes, bytearray))
                or len(pair) != 2
            ):
                raise ValueError(f"transition {state}.{symbol} must be [next_state, output]")
            next_state = _int_value(pair[0], f"transition {state}.{symbol} next_state")
            if not (0 <= next_state < n):
                raise ValueError(f"transition {state}.{symbol} next_state out of range")
            output = pair[1]
            if allowed_outputs is not None and output not in allowed_outputs:
                raise ValueError(f"transition {state}.{symbol} output is not allowed")
            canonical_row[str(symbol)] = [next_state, output]
        trans[str(state)] = canonical_row

    return {"n": n, "start": start, "trans": trans}


def parse_transducer_transitions(
    table: Mapping[str, Any],
) -> dict[int, dict[str, tuple[int, Any]]]:
    """Convert a canonical table into transition-function lookup form."""

    raw = table.get("trans", {})
    if not isinstance(raw, Mapping):
        raise ValueError("trans must be a mapping")
    return {
        int(state): {str(action): (int(pair[0]), pair[1]) for action, pair in row.items()}
        for state, row in raw.items()
        if isinstance(row, Mapping)
    }


def transducer_tables_are_isomorphic(
    truth: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
    alphabet: Sequence[str],
    *,
    outputs: Sequence[Any] | None = None,
    start_const: int | None = None,
) -> bool:
    """Check finite-transducer isomorphism up to state relabeling."""

    try:
        true_table = normalize_transducer_table(
            truth,
            alphabet,
            outputs=outputs,
            start_const=start_const,
        )
        hyp_table = normalize_transducer_table(
            hypothesis,
            alphabet,
            outputs=outputs,
            start_const=start_const,
        )
    except Exception:
        return False

    n = int(hyp_table["n"])
    if n != int(true_table["n"]):
        return False

    return _has_output_preserving_bijection(
        parse_transducer_transitions(true_table),
        parse_transducer_transitions(hyp_table),
        n,
        int(true_table["start"]),
        int(hyp_table["start"]),
        alphabet,
    )


def transducer_counterexample(
    truth: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
    alphabet: Sequence[str],
    *,
    outputs: Sequence[Any] | None = None,
    start_const: int | None = None,
) -> list[dict[str, Any]] | None:
    """Return a shortest distinguishing trace for two finite transducers."""

    try:
        true_table = normalize_transducer_table(
            truth,
            alphabet,
            outputs=outputs,
            start_const=start_const,
        )
        hyp_table = normalize_transducer_table(
            hypothesis,
            alphabet,
            outputs=outputs,
            start_const=start_const,
        )
        true_trans = parse_transducer_transitions(true_table)
        hyp_trans = parse_transducer_transitions(hyp_table)
        cex = find_counterexample(
            int(true_table["start"]),
            int(hyp_table["start"]),
            list(alphabet),
            lambda state, action: true_trans[state][action],
            lambda state, action: hyp_trans[state][action],
        )
    except Exception:
        first = str(alphabet[0]) if alphabet else ""
        return [{"in": first, "out": None}]

    if cex is None:
        return None
    return [{"in": action, "out": output} for action, output in cex]


def _int_field(table: Mapping[str, Any], key: str) -> int:
    return _int_value(table.get(key), key)


def _int_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _homogeneous_json_type(values: Sequence[Any]) -> str | None:
    if not values:
        return None
    if all(isinstance(value, str) for value in values):
        return "string"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return "number"
    return None


def _has_output_preserving_bijection(
    true_transitions: Mapping[int, Mapping[str, tuple[int, Any]]],
    hypothesis_transitions: Mapping[int, Mapping[str, tuple[int, Any]]],
    n_states: int,
    true_start: int,
    hypothesis_start: int,
    alphabet: Sequence[str],
) -> bool:
    mapping: dict[int, int] = {hypothesis_start: true_start}
    used_true = {true_start}

    def partial_consistent(hyp_state: int, true_state: int) -> bool:
        for symbol in alphabet:
            hyp_next, hyp_output = hypothesis_transitions[hyp_state][symbol]
            true_next, true_output = true_transitions[true_state][symbol]
            if hyp_output != true_output:
                return False
            if hyp_next in mapping:
                if mapping[hyp_next] != true_next:
                    return False
            elif true_next in used_true:
                return False
        return True

    def all_mapped_consistent() -> bool:
        return all(
            partial_consistent(hyp_state, true_state)
            for hyp_state, true_state in mapping.items()
        )

    def search() -> bool:
        if not all_mapped_consistent():
            return False
        if len(mapping) == n_states:
            return True

        hyp_state = next(state for state in range(n_states) if state not in mapping)
        for true_state in range(n_states):
            if true_state in used_true:
                continue
            mapping[hyp_state] = true_state
            used_true.add(true_state)
            if search():
                return True
            used_true.remove(true_state)
            del mapping[hyp_state]
        return False

    return search()
