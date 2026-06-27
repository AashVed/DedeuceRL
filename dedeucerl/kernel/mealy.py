"""Reference Mealy-machine kernel."""

from __future__ import annotations

import json
from typing import Any, Mapping

from dedeucerl.core.automata import (
    check_isomorphism_with_signatures,
    find_counterexample,
    generate_random_traps,
    is_fully_reachable,
    is_minimal,
    verify_trap_free_path_exists,
)
from dedeucerl.kernel.types import (
    KernelInputError,
    KernelJudgment,
    KernelParam,
    KernelTransition,
    TaskInstance,
    ToolContract,
)
from dedeucerl.utils import error_invalid_json, error_invalid_symbol, error_malformed_hypothesis
from dedeucerl.utils.rng import get_rng
from dedeucerl.utils.schema import validate_jsonschema


ALPHABET = ["A", "B", "C"]
OUTPUTS = [0, 1, 2]


def _hypothesis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "minimum": 1},
            "start": {"type": "integer", "const": 0},
            "trans": {
                "type": "object",
                "patternProperties": {
                    "^[0-9]+$": {
                        "type": "object",
                        "properties": {
                            sym: {
                                "type": "array",
                                "items": [{"type": "integer"}, {"type": "integer"}],
                                "minItems": 2,
                                "maxItems": 2,
                            }
                            for sym in ALPHABET
                        },
                        "required": ALPHABET,
                    }
                },
            },
        },
        "required": ["n", "start", "trans"],
    }


class MealyKernel:
    """Pure semantics for hidden Mealy-machine identification."""

    name = "mealy"
    version = "2.0"

    def initial_state(self, instance: TaskInstance) -> int:
        table = instance.private["table"]
        return int(table.get("start", 0))

    def public_observation(self, instance: TaskInstance) -> Mapping[str, Any]:
        table = instance.private["table"]
        return {
            "kernel": self.name,
            "alphabet": list(ALPHABET),
            "outputs": list(OUTPUTS),
            "n_states": int(table["n"]),
            "budget": int(instance.budget),
            "trap": bool(instance.params.get("trap", True)),
        }

    def tool_contracts(self, instance: TaskInstance, state: Any) -> list[ToolContract]:
        _ = (instance, state)
        return [
            ToolContract(
                name="act",
                kind="probe",
                description="Execute one input symbol on the hidden Mealy machine.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "enum": ALPHABET,
                            "description": "Input symbol to execute.",
                        }
                    },
                    "required": ["symbol"],
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "out": {"type": "integer"},
                        "budget_left": {"type": "integer"},
                        "t": {"type": "integer"},
                        "trap_hit": {"type": "boolean"},
                        "queries_used": {"type": "integer"},
                    },
                    "required": ["out", "budget_left", "t", "trap_hit", "queries_used"],
                },
                cost=1,
            ),
            ToolContract(
                name="submit_table",
                kind="submit",
                description="Submit a complete Mealy transition table as a JSON string.",
                args_schema={
                    "type": "object",
                    "properties": {
                        "table_json": {
                            "type": "string",
                            "description": (
                                'JSON: {"n": <int>, "start": 0, "trans": '
                                '{"0": {"A": [next_state, output], ...}, ...}}'
                            ),
                        }
                    },
                    "required": ["table_json"],
                },
                return_schema={
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "budget_left": {"type": "integer"},
                        "queries_used": {"type": "integer"},
                        "trap_hit": {"type": "boolean"},
                        "counterexample": {"type": ["array", "null"]},
                    },
                    "required": ["ok", "budget_left", "queries_used", "trap_hit"],
                },
                cost=1,
            ),
        ]

    def call(
        self,
        instance: TaskInstance,
        state: Any,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> KernelTransition | KernelJudgment:
        if tool_name == "act":
            return self._act(instance, int(state), args)
        if tool_name == "submit_table":
            return self._submit(instance, args)
        raise KeyError(tool_name)

    def _act(
        self, instance: TaskInstance, state: int, args: Mapping[str, Any]
    ) -> KernelTransition:
        symbol = str(args.get("symbol", ""))
        if symbol not in ALPHABET:
            raise KernelInputError(error_invalid_symbol(symbol, ALPHABET))

        trans = _parse_transitions(instance.private["table"])
        next_state, out = trans[state][symbol]
        trap_pairs = {(int(s), str(a)) for s, a in instance.private.get("trap_pairs", [])}
        return KernelTransition(
            next_state=next_state,
            observation={"out": int(out)},
            trap=(state, symbol) in trap_pairs,
        )

    def _submit(self, instance: TaskInstance, args: Mapping[str, Any]) -> KernelJudgment:
        raw = args.get("table_json")
        if not isinstance(raw, str):
            raise KernelInputError(error_malformed_hypothesis("table_json must be a string"))

        try:
            table = json.loads(raw)
        except Exception as e:
            raise KernelInputError(error_invalid_json("table_json")) from e

        reason = validate_jsonschema(table, _hypothesis_schema())
        if reason:
            raise KernelInputError(error_malformed_hypothesis(reason))

        if not isinstance(table, dict):
            raise KernelInputError(error_malformed_hypothesis("table_json must encode an object"))

        ok = self._isomorphic(instance.private["table"], table)
        cex = None if ok else self._counterexample(instance.private["table"], table)
        return KernelJudgment(
            ok=ok,
            observation={"ok": ok},
            counterexample=cex,
        )

    def _isomorphic(self, truth: Mapping[str, Any], hypothesis: Mapping[str, Any]) -> bool:
        try:
            n = int(hypothesis.get("n", -1))
            start = int(hypothesis.get("start", -1))
            true_n = int(truth.get("n", -2))
            true_start = int(truth.get("start", -2))
            hyp_trans = _parse_transitions(hypothesis)
            true_trans = _parse_transitions(truth)
        except Exception:
            return False

        if n != true_n or start != true_start:
            return False

        try:
            for s in range(n):
                for a in ALPHABET:
                    ns, _ = hyp_trans[s][a]
                    if not (0 <= ns < n):
                        return False
        except Exception:
            return False

        return check_isomorphism_with_signatures(
            n,
            true_start,
            start,
            ALPHABET,
            lambda s, a: true_trans[s][a],
            lambda s, a: hyp_trans[s][a],
        )

    def _counterexample(
        self, truth: Mapping[str, Any], hypothesis: Mapping[str, Any]
    ) -> list[dict[str, Any]] | None:
        try:
            true_start = int(truth.get("start", 0))
            hyp_start = int(hypothesis.get("start", 0))
            true_trans = _parse_transitions(truth)
            hyp_trans = _parse_transitions(hypothesis)
        except Exception:
            return [{"in": "A", "out": 0}]

        cex = find_counterexample(
            true_start,
            hyp_start,
            ALPHABET,
            lambda s, a: true_trans[s][a],
            lambda s, a: hyp_trans[s][a],
        )
        if cex is None:
            return None
        return [{"in": action, "out": output} for action, output in cex]


class MealySampler:
    """Deterministic sampler for Mealy task instances."""

    kernel = MealyKernel()
    params = {
        "n_states": KernelParam(
            type="int",
            description="Number of states in the hidden Mealy machine.",
            default=3,
            choices=None,
        ),
        "trap": KernelParam(
            type="bool",
            description="Whether to include trap transitions.",
            default=True,
        ),
    }

    def sample(self, *, seed: int, budget: int, **kwargs: Any) -> TaskInstance:
        n_states = int(kwargs.get("n_states", self.params["n_states"].default))
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if int(budget) < 0:
            raise ValueError("budget must be >= 0")
        trap = bool(kwargs.get("trap", self.params["trap"].default))
        private = generate_mealy_system(seed=seed, n_states=n_states, trap=trap)
        return TaskInstance(
            id=f"mealy-{seed}",
            seed=int(seed),
            kernel_name=self.kernel.name,
            kernel_version=self.kernel.version,
            private=private,
            params={"n_states": n_states, "trap": trap, "trap_ends_episode": False},
            budget=int(budget),
        )


def generate_mealy_system(seed: int, n_states: int = 3, trap: bool = True) -> dict[str, Any]:
    rng = get_rng(seed)
    n = int(n_states)
    if n < 1:
        raise ValueError("n_states must be >= 1")

    def gen_once() -> dict[int, dict[str, tuple[int, int]]]:
        trans: dict[int, dict[str, tuple[int, int]]] = {s: {} for s in range(n)}
        for s in range(n):
            trans[s]["A"] = ((s + 1) % n, rng.choice(OUTPUTS))
        for s in range(n):
            for a in [x for x in ALPHABET if x != "A"]:
                trans[s][a] = (rng.randrange(n), rng.choice(OUTPUTS))
        return trans

    for _ in range(10_000):
        trans = gen_once()
        if not is_fully_reachable(n, 0, ALPHABET, lambda s, a: trans[s][a][0]):
            continue
        if not is_minimal(n, ALPHABET, lambda s, a: trans[s][a]):
            continue
        break
    else:
        raise RuntimeError(f"Failed to generate reachable minimal Mealy machine for n_states={n}")

    trap_pairs: list[tuple[int, str]] = []
    if trap:
        target_count = max(1, n // 3)
        seen: set[tuple[int, str]] = set()
        attempts = 0
        while len(trap_pairs) < target_count and attempts < 100:
            attempts += 1
            candidates = generate_random_traps(
                n, ALPHABET, rng, n_traps=1, avoid_start=True, start_state=0
            )
            if not candidates:
                continue
            candidate = candidates[0]
            if candidate in seen:
                continue
            trial = seen | {candidate}
            if verify_trap_free_path_exists(n, 0, ALPHABET, lambda s, a: trans[s][a][0], trial):
                seen.add(candidate)
                trap_pairs.append(candidate)

    table = {
        "n": n,
        "start": 0,
        "trans": {
            str(s): {a: [ns, out] for a, (ns, out) in trans[s].items()} for s in range(n)
        },
    }
    return {"table": table, "trap_pairs": [[s, a] for s, a in trap_pairs]}


def _parse_transitions(table: Mapping[str, Any]) -> dict[int, dict[str, tuple[int, int]]]:
    raw = table.get("trans", {})
    if not isinstance(raw, Mapping):
        raise ValueError("trans must be a mapping")
    return {
        int(s): {str(a): (int(v[0]), int(v[1])) for a, v in m.items()}
        for s, m in raw.items()
    }
