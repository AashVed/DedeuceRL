"""Pure Mealy-machine kernel utilities."""

from __future__ import annotations

from typing import Any, Mapping

from dedeucerl.core.automata import (
    generate_random_traps,
    is_fully_reachable,
    is_minimal,
    verify_trap_free_path_exists,
)
from dedeucerl.kernel.types import KernelInputError, KernelTransition, TaskInstance
from dedeucerl.utils import error_invalid_symbol
from dedeucerl.utils.rng import get_rng


ALPHABET = ["A", "B", "C"]
OUTPUTS = [0, 1, 2]


class MealyKernel:
    """Pure transition semantics for hidden Mealy-machine identification."""

    name = "mealy"
    version = "2.2"

    def initial_state(self, instance: TaskInstance) -> int:
        table = instance.private["table"]
        return int(table.get("start", 0))

    def call(
        self,
        instance: TaskInstance,
        state: Any,
        tool_name: str,
        action: Any,
    ) -> KernelTransition:
        if tool_name != "act":
            raise KeyError(tool_name)
        return self._act(instance, int(state), action)

    def _act(
        self,
        instance: TaskInstance,
        state: int,
        action: Any,
    ) -> KernelTransition:
        if not isinstance(action, Mapping):
            raise KernelInputError(error_invalid_symbol(str(action), ALPHABET))
        symbol = str(action.get("symbol", ""))
        if symbol not in ALPHABET:
            raise KernelInputError(error_invalid_symbol(symbol, ALPHABET))

        trans = parse_transitions(instance.private["table"])
        next_state, out = trans[state][symbol]
        trap_pairs = {(int(s), str(a)) for s, a in instance.private.get("trap_pairs", [])}
        return KernelTransition(
            next_state=next_state,
            observation={"out": int(out)},
            trap=(state, symbol) in trap_pairs,
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


def parse_transitions(table: Mapping[str, Any]) -> dict[int, dict[str, tuple[int, int]]]:
    raw = table.get("trans", {})
    if not isinstance(raw, Mapping):
        raise ValueError("trans must be a mapping")
    return {
        int(s): {str(a): (int(v[0]), int(v[1])) for a, v in m.items()}
        for s, m in raw.items()
    }
