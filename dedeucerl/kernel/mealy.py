"""Pure Mealy-machine kernel utilities."""

from __future__ import annotations

from typing import Any, Mapping

from dedeucerl.core.automata import (
    is_minimal,
    is_strongly_connected,
    sample_strongly_connected_transitions,
)
from dedeucerl.core.transducers import parse_transducer_transitions
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
        state: int,
        tool_name: str,
        action: Any,
    ) -> KernelTransition[int]:
        if tool_name != "act":
            raise KeyError(tool_name)
        return self._act(instance, int(state), action)

    def _act(
        self,
        instance: TaskInstance,
        state: int,
        action: Any,
    ) -> KernelTransition[int]:
        if not isinstance(action, Mapping):
            raise KernelInputError(error_invalid_symbol(str(action), ALPHABET))
        symbol = str(action.get("symbol", ""))
        if symbol not in ALPHABET:
            raise KernelInputError(error_invalid_symbol(symbol, ALPHABET))

        trans = parse_transducer_transitions(instance.private["table"])
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

    for _ in range(10_000):
        targets = sample_strongly_connected_transitions(n, ALPHABET, rng)
        trans = {s: {a: (targets[s][a], rng.choice(OUTPUTS)) for a in ALPHABET} for s in range(n)}
        if is_minimal(n, ALPHABET, lambda s, a: trans[s][a]):
            break
    else:
        raise RuntimeError(
            f"Failed to generate strongly connected minimal Mealy machine for n_states={n}"
        )

    trap_pairs: list[tuple[int, str]] = []
    if trap:
        # Keep safe recovery from every state, not just initial-state reachability.
        # A shuffled finite pool avoids repeated draws and the old 100-try ceiling.
        candidates = [(s, a) for s in range(1, n) for a in ALPHABET]
        rng.shuffle(candidates)
        blocked: set[tuple[int, str]] = set()
        for candidate in candidates:
            trial = blocked | {candidate}
            if is_strongly_connected(n, ALPHABET, lambda s, a: targets[s][a], trial):
                blocked = trial
                trap_pairs.append(candidate)
                if len(trap_pairs) == max(1, n // 3):
                    break

    table = {
        "n": n,
        "start": 0,
        "trans": {str(s): {a: [ns, out] for a, (ns, out) in trans[s].items()} for s in range(n)},
    }
    return {"table": table, "trap_pairs": [[s, a] for s, a in trap_pairs]}
