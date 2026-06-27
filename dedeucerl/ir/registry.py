"""TaskIR registry."""

from __future__ import annotations

from dataclasses import dataclass

from dedeucerl.ir.mealy import build_mealy_ir
from dedeucerl.ir.types import TaskIR


@dataclass(frozen=True)
class TaskEntry:
    name: str
    ir: TaskIR


MEALY_IR = build_mealy_ir()

TASK_REGISTRY: dict[str, TaskEntry] = {
    "mealy": TaskEntry(name="mealy", ir=MEALY_IR),
}


def get_task_entry(name: str) -> TaskEntry:
    try:
        return TASK_REGISTRY[name]
    except KeyError as e:
        available = ", ".join(sorted(TASK_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}") from e
