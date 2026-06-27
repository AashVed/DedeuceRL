"""Dataset and split compilers for TaskIR tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from datasets import Dataset

from dedeucerl.ir.types import TaskIR
from dedeucerl.ir.registry import get_task_entry
from dedeucerl.kernel.types import TaskInstance
from dedeucerl.surface.prompt import compile_prompt


def instance_to_dict(instance: TaskInstance) -> dict[str, Any]:
    return {
        "id": instance.id,
        "seed": instance.seed,
        "kernel_name": instance.kernel_name,
        "kernel_version": instance.kernel_version,
        "private": dict(instance.private),
        "params": dict(instance.params),
        "budget": int(instance.budget),
        "metadata": dict(instance.metadata),
    }


def instance_from_dict(data: Mapping[str, Any]) -> TaskInstance:
    return TaskInstance(
        id=str(data["id"]),
        seed=None if data.get("seed") is None else int(data["seed"]),
        kernel_name=str(data["kernel_name"]),
        kernel_version=str(data["kernel_version"]),
        private=dict(data.get("private", {})),
        params=dict(data.get("params", {})),
        budget=int(data["budget"]),
        metadata=dict(data.get("metadata", {})),
    )


def generate_split(
    task: TaskIR,
    *,
    seeds: list[int],
    budget: int,
    subset_name: str = "dev",
    **params: Any,
) -> dict[str, Any]:
    items = [
        {"instance": instance_to_dict(task.generator.sample(seed=seed, budget=budget, **params))}
        for seed in seeds
    ]
    return {
        "version": 2,
        "metadata": {
            "kernel": task.name,
            "kernel_version": task.version,
        },
        subset_name: {
            "kernel": task.name,
            "budget": int(budget),
            "params": dict(params),
            "items": items,
        },
    }


def save_split(split: Mapping[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")


def load_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_dataset_from_split(
    split_data: Mapping[str, Any],
    subset: str,
    *,
    feedback: bool = False,
) -> Dataset:
    if subset not in split_data:
        available = [k for k in split_data.keys() if k not in ("version", "metadata")]
        raise ValueError(f"Subset '{subset}' not found. Available: {available}")

    cfg = split_data[subset]
    items = cfg.get("items", [])
    prompts: list[list[dict[str, Any]]] = []
    answers: list[str] = []

    for item in items:
        instance = instance_from_dict(item["instance"])

        entry = get_task_entry(instance.kernel_name)
        context = entry.ir.action_context(
            instance,
            entry.ir.kernel.initial_state(instance),
            budget=instance.budget,
            queries_used=0,
            tool_calls=0,
            done=instance.budget <= 0,
        )
        runtime_contracts = entry.ir.action_contracts(context)
        prompts.append(compile_prompt(entry.ir, instance, runtime_contracts, feedback=feedback))
        answers.append(json.dumps(instance_to_dict(instance)))

    return Dataset.from_dict({"prompt": prompts, "answer": answers})
