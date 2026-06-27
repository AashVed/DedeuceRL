"""Verifiers-compatible environment entrypoint for DedeuceRL TaskIR tasks."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import verifiers as vf

from dedeucerl.core import make_rubric, make_train_rubric
from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.surface import build_dataset_from_split, generate_split, load_split, make_verifiers_env


def _coerce_seeds(seeds: Any) -> list[int]:
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [seeds]
    if isinstance(seeds, str):
        out: list[int] = []
        for part in seeds.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                out.extend(range(int(start), int(end) + 1))
            else:
                out.append(int(part))
        return out
    return [int(seed) for seed in seeds]


def _get_rubric(reward_mode: str) -> vf.Rubric:
    mode = (reward_mode or "benchmark").strip().lower()
    if mode in ("train", "train_dense", "dense"):
        return make_train_rubric()
    return make_rubric()


def _build_dataset(
    *,
    kernel: str,
    split_path: str | None,
    subset: str,
    seeds: Any,
    budget: Optional[int],
    feedback: bool,
    kernel_kwargs: dict[str, Any],
):
    if kernel not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{kernel}'. Available: {sorted(TASK_REGISTRY)}")
    if split_path is not None and (seeds is not None or budget is not None or kernel_kwargs):
        raise ValueError("Use split_path alone; do not combine with seeds/budget/kernel args.")
    if split_path is not None:
        return build_dataset_from_split(load_split(split_path), subset, feedback=feedback)
    seed_list = _coerce_seeds(seeds)
    if not seed_list:
        raise ValueError("Provide split_path or non-empty seeds.")
    budget_value = int(budget) if budget is not None else 25
    split = generate_split(
        TASK_REGISTRY[kernel].ir,
        seeds=seed_list,
        budget=budget_value,
        subset_name=subset,
        **kernel_kwargs,
    )
    return build_dataset_from_split(split, subset, feedback=feedback)


def load_environment(
    *,
    skin: str = "mealy",
    skins: Optional[Sequence[str]] = None,
    split_path: Optional[str] = None,
    subset: str = "dev",
    seeds: Any = None,
    budget: Optional[int] = None,
    feedback: bool = False,
    max_turns: Optional[int] = None,
    reward_mode: str = "benchmark",
    eval_split_path: Optional[str] = None,
    eval_subset: Optional[str] = None,
    eval_seeds: Any = None,
    eval_budget: Optional[int] = None,
    skin_args: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load a DedeuceRL kernel as a Verifiers environment."""

    kernel_kwargs = dict(skin_args or {})
    kernel_kwargs.update(kwargs)

    def one(kernel_name: str):
        dataset = _build_dataset(
            kernel=kernel_name,
            split_path=split_path,
            subset=subset,
            seeds=seeds,
            budget=budget,
            feedback=feedback,
            kernel_kwargs=kernel_kwargs,
        )
        eval_dataset = None
        if eval_split_path is not None or eval_seeds is not None:
            eval_dataset = _build_dataset(
                kernel=kernel_name,
                split_path=eval_split_path,
                subset=eval_subset or subset,
                seeds=eval_seeds,
                budget=eval_budget,
                feedback=feedback,
                kernel_kwargs=kernel_kwargs,
            )
        return make_verifiers_env(
            dataset=dataset,
            eval_dataset=eval_dataset,
            feedback=feedback,
            max_turns=max_turns,
            rubric=_get_rubric(reward_mode),
        )

    if skins:
        return vf.EnvGroup(envs=[one(name) for name in skins])
    return one(skin)


__all__ = ["load_environment"]
