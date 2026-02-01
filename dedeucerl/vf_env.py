"""Verifiers-compatible environment entrypoint for any DedeuceRL skin.

Usage in vf-rl config:

  [env]
  id = "dedeucerl.vf_env"

  [env.args]
  skin = "mealy"
  seeds = [0, 1, 2, 3, 4]
  budget = 25
  n_states = 4
  feedback = true
"""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import verifiers as vf

from dedeucerl.core import TaskGenerator, make_rubric, make_train_rubric
from dedeucerl.skins import SKIN_REGISTRY


def _resolve_entrypoint_skin(name: str) -> Optional[type]:
    try:
        eps = metadata.entry_points()
    except Exception:
        return None

    if hasattr(eps, "select"):
        matches = eps.select(group="dedeucerl.skins", name=name)
    else:
        matches = [ep for ep in eps.get("dedeucerl.skins", []) if ep.name == name]

    for ep in matches:
        return ep.load()
    return None


def _resolve_skin_class(skin: str) -> type:
    if skin in SKIN_REGISTRY:
        return SKIN_REGISTRY[skin]

    ep_cls = _resolve_entrypoint_skin(skin)
    if ep_cls is not None:
        return ep_cls

    if ":" in skin:
        module_name, class_name = skin.split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    if "." in skin:
        module_name, class_name = skin.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    raise ValueError(f"Unknown skin '{skin}'. Use a built-in name, entrypoint, or import path.")


def _coerce_seeds(seeds: Any) -> List[int]:
    if seeds is None:
        return []
    if isinstance(seeds, int):
        return [int(seeds)]
    if isinstance(seeds, str):
        spec = seeds.strip()
        if not spec:
            return []
        out: List[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_raw, end_raw = part.split("-", 1)
                if start_raw == "" or end_raw == "":
                    raise ValueError(f"Invalid seed range: '{part}'")
                start = int(start_raw)
                end = int(end_raw)
                if end < start:
                    raise ValueError(f"Invalid seed range: '{part}' (end < start)")
                out.extend(range(start, end + 1))
            else:
                out.append(int(part))
        return out
    if isinstance(seeds, Iterable):
        return [int(s) for s in seeds]
    raise ValueError(f"Unsupported seeds type: {type(seeds).__name__}")


def _infer_n_states(cfg: Dict[str, Any], first_answer: Dict[str, Any], skin_cls: type) -> int:
    if "n_states" in cfg:
        try:
            return int(cfg["n_states"])
        except Exception:
            pass

    if isinstance(first_answer, dict) and "table" in first_answer:
        table = first_answer.get("table", {})
        try:
            return int(table.get("n", 5))
        except Exception:
            pass

    if isinstance(first_answer, dict) and "spec" in first_answer:
        spec = first_answer.get("spec", {})
        try:
            return int(spec.get("n_states", 3))
        except Exception:
            pass

    extractor = getattr(skin_cls, "domain_params_from_answer", None)
    if callable(extractor):
        try:
            params = extractor(first_answer)
        except Exception:
            params = None
        if isinstance(params, dict) and "n_states" in params:
            try:
                return int(params["n_states"])
            except Exception:
                pass

    return 5


def _build_dataset_from_split(
    generator: TaskGenerator,
    split_data: Dict[str, Any],
    subset: str,
    feedback: bool,
) -> Tuple[Any, Dict[str, Any]]:
    dataset = generator.build_dataset_from_split(split_data, subset, feedback=feedback)
    if subset not in split_data:
        raise ValueError(f"Subset '{subset}' not found in split data.")
    cfg = split_data[subset]
    return dataset, cfg


def _load_split_json(path: str | Path) -> Dict[str, Any]:
    import json

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Split file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _get_rubric(reward_mode: str, skin_cls: type) -> vf.Rubric:
    mode = (reward_mode or "benchmark").strip().lower()
    if mode in ("benchmark", "eval"):
        return make_rubric()
    if mode in ("train", "train_dense", "dense"):
        hook = getattr(skin_cls, "make_train_rubric", None)
        if callable(hook):
            return hook()
        return make_train_rubric()

    return make_rubric()


def _build_env_for_skin(
    skin: str,
    *,
    split_path: str | None,
    subset: str,
    seeds: Any,
    budget: Optional[int],
    feedback: bool,
    max_turns: Optional[int],
    reward_mode: str,
    skin_kwargs: Dict[str, Any],
    eval_split_path: str | None,
    eval_subset: Optional[str],
    eval_seeds: Any,
    eval_budget: Optional[int],
) -> vf.Environment:
    skin_cls = _resolve_skin_class(skin)
    generator = TaskGenerator(skin_cls)

    if split_path is not None and (seeds is not None or budget is not None or skin_kwargs):
        raise ValueError("Use split_path alone; do not combine with seeds/budget/skin args.")

    if split_path is not None:
        split_data = _load_split_json(split_path)
        dataset, cfg = _build_dataset_from_split(generator, split_data, subset, feedback)
    else:
        seed_list = _coerce_seeds(seeds)
        if not seed_list:
            raise ValueError("Provide split_path or non-empty seeds.")
        budget_val = int(budget) if budget is not None else int(skin_cls.config.default_budget)
        split_data = generator.generate_split(
            seed_list, budget_val, subset_name=subset, **skin_kwargs
        )
        dataset, cfg = _build_dataset_from_split(generator, split_data, subset, feedback)

    eval_dataset = None
    if eval_split_path is not None or eval_seeds is not None:
        if eval_split_path is not None and (eval_seeds is not None or eval_budget is not None):
            raise ValueError(
                "Use eval_split_path alone; do not combine with eval_seeds/eval_budget."
            )
        eval_subset_name = eval_subset or subset
        if eval_split_path is not None:
            eval_split_data = _load_split_json(eval_split_path)
            eval_dataset, _ = _build_dataset_from_split(
                generator, eval_split_data, eval_subset_name, feedback
            )
        else:
            eval_seed_list = _coerce_seeds(eval_seeds)
            if not eval_seed_list:
                raise ValueError("eval_seeds must be non-empty if provided.")
            eval_budget_val = (
                int(eval_budget) if eval_budget is not None else int(skin_cls.config.default_budget)
            )
            eval_split_data = generator.generate_split(
                eval_seed_list, eval_budget_val, subset_name=eval_subset_name, **skin_kwargs
            )
            eval_dataset, _ = _build_dataset_from_split(
                generator, eval_split_data, eval_subset_name, feedback
            )

    if max_turns is None:
        budget_val = int(cfg.get("budget", skin_cls.config.default_budget))
        first_answer = {}
        if dataset and len(dataset) > 0:
            try:
                import json as _json

                first_answer = _json.loads(dataset[0]["answer"])
            except Exception:
                first_answer = {}
        n_states = _infer_n_states(cfg, first_answer, skin_cls)
        max_turns = generator.derive_max_turns(
            budget=budget_val, n_states=n_states, feedback=feedback
        )

    rubric = _get_rubric(reward_mode, skin_cls)

    env = skin_cls(
        dataset=dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        feedback=feedback,
        max_turns=max_turns,
    )
    return env


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
    skin_args: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> vf.Environment:
    """
    Verifiers entrypoint to load any DedeuceRL skin as an environment.

    Supports:
    - Built-in skin names (e.g., "mealy")
    - Entry points (group: "dedeucerl.skins")
    - Import paths ("pkg.module:Class" or "pkg.module.Class")
    """
    skin_kwargs = dict(skin_args or {})
    skin_kwargs.update(kwargs)

    if skins:
        envs = [
            _build_env_for_skin(
                s,
                split_path=split_path,
                subset=subset,
                seeds=seeds,
                budget=budget,
                feedback=feedback,
                max_turns=max_turns,
                reward_mode=reward_mode,
                skin_kwargs=skin_kwargs,
                eval_split_path=eval_split_path,
                eval_subset=eval_subset,
                eval_seeds=eval_seeds,
                eval_budget=eval_budget,
            )
            for s in skins
        ]
        return vf.EnvGroup(envs=envs)

    return _build_env_for_skin(
        skin,
        split_path=split_path,
        subset=subset,
        seeds=seeds,
        budget=budget,
        feedback=feedback,
        max_turns=max_turns,
        reward_mode=reward_mode,
        skin_kwargs=skin_kwargs,
        eval_split_path=eval_split_path,
        eval_subset=eval_subset,
        eval_seeds=eval_seeds,
        eval_budget=eval_budget,
    )


__all__ = ["load_environment"]
