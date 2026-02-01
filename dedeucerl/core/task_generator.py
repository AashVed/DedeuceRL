"""Task generator for building evaluation datasets."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Protocol, Type

from datasets import Dataset


class SkinLike(Protocol):
    """Minimum surface area TaskGenerator needs from a skin.

    Intentionally typed with `Callable[..., ...]` to avoid over-constraining
    skin signatures (skins may have different kwargs).
    """

    config: Any
    generate_system_static: Any
    get_prompt_template: Any
    domain_spec: Any
    domain_params_from_answer: Any


class TaskGenerator:
    """
    Unified task generator for all skins.

    Generates reproducible evaluation datasets by:
    1. Iterating over seeds
    2. Calling skin.generate_system_static() for each seed
    3. Building prompts using skin.get_prompt_template()
    4. Serializing to split JSON format
    """

    def __init__(self, skin_cls: Type[SkinLike]):
        """
        Initialize the task generator.

        Args:
            skin_cls: The skin class to generate tasks for.
        """
        self.skin_cls = skin_cls
        self.skin_config = skin_cls.config

    def generate_split(
        self,
        seeds: List[int],
        budget: int,
        *,
        subset_name: str = "dev",
        **skin_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a split configuration for evaluation.

        Args:
            seeds: List of integer seeds for reproducible generation.
            budget: Query budget for each episode.
            subset_name: Name of the subset (e.g., 'dev', 'test').
            **skin_kwargs: Skin-specific parameters (n_states, trap, etc.)

        Returns:
            Split JSON structure ready for serialization.
        """
        items = []
        for seed in seeds:
            system = self.skin_cls.generate_system_static(seed, **skin_kwargs)
            items.append(
                {
                    "seed": seed,
                    "system": system,
                }
            )

        return {
            subset_name: {
                "budget": budget,
                "items": items,
                **skin_kwargs,
            }
        }

    def save_split(self, split: Dict[str, Any], path: str) -> None:
        """
        Save a split configuration to a JSON file.

        Args:
            split: Split configuration dict.
            path: Output file path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)

    def build_dataset(
        self,
        split_path: str,
        subset: str,
        *,
        feedback: bool = False,
    ) -> Dataset:
        """
        Build a HuggingFace Dataset from a split JSON.

        Args:
            split_path: Path to the split JSON file.
            subset: Name of the subset to load.
            feedback: Whether feedback mode is enabled.

        Returns:
            Dataset with 'prompt' and 'answer' columns.
        """
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)

        return self.build_dataset_from_split(split_data, subset, feedback=feedback)

    def build_dataset_from_split(
        self,
        split_data: Dict[str, Any],
        subset: str,
        *,
        feedback: bool = False,
    ) -> Dataset:
        """
        Build a HuggingFace Dataset from an in-memory split dict.

        Args:
            split_data: Split configuration dict.
            subset: Name of the subset to load.
            feedback: Whether feedback mode is enabled.

        Returns:
            Dataset with 'prompt' and 'answer' columns.
        """
        if subset not in split_data:
            available = [k for k in split_data.keys() if k not in ("version", "metadata")]
            raise ValueError(f"Subset '{subset}' not found. Available: {available}")

        cfg = split_data[subset]
        budget = int(cfg.get("budget", self.skin_config.default_budget))
        items = cfg.get("items", [])

        prompts = []
        answers = []

        for item in items:
            seed = item["seed"]
            system = item["system"]

            # Answer contains full system data for _configure_from_metadata.
            # The structure varies by skin, so we pass the entire system dict.
            answer_data = {
                "seed": seed,
                "budget": budget,
                **system,  # Include all system fields (skin-specific)
            }

            # Build observation using per-episode answer payload.
            # This is the most robust way to ensure vocab/tool enums match the episode.
            obs = self._build_observation_from_answer(answer_data, cfg)

            # Build prompt from skin template
            prompt = self.skin_cls.get_prompt_template(obs, feedback=feedback)

            answer = json.dumps(answer_data)

            prompts.append(prompt)
            answers.append(answer)

        return Dataset.from_dict({"prompt": prompts, "answer": answers})

    def _build_observation_from_answer(
        self, answer_data: Dict[str, Any], cfg: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the observation dict shown to the agent using domain_spec.

        This prefers per-episode answer payload (the generated system) over split-level
        config, because complex skins may sample episode-specific alphabets (endpoints,
        request variants, etc.).

        If the skin defines `domain_params_from_answer(answer_data)`, we use it.
        Otherwise, we fall back to split config only.
        """
        budget = int(cfg.get("budget", self.skin_config.default_budget))
        trap = bool(cfg.get("trap", True))

        # Never forward large/non-param fields like items.
        candidate_kwargs = dict(cfg)
        candidate_kwargs.pop("items", None)

        # Prefer episode-derived domain params when available.
        answer_kwargs: Dict[str, Any] = {}
        extractor = getattr(self.skin_cls, "domain_params_from_answer", None)
        if callable(extractor):
            try:
                raw = extractor(answer_data)
            except Exception:
                raw = None

            if isinstance(raw, dict):
                answer_kwargs = raw

        merged = dict(candidate_kwargs)
        merged.update(answer_kwargs)

        # Always prefer the effective episode values for these.
        merged["budget"] = budget
        merged["trap"] = trap

        # Never forward None values into domain_spec/build_observation.
        # Passing None can overwrite DomainSpec defaults/examples, producing
        # prompt observations like "endpoints": null.
        merged_non_null = {k: v for k, v in merged.items() if v is not None}
        merged_non_null["budget"] = budget
        merged_non_null["trap"] = trap

        sig = inspect.signature(self.skin_cls.domain_spec)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if accepts_var_kw:
            spec_kwargs = merged_non_null
        else:
            allowed = set(sig.parameters.keys())
            allowed.discard("cls")
            spec_kwargs = {k: v for k, v in merged_non_null.items() if k in allowed}

        spec = self.skin_cls.domain_spec(**spec_kwargs)

        # Populate observation fields with any values we have.
        obs_values = {k: v for k, v in merged_non_null.items() if k in spec.observation_fields}
        obs_values["budget"] = budget
        obs_values["trap"] = trap
        return spec.build_observation(**obs_values)

    def derive_max_turns(
        self,
        budget: int,
        n_states: int,
        feedback: bool = False,
    ) -> int:
        """
        Derive max_turns from budget and feedback setting.

        Args:
            budget: Query budget.
            n_states: Number of states in the system.
            feedback: Whether feedback mode is enabled.

        Returns:
            Computed max_turns value.
        """
        if feedback:
            return budget + max(3, min(10, 2 * n_states))
        else:
            return budget + 2
