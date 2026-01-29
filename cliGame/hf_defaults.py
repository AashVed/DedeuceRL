"""HF_DATA-backed defaults for the interactive CLI game.

The game uses the same generation parameters as the HF_DATA exports
so humans can play the exact benchmark configurations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# Fallback defaults (kept in sync with HF_DATA/GENERATION.json).
_FALLBACK: Dict[str, Dict[str, Any]] = {
    "mealy": {"n_states": 4, "budget": 100, "trap": False},
    "protocol": {"n_endpoints": 5, "n_states": 4, "budget": 120, "trap": False},
    "apienv": {"n_endpoints": 7, "n_states": 7, "budget": 200, "trap": False},
    "exprpolicy": {"budget": 60, "n_public": 8, "n_hidden": 80, "trap": False},
}


@dataclass(frozen=True)
class HFDefaults:
    skin: str
    subset: str
    budget: int
    gen_kwargs: Dict[str, Any]
    split_path: Optional[str] = None


def _repo_root() -> Path:
    # cliGame/ is at repo root.
    return Path(__file__).resolve().parents[1]


def load_generation_index() -> Dict[str, Any]:
    p = _repo_root() / "HF_DATA" / "GENERATION.json"
    if not p.exists():
        # Fall back to embedded defaults.
        return {"splits": []}
    return json.loads(p.read_text(encoding="utf-8"))


def get_hf_defaults(
    skin: str,
    *,
    subset: str = "test",
) -> HFDefaults:
    """Return HF_DATA params for (skin, subset).

    The HF generation index stores params including budget. We split those into:
    - budget: episode query budget
    - gen_kwargs: args passed to SkinClass.generate_system_static()
    """
    data = load_generation_index()
    splits = data.get("splits") or []
    if not isinstance(splits, list):
        raise ValueError("HF_DATA/GENERATION.json: expected 'splits' list")

    match: Optional[Dict[str, Any]] = None
    for rec in splits:
        if not isinstance(rec, dict):
            continue
        if rec.get("skin") == skin and rec.get("subset") == subset:
            match = rec
            break

    if match is None:
        # Fall back to embedded defaults for known skins.
        params_fb = _FALLBACK.get(str(skin))
        if params_fb is None:
            available_set: set[str] = set()
            for r in splits:
                if not isinstance(r, dict):
                    continue
                v = r.get("skin")
                if v is not None:
                    available_set.add(str(v))
            available = sorted(available_set)
            raise KeyError(
                f"No HF defaults found for skin='{skin}' subset='{subset}'. Available skins: {available}"
            )
        params = dict(params_fb)
        budget = int(params["budget"])
        gen_kwargs = {k: v for k, v in params.items() if k != "budget"}
        if "trap" in gen_kwargs:
            gen_kwargs["trap"] = bool(gen_kwargs["trap"])
        return HFDefaults(
            skin=str(skin),
            subset=str(subset),
            budget=budget,
            gen_kwargs=gen_kwargs,
            split_path=None,
        )

    params = match.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"HF defaults missing params for skin='{skin}' subset='{subset}'")

    if "budget" not in params:
        raise ValueError(f"HF defaults missing budget for skin='{skin}' subset='{subset}'")
    budget = int(params["budget"])

    gen_kwargs = {k: v for k, v in params.items() if k != "budget"}
    # Normalize common keys.
    if "trap" in gen_kwargs:
        gen_kwargs["trap"] = bool(gen_kwargs["trap"])

    split_path = match.get("split_path")
    if isinstance(split_path, str) and split_path:
        split_path = str((_repo_root() / split_path).resolve())
    else:
        split_path = None

    return HFDefaults(
        skin=str(skin),
        subset=str(subset),
        budget=budget,
        gen_kwargs=gen_kwargs,
        split_path=split_path,
    )
