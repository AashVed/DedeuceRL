"""Helpers for episode selection, sharding, and split hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_index_spec(spec: str | None, *, max_len: int) -> List[int]:
    """Parse an index spec like "0-9,15,20-30" into a de-duplicated list.

    Args:
        spec: Spec string or None for all indices.
        max_len: Exclusive upper bound for indices.
    """
    if max_len < 0:
        raise ValueError("max_len must be non-negative")

    if spec is None or str(spec).strip() in ("", "all", "*"):
        return list(range(max_len))

    items: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            if start_raw == "" or end_raw == "":
                raise ValueError(f"Invalid range: '{part}'")
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"Invalid range: '{part}' (end < start)")
            items.extend(range(start, end + 1))
        else:
            items.append(int(part))

    bad = [i for i in items if i < 0 or i >= max_len]
    if bad:
        raise ValueError(f"Episode index out of bounds: {bad[0]} (max={max_len - 1})")

    # Preserve order while de-duplicating.
    seen = set()
    out: List[int] = []
    for idx in items:
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
    return out


def parse_shard(spec: str | None) -> Tuple[int, int] | None:
    """Parse a shard spec like 'i/N' into (i, N)."""
    if spec is None:
        return None
    raw = str(spec).strip()
    if raw == "":
        return None
    if "/" not in raw:
        raise ValueError("Shard must be in 'i/N' format")
    i_raw, n_raw = raw.split("/", 1)
    shard_index = int(i_raw)
    shard_count = int(n_raw)
    if shard_count <= 0:
        raise ValueError("Shard count must be > 0")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("Shard index must satisfy 0 <= i < N")
    return shard_index, shard_count


def apply_shard(indices: Iterable[int], shard_index: int, shard_count: int) -> List[int]:
    """Filter indices to the requested shard."""
    return [i for i in indices if (i % shard_count) == shard_index]


def compute_split_hash(split_path: str | Path, subset: str, feedback: bool) -> str:
    """Compute a stable hash for a split file + subset + feedback."""
    path = Path(split_path)
    payload = path.read_bytes()
    h = hashlib.sha256()
    h.update(payload)
    h.update(f"|subset={subset}|feedback={int(bool(feedback))}".encode("utf-8"))
    return h.hexdigest()
