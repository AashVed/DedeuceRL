"""Measure frozen generator cohorts; no model receives these private diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import statistics
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path
from functools import lru_cache


DRAW = """
import json, sys, time, importlib.metadata
from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.surface.dataset import instance_to_dict
request = json.load(sys.stdin)
ir = TASK_REGISTRY['mealy'].ir
rows = []
start = time.monotonic()
for n in request['sizes']:
    for seed in range(request['seeds']):
        before = time.monotonic()
        instance = ir.generator.sample(seed=seed, budget=200, n_states=n, trap=True)
        rows.append({'instance': instance_to_dict(instance), 'seconds': time.monotonic()-before})
json.dump({'package': importlib.metadata.version('dedeucerl'), 'task_version': ir.version,
           'seconds': time.monotonic()-start, 'rows': rows}, sys.stdout)
"""


def distances(table, start, blocked=()):
    found, pending = {start: 0}, deque([start])
    while pending:
        state = pending.popleft()
        for action, (target, _) in table["trans"][str(state)].items():
            if (state, action) not in blocked and target not in found:
                found[target] = found[state] + 1
                pending.append(target)
    return found


def separating_depths(table):
    """Shortest distinguishing-word length via reverse BFS on unordered state pairs."""
    n = table["n"]
    pairs = [(s, t) for s in range(n) for t in range(s + 1, n)]
    reverse = {pair: set() for pair in pairs}
    depths, pending = {}, deque()
    trans = table["trans"]
    for pair in pairs:
        s, t = pair
        for action in trans[str(s)]:
            ns, out_s = trans[str(s)][action]
            nt, out_t = trans[str(t)][action]
            if out_s != out_t:
                if pair not in depths:
                    depths[pair] = 1
                    pending.append(pair)
            elif ns != nt:
                reverse[tuple(sorted((ns, nt)))].add(pair)
    while pending:
        pair = pending.popleft()
        for predecessor in reverse[pair]:
            if predecessor not in depths:
                depths[predecessor] = depths[pair] + 1
                pending.append(predecessor)
    return depths, len(pairs)


def metrics(instance):
    table = instance["private"]["table"]
    n, trans = table["n"], table["trans"]
    actions = list(trans["0"])
    traps = {tuple(pair) for pair in instance["private"]["trap_pairs"]}
    full = [distances(table, start) for start in range(n)]
    safe = [distances(table, start, traps) for start in range(n)]
    cycles, coverage, merges, loops = {}, {}, {}, {}
    for action in actions:
        target_set = {trans[str(s)][action][0] for s in range(n)}
        seen, state = set(), 0
        while state not in seen:
            seen.add(state)
            state = trans[str(state)][action][0]
        cycles[action] = len(seen) == n and state == 0
        coverage[action] = len(seen) / n
        merges[action] = n - len(target_set)
        loops[action] = sum(trans[str(s)][action][0] == s for s in range(n))
    depths, pair_count = separating_depths(table)
    outputs = [out for row in trans.values() for _, out in row.values()]
    hamiltonian = None
    if n <= 12:

        @lru_cache(None)
        def visit(state, mask):
            if mask == (1 << n) - 1:
                return any(target == 0 for target, _ in trans[str(state)].values())
            return any(
                not mask & (1 << target) and visit(target, mask | (1 << target))
                for target, _ in trans[str(state)].values()
            )

        hamiltonian = visit(0, 1)
    canonical = []
    labels, pending = {0: 0}, [0]
    for state in pending:
        row = []
        for action in actions:
            target, output = trans[str(state)][action]
            if target not in labels:
                labels[target] = len(labels)
                pending.append(target)
            row.append((labels[target], output))
        canonical.append(row)
    return {
        "strongly_connected": all(len(d) == n for d in full),
        "safely_strongly_connected": all(len(d) == n for d in safe),
        "safely_reachable": len(safe[0]) == n,
        "minimal": len(depths) == pair_count,
        "action_full_cycle": cycles,
        "hamiltonian_cycle": hamiltonian,
        "action_coverage": coverage,
        "action_merges": merges,
        "action_self_loops": loops,
        "access_depth": max(full[0].values()),
        "diameter": max(max(d.values()) for d in full),
        "max_separating_depth": max(depths.values(), default=0),
        "ambiguous_one_step_pairs": pair_count - sum(d == 1 for d in depths.values()),
        "output_counts": dict(Counter(outputs)),
        "trap_count": len(traps),
        "rooted_isomorphism_hash": hashlib.sha256(json.dumps(canonical).encode()).hexdigest(),
    }


def summarize(rows):
    summaries = {}
    for n in sorted({row["instance"]["params"]["n_states"] for row in rows}):
        selected = [row for row in rows if row["instance"]["params"]["n_states"] == n]
        records = [row["metrics"] for row in selected]
        count = len(records)
        summary = {"count": count}
        for key in (
            "strongly_connected",
            "safely_strongly_connected",
            "safely_reachable",
            "minimal",
        ):
            summary[key] = sum(r[key] for r in records)
        for key in ("action_full_cycle", "action_coverage", "action_merges", "action_self_loops"):
            summary[key] = {a: statistics.mean(r[key][a] for r in records) for a in "ABC"}
        for key in ("access_depth", "diameter", "max_separating_depth", "trap_count"):
            summary[key] = dict(sorted(Counter(r[key] for r in records).items()))
        summary["unique_rooted_machines"] = len({r["rooted_isomorphism_hash"] for r in records})
        if n <= 12:
            summary["hamiltonian_cycle"] = sum(r["hamiltonian_cycle"] for r in records)
        summary["mean_ambiguous_one_step_pairs"] = statistics.mean(
            r["ambiguous_one_step_pairs"] for r in records
        )
        summary["mean_generation_ms"] = 1000 * statistics.mean(row["seconds"] for row in selected)
        summary["max_generation_ms"] = 1000 * max(row["seconds"] for row in selected)
        summaries[n] = summary
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 12, 32, 64])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    request = {"seeds": args.seeds, "sizes": args.sizes}
    for label, python in (
        ("cycle", str(args.baseline_python.absolute())),
        ("uniform", sys.executable),
    ):
        result = subprocess.run(
            [python, "-I", "-c", DRAW],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            check=True,
            cwd=args.out_dir,
        )
        data = json.loads(result.stdout)
        for row in data["rows"]:
            row["metrics"] = metrics(row["instance"])
        summary = {
            "request": request,
            "package": data["package"],
            "task_version": data["task_version"],
            "seconds": data["seconds"],
            "summary": summarize(data["rows"]),
        }
        (args.out_dir / f"{label}.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.out_dir / f"{label}_instances.json.gz").write_bytes(
            gzip.compress((json.dumps(data) + "\n").encode(), mtime=0)
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
