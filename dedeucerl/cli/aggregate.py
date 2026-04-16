"""dedeucerl-aggregate: Aggregate evaluation results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="dedeucerl-aggregate",
        description="Aggregate DedeuceRL evaluation results into a leaderboard.",
    )

    parser.add_argument(
        "input",
        nargs="+",
        type=str,
        help="Input JSONL file(s) to aggregate.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        type=str,
        help="Output CSV file (default: stdout).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "markdown"],
        default="csv",
        help="Output format.",
    )

    return parser.parse_args()


def load_results(paths: List[str]) -> List[Dict[str, Any]]:
    """Load results from JSONL files."""
    results = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def _format_optional_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _group_rollouts(runs: List[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    per_episode: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in runs:
        try:
            episode_idx = int(row["episode_idx"])
            rollout = int(row.get("rollout", 0))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid episode_idx/rollout in row: {row!r}") from e

        if rollout in per_episode[episode_idx]:
            raise ValueError(
                f"Duplicate result row for episode_idx={episode_idx}, rollout={rollout}."
            )

        per_episode[episode_idx][rollout] = row
    return dict(per_episode)


def _compute_max_complete_k(per_episode: Dict[int, Dict[int, Dict[str, Any]]]) -> int:
    if not per_episode:
        return 0

    k = 0
    while all(k in rollouts for rollouts in per_episode.values()):
        k += 1
    return k


def _episode_success_within_k(rollouts: Dict[int, Dict[str, Any]], k: int) -> bool:
    return any(bool(rollouts[r].get("ok", False)) for r in range(k))


def aggregate_by_group(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate results by model + skin + split hash + eval config."""
    by_group: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        model = str(r.get("model", "unknown"))
        skin = str(r.get("skin", "unknown"))
        split_hash = str(r.get("split_hash", "unknown"))
        eval_config_hash = str(r.get("eval_config_hash", "legacy"))
        by_group[(model, skin, split_hash, eval_config_hash)].append(r)

    aggregated: List[Dict[str, Any]] = []
    for model, skin, split_hash, eval_config_hash in sorted(by_group.keys()):
        runs = by_group[(model, skin, split_hash, eval_config_hash)]
        n_runs = len(runs)
        per_episode = _group_rollouts(runs)
        n_episodes = len(per_episode)
        max_complete_k = _compute_max_complete_k(per_episode)
        n_success = sum(1 for r in runs if r.get("ok", False))
        n_trap = sum(1 for r in runs if r.get("trap_hit", False))
        total_queries = sum(r.get("queries_used", 0) for r in runs)
        total_reward = sum(r.get("reward", 0) for r in runs)
        n_episode_success_at_1 = (
            sum(1 for rollouts in per_episode.values() if _episode_success_within_k(rollouts, 1))
            if max_complete_k >= 1
            else 0
        )
        pass_at_1 = n_episode_success_at_1 / n_episodes if n_episodes > 0 else 0.0

        n_episode_success_at_3 = None
        pass_at_3 = None
        if max_complete_k >= 3:
            n_episode_success_at_3 = sum(
                1 for rollouts in per_episode.values() if _episode_success_within_k(rollouts, 3)
            )
            pass_at_3 = n_episode_success_at_3 / n_episodes if n_episodes > 0 else 0.0

        aggregated.append(
            {
                "model": model,
                "skin": skin,
                "split_hash": split_hash,
                "eval_config_hash": eval_config_hash,
                "n_runs": n_runs,
                "n_episodes": n_episodes,
                "success_rate": n_success / n_runs if n_runs > 0 else 0,
                "trap_rate": n_trap / n_runs if n_runs > 0 else 0,
                "avg_queries": total_queries / n_runs if n_runs > 0 else 0,
                "avg_reward": total_reward / n_runs if n_runs > 0 else 0,
                "max_complete_k": max_complete_k,
                "n_episode_success_at_1": n_episode_success_at_1,
                "pass_at_1": pass_at_1,
                "n_episode_success_at_3": n_episode_success_at_3,
                "pass_at_3": pass_at_3,
            }
        )

    return aggregated


def format_csv(aggregated: List[Dict[str, Any]]) -> str:
    """Format aggregated results as CSV."""
    lines = [
        (
            "model,skin,split_hash,eval_config_hash,n_runs,n_episodes,success_rate,trap_rate,"
            "avg_queries,avg_reward,max_complete_k,n_episode_success_at_1,pass_at_1,"
            "n_episode_success_at_3,pass_at_3"
        )
    ]
    for a in aggregated:
        n_episode_success_at_3 = (
            "" if a["n_episode_success_at_3"] is None else str(a["n_episode_success_at_3"])
        )
        pass_at_3 = "" if a["pass_at_3"] is None else f"{a['pass_at_3']:.4f}"
        lines.append(
            f"{a['model']},{a['skin']},{a['split_hash']},{a['eval_config_hash']},"
            f"{a['n_runs']},{a['n_episodes']},"
            f"{a['success_rate']:.4f},{a['trap_rate']:.4f},"
            f"{a['avg_queries']:.2f},{a['avg_reward']:.4f},{a['max_complete_k']},"
            f"{a['n_episode_success_at_1']},{a['pass_at_1']:.4f},"
            f"{n_episode_success_at_3},{pass_at_3}"
        )
    return "\n".join(lines)


def format_json(aggregated: List[Dict[str, Any]]) -> str:
    """Format aggregated results as JSON."""
    return json.dumps(aggregated, indent=2)


def format_markdown(aggregated: List[Dict[str, Any]]) -> str:
    """Format aggregated results as Markdown table."""
    lines = [
        (
            "| Model | Skin | Split Hash | Eval Config | Runs | Episodes | Success Rate | "
            "Trap Rate | Avg Queries | Avg Reward | Max k | Pass@1 | Pass@3 |"
        ),
        (
            "|-------|------|------------|-------------|------|----------|--------------|"
            "-----------|-------------|------------|-------|--------|--------|"
        ),
    ]
    for a in aggregated:
        lines.append(
            f"| {a['model']} | {a['skin']} | {a['split_hash']} | {a['eval_config_hash']} | "
            f"{a['n_runs']} | "
            f"{a['n_episodes']} | "
            f"{a['success_rate']:.1%} | {a['trap_rate']:.1%} | "
            f"{a['avg_queries']:.1f} | {a['avg_reward']:.3f} | {a['max_complete_k']} | "
            f"{a['pass_at_1']:.1%} | {_format_optional_percent(a['pass_at_3'])} |"
        )
    return "\n".join(lines)


def main():
    """Entry point for dedeucerl-aggregate CLI."""
    args = parse_args()

    # Load results
    results = load_results(args.input)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Aggregate
    try:
        aggregated = aggregate_by_group(results)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Format output
    if args.format == "csv":
        output = format_csv(aggregated)
    elif args.format == "json":
        output = format_json(aggregated)
    else:
        output = format_markdown(aggregated)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
