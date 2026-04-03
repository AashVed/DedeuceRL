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


def aggregate_by_group(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate results by model + skin + split hash."""
    by_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        model = str(r.get("model", "unknown"))
        skin = str(r.get("skin", "unknown"))
        split_hash = str(r.get("split_hash", "unknown"))
        by_group[(model, skin, split_hash)].append(r)

    aggregated: List[Dict[str, Any]] = []
    for model, skin, split_hash in sorted(by_group.keys()):
        runs = by_group[(model, skin, split_hash)]
        n_runs = len(runs)
        episode_ids = {
            int(r["episode_idx"])
            for r in runs
            if "episode_idx" in r and isinstance(r.get("episode_idx"), int)
        }
        n_episodes = len(episode_ids)
        n_success = sum(1 for r in runs if r.get("ok", False))
        n_trap = sum(1 for r in runs if r.get("trap_hit", False))
        total_queries = sum(r.get("queries_used", 0) for r in runs)
        total_reward = sum(r.get("reward", 0) for r in runs)

        aggregated.append(
            {
                "model": model,
                "skin": skin,
                "split_hash": split_hash,
                "n_runs": n_runs,
                "n_episodes": n_episodes,
                "success_rate": n_success / n_runs if n_runs > 0 else 0,
                "trap_rate": n_trap / n_runs if n_runs > 0 else 0,
                "avg_queries": total_queries / n_runs if n_runs > 0 else 0,
                "avg_reward": total_reward / n_runs if n_runs > 0 else 0,
            }
        )

    return aggregated


def format_csv(aggregated: List[Dict[str, Any]]) -> str:
    """Format aggregated results as CSV."""
    lines = [
        "model,skin,split_hash,n_runs,n_episodes,success_rate,trap_rate,avg_queries,avg_reward"
    ]
    for a in aggregated:
        lines.append(
            f"{a['model']},{a['skin']},{a['split_hash']},{a['n_runs']},{a['n_episodes']},"
            f"{a['success_rate']:.4f},{a['trap_rate']:.4f},"
            f"{a['avg_queries']:.2f},{a['avg_reward']:.4f}"
        )
    return "\n".join(lines)


def format_json(aggregated: List[Dict[str, Any]]) -> str:
    """Format aggregated results as JSON."""
    return json.dumps(aggregated, indent=2)


def format_markdown(aggregated: List[Dict[str, Any]]) -> str:
    """Format aggregated results as Markdown table."""
    lines = [
        (
            "| Model | Skin | Split Hash | Runs | Episodes | Success Rate | Trap Rate | "
            "Avg Queries | Avg Reward |"
        ),
        (
            "|-------|------|------------|------|----------|--------------|-----------|"
            "-------------|------------|"
        ),
    ]
    for a in aggregated:
        lines.append(
            f"| {a['model']} | {a['skin']} | {a['split_hash']} | {a['n_runs']} | "
            f"{a['n_episodes']} | "
            f"{a['success_rate']:.1%} | {a['trap_rate']:.1%} | "
            f"{a['avg_queries']:.1f} | {a['avg_reward']:.3f} |"
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
    aggregated = aggregate_by_group(results)

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
