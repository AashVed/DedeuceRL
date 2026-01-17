"""dedeucerl-aggregate: Aggregate evaluation results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, Dict, List


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


def aggregate_by_model(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate results by model."""
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        model = r.get("model", "unknown")
        by_model[model].append(r)

    aggregated = {}
    for model, runs in by_model.items():
        n = len(runs)
        n_success = sum(1 for r in runs if r.get("ok", False))
        n_trap = sum(1 for r in runs if r.get("trap_hit", False))
        total_queries = sum(r.get("queries_used", 0) for r in runs)
        total_reward = sum(r.get("reward", 0) for r in runs)

        aggregated[model] = {
            "model": model,
            "n_episodes": n,
            "success_rate": n_success / n if n > 0 else 0,
            "trap_rate": n_trap / n if n > 0 else 0,
            "avg_queries": total_queries / n if n > 0 else 0,
            "avg_reward": total_reward / n if n > 0 else 0,
        }

    return aggregated


def format_csv(aggregated: Dict[str, Dict[str, Any]]) -> str:
    """Format aggregated results as CSV."""
    lines = ["model,n_episodes,success_rate,trap_rate,avg_queries,avg_reward"]
    for model in sorted(aggregated.keys()):
        a = aggregated[model]
        lines.append(
            f"{a['model']},{a['n_episodes']},"
            f"{a['success_rate']:.4f},{a['trap_rate']:.4f},"
            f"{a['avg_queries']:.2f},{a['avg_reward']:.4f}"
        )
    return "\n".join(lines)


def format_json(aggregated: Dict[str, Dict[str, Any]]) -> str:
    """Format aggregated results as JSON."""
    return json.dumps(list(aggregated.values()), indent=2)


def format_markdown(aggregated: Dict[str, Dict[str, Any]]) -> str:
    """Format aggregated results as Markdown table."""
    lines = [
        "| Model | Episodes | Success Rate | Trap Rate | Avg Queries | Avg Reward |",
        "|-------|----------|--------------|-----------|-------------|------------|",
    ]
    for model in sorted(aggregated.keys()):
        a = aggregated[model]
        lines.append(
            f"| {a['model']} | {a['n_episodes']} | "
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
    aggregated = aggregate_by_model(results)

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
