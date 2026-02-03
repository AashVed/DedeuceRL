#!/usr/bin/env python3
"""Example: Basic evaluation workflow with DedeuceRL (Mealy skin).

This example demonstrates the complete DedeuceRL workflow:
1. Generate a task split (reproducible evaluation dataset)
2. Build a Hugging Face Dataset from the split
3. Create an environment and examine a single episode
4. Interact with the environment programmatically

Run from repository root:
    python examples/basic_eval.py

Note: This example does NOT require any LLM API keys - it demonstrates
the framework programmatically without calling external models.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from verifiers.types import State

from dedeucerl.skins import MealyEnv
from dedeucerl.core import TaskGenerator, make_rubric


def main() -> None:
    """Demonstrate basic DedeuceRL usage with the Mealy skin."""

    print("=" * 60)
    print("DedeuceRL Basic Example: Mealy Machine Identification")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # Step 1: Generate a reproducible evaluation split
    # ─────────────────────────────────────────────────────────────
    print("\n[1/4] Generating evaluation split...")

    generator = TaskGenerator(MealyEnv)
    split = generator.generate_split(
        seeds=list(range(5)),  # 5 episodes with seeds 0-4
        budget=25,  # 25 queries per episode
        subset_name="demo",
        n_states=3,  # 3-state Mealy machines
        trap=True,  # Enable trap actions
    )

    # Save to a temporary file (in real usage, save to dataset/ directory)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(split, f, indent=2)
        split_path = f.name

    print(f"    Generated {len(split['demo']['items'])} episodes")
    print(f"    Saved to: {split_path}")

    # ─────────────────────────────────────────────────────────────
    # Step 2: Build a Hugging Face Dataset from the split
    # ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building dataset...")

    dataset = generator.build_dataset(split_path, "demo", feedback=True)
    print(f"    Dataset size: {len(dataset)} episodes")
    print(f"    Columns: {dataset.column_names}")

    # ─────────────────────────────────────────────────────────────
    # Step 3: Create the environment
    # ─────────────────────────────────────────────────────────────
    print("\n[3/4] Creating environment...")

    rubric = make_rubric()
    env = MealyEnv(
        dataset=dataset,
        rubric=rubric,
        feedback=True,  # Enable counterexample feedback
        max_turns=30,
    )
    print(f"    Environment: {env.__class__.__name__}")
    print(f"    Max turns: {env.max_turns}")
    print(f"    Feedback enabled: {env.feedback_enabled}")

    # ─────────────────────────────────────────────────────────────
    # Step 4: Manual episode walkthrough (simulating an agent)
    # ─────────────────────────────────────────────────────────────
    print("\n[4/4] Manual episode walkthrough (episode 0)...")

    # Get first episode metadata
    item = dataset[0]
    meta = json.loads(item["answer"])

    # Configure environment with episode metadata
    env._configure_from_metadata(meta)

    # Initialize episode state (normally done by setup_state in verifiers)
    state = State(
        {
            "cs": env._get_start_state(),
            "budget": meta.get("budget", 25),
            "budget_init": meta.get("budget", 25),
            "queries_used": 0,
            "steps": 0,
            "trap_hit": False,
            "ok": False,
            "done": False,
        }
    )
    env._state_ref = state

    print("\n    Episode metadata:")
    print(f"      n_states: {env._n_states}")
    print(f"      budget: {state['budget']}")
    print(f"      traps: {len(env._trap_pairs)} trap actions defined")

    # Probe the hidden system with a few actions
    print("\n    Probing actions:")
    for symbol in ["A", "B", "C", "A", "B"]:
        result = json.loads(env.act(symbol))
        print(
            f"      act('{symbol}') → output={result.get('out')}, "
            f"budget_left={result.get('budget_left')}"
        )

    # Submit an incorrect hypothesis to demonstrate counterexample feedback
    print("\n    Submitting incorrect hypothesis...")
    wrong_table = json.dumps(
        {
            "n": env._n_states,
            "start": 0,
            "trans": {
                str(s): {"A": [0, 0], "B": [0, 1], "C": [0, 2]} for s in range(env._n_states)
            },
        }
    )
    result = json.loads(env.submit_table(wrong_table))
    print(f"      ok: {result.get('ok')}")
    print(f"      budget_left: {result.get('budget_left')}")

    if result.get("counterexample"):
        print(f"      counterexample: {result['counterexample']}")
        print("      (This shows where the hypothesis diverges from reality)")

    # Cleanup temporary file
    Path(split_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("Example complete! See README.md for full documentation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
