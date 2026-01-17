#!/usr/bin/env python3
"""Example: APIEnv skin - Stateful SaaS API reverse engineering.

This example demonstrates the APIEnv skin, which models realistic
stateful REST API workflows including:
- Authentication (login/logout)
- Email verification
- Organization selection
- Plan upgrades
- State-dependent responses

The agent must reverse-engineer the hidden state machine by making
API calls and observing the responses (status code + schema tag).

Run from repository root:
    python examples/apienv_basic.py

Note: This example does NOT require any LLM API keys - it demonstrates
the framework programmatically without calling external models.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from verifiers.types import State

from dedeucerl.core import TaskGenerator, make_rubric
from dedeucerl.skins import APIEnv


def main() -> None:
    """Demonstrate APIEnv usage for stateful API reverse engineering."""

    print("=" * 60)
    print("DedeuceRL APIEnv Example: SaaS API Reverse Engineering")
    print("=" * 60)

    # ─────────────────────────────────────────────────────────────
    # Step 1: Generate evaluation split
    # ─────────────────────────────────────────────────────────────
    print("\n[1/4] Generating APIEnv evaluation split...")

    generator = TaskGenerator(APIEnv)
    split = generator.generate_split(
        seeds=[0, 1, 2],  # 3 episodes
        budget=30,  # 30 queries per episode
        subset_name="demo",
        n_states=5,  # 5-state workflow
        n_endpoints=5,  # Use 5 API endpoints
        trap=True,  # Enable trap calls
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(split, f, indent=2)
        split_path = f.name

    print(f"    Generated {len(split['demo']['items'])} episodes")
    print(f"    Endpoints: {split['demo']['items'][0]['system']['spec']['endpoints']}")

    # ─────────────────────────────────────────────────────────────
    # Step 2: Build dataset
    # ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building dataset...")

    dataset = generator.build_dataset(split_path, "demo", feedback=True)
    print(f"    Dataset size: {len(dataset)} episodes")

    # ─────────────────────────────────────────────────────────────
    # Step 3: Create environment
    # ─────────────────────────────────────────────────────────────
    print("\n[3/4] Creating APIEnv environment...")

    rubric = make_rubric()
    env = APIEnv(dataset=dataset, rubric=rubric, feedback=True, max_turns=60)
    print(f"    Environment: {env.__class__.__name__}")
    print(f"    Max turns: {env.max_turns}")

    # ─────────────────────────────────────────────────────────────
    # Step 4: Manual episode walkthrough
    # ─────────────────────────────────────────────────────────────
    print("\n[4/4] Manual episode walkthrough (episode 0)...")

    item = dataset[0]
    meta = json.loads(item["answer"])
    env._configure_from_metadata(meta)

    env._state_ref = State(
        {
            "cs": env._get_start_state(),
            "budget": int(meta.get("budget", 30)),
            "budget_init": int(meta.get("budget", 30)),
            "queries_used": 0,
            "steps": 0,
            "trap_hit": False,
            "ok": False,
            "done": False,
        }
    )

    spec = meta["spec"]
    print("\n    Hidden API spec:")
    print(f"      States: {spec['n_states']}")
    print(f"      Endpoints: {spec['endpoints']}")
    print(f"      Traps: {len(meta.get('trap_calls', []))} trap calls")

    # Simulate a typical authentication workflow
    print("\n    Simulating authentication workflow:")

    workflow = [
        ("POST", "/login", "valid", "Login with valid credentials"),
        ("POST", "/verify_email", "code_ok", "Verify email with correct code"),
        ("POST", "/select_org", "orgA", "Select organization A"),
        ("GET", "/projects", "list", "List projects"),
    ]

    for method, endpoint, variant, description in workflow:
        result = json.loads(env.api_call(method, endpoint, variant))
        status = result.get("status")
        schema = result.get("schema")
        budget = result.get("budget_left")
        print(f"      {description}:")
        print(f"        {method} {endpoint}#{variant} → {status} {schema} (budget: {budget})")

    # Submit the ground-truth spec (cheating for demonstration)
    print("\n    Submitting ground-truth specification...")
    gt_spec = {
        "n_states": spec["n_states"],
        "start": spec["start"],
        "transitions": spec["transitions"],
    }
    result = json.loads(env.submit_spec(json.dumps(gt_spec)))
    print(f"      ok: {result.get('ok')}")
    print(f"      budget_left: {result.get('budget_left')}")

    if result.get("ok"):
        print("      ✓ Specification accepted!")

    # Cleanup
    Path(split_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("APIEnv example complete!")
    print("This demonstrates how an agent would interact with")
    print("a realistic stateful API to infer its hidden behavior.")
    print("=" * 60)


if __name__ == "__main__":
    main()
