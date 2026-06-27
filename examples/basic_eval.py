#!/usr/bin/env python3
"""Basic DedeuceRL kernel/runtime workflow."""

from __future__ import annotations

import json

from dedeucerl.ir import TASK_REGISTRY
from dedeucerl.runtime import EpisodeRuntime
from dedeucerl.surface import build_dataset_from_split, compile_prompt, generate_split


def main() -> None:
    entry = TASK_REGISTRY["mealy"]

    split = generate_split(
        entry.ir,
        seeds=list(range(3)),
        budget=10,
        subset_name="demo",
        n_states=3,
        trap=False,
    )
    dataset = build_dataset_from_split(split, "demo", feedback=True)
    instance = entry.ir.generator.sample(seed=0, budget=10, n_states=3, trap=False)
    runtime = EpisodeRuntime(entry.ir, instance, feedback=True)
    prompt = compile_prompt(entry.ir, instance, runtime.contracts(), feedback=True)

    print("DedeuceRL Basic Example")
    print(f"Dataset rows: {len(dataset)}")
    print(f"System prompt starts: {prompt[0]['content'][:80]}...")

    for symbol in ["A", "B", "C"]:
        event = runtime.call_tool("act", {"symbol": symbol})
        print(f"act({symbol}) -> {json.dumps(event.output)}")

    submit = runtime.call_tool(
        "submit_table",
        {"table_json": json.dumps(instance.private["table"])},
    )
    print(f"submit_table(correct) -> {json.dumps(submit.output)}")


if __name__ == "__main__":
    main()
