"""Reproduce proposal-acceptance and exact-count memory measurements."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from dedeucerl.core.automata import _partition_counts, _sample_surjection, is_strongly_connected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for n in (4, 8, 12, 32, 64, 128, 256):
        for method, draws in (("independent", 10_000), ("surjection", 1000)):
            rng = random.Random(888 + n)
            started = time.monotonic()
            counts = _partition_counts(n * 3, n) if method == "surjection" else None
            setup = time.monotonic() - started
            # Count shared small integers once; include the list storage.
            integers = {} if counts is None else {id(v): v for row in counts for v in row}
            memory = (
                0
                if counts is None
                else (
                    sys.getsizeof(counts)
                    + sum(sys.getsizeof(row) for row in counts)
                    + sum(sys.getsizeof(v) for v in integers.values())
                )
            )
            accepted = 0
            started = time.monotonic()
            for _ in range(draws):
                targets = (
                    [rng.randrange(n) for _ in range(3 * n)]
                    if counts is None
                    else _sample_surjection(counts, n, rng)
                )
                accepted += is_strongly_connected(n, [0, 1, 2], lambda s, a: targets[3 * s + a])
            record = {
                "n_states": n,
                "method": method,
                "draws": draws,
                "accepted": accepted,
                "count_table_bytes": memory,
                "setup_seconds": setup,
                "proposal_seconds": time.monotonic() - started,
            }
            results.append(record)
            print(json.dumps(record), flush=True)
    args.out.write_text(
        json.dumps({"seed": "888 + n_states", "python": sys.version, "results": results}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
