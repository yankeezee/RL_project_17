"""Measure frozen analysis on one reproducible shared 50k-transition CSV."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from outage_replay import ReplayBuffer
from outage_replay.frozen import analyze_frozen, export_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, choices=["before", "after"])
    parser.add_argument("--root", type=Path, default=Path("results/improvements/frozen_benchmark"))
    args = parser.parse_args()
    source = args.root / "source" / "frozen.csv"
    if not source.exists():
        buffer, rng = ReplayBuffer(50_000), np.random.default_rng(5412)
        for k in range(50_000):
            buffer.add(None, current_delivered=k % 4 != 0, next_delivered=True, terminated=False)
        tags = buffer.snapshot()["tag"]
        buffer.update_priorities(np.arange(50000), rng.lognormal(2*tags, 1), transition_ids=np.arange(50000))
        export_frozen(buffer, source.parent, synthetic=True, metadata={"seed": 5412})
    durations = []
    output = args.root / args.label
    for _ in range(3):
        start = perf_counter()
        analyze_frozen(source, output, beta=.7)
        durations.append(perf_counter()-start)
    report = {"synthetic": True, "label": args.label, "seed": 5412, "n": 50000,
              "repetitions": 3, "seconds": durations, "median_seconds": float(np.median(durations)),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "probabilities_csv_sha256": hashlib.sha256((output/"probabilities.csv").read_bytes()).hexdigest(),
              "summary_json_sha256": hashlib.sha256((output/"summary.json").read_bytes()).hexdigest()}
    (args.root/f"{args.label}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
