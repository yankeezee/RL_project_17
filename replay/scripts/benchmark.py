"""Small local CPU benchmark; excludes learner/env and measures logging separately."""

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from outage_replay import ReplayBuffer
from outage_replay.buffer import METHODS
from outage_replay.diagnostics import ReplayLogger, huber


def milliseconds(values):
    values = np.asarray(values) * 1000
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/benchmark.json")
    parser.add_argument("--capacity", type=int, default=50000)
    parser.add_argument("--batches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    buffer, rng = ReplayBuffer(args.capacity, seed=args.seed), np.random.default_rng(args.seed)
    start = perf_counter()
    for k in range(args.capacity):
        buffer.add(None, current_delivered=k % 4 != 0, next_delivered=True, terminated=False)
    fill_seconds = perf_counter() - start
    ids = np.arange(args.capacity)
    buffer.update_priorities(ids, rng.lognormal(0, 1, size=args.capacity), transition_ids=ids)
    output = {"synthetic": True, "seed": args.seed, "capacity": args.capacity, "batch_size": 64,
              "batches_per_method": args.batches, "python": sys.version, "numpy": np.__version__,
              "platform": platform.platform(), "fill_seconds": fill_seconds,
              "metadata_ndarray_bytes": buffer.metadata_bytes,
              "memory_excludes": "payloads, Python objects/list overhead, batch allocations, process RSS",
              "methods": {}}
    for method in METHODS:
        for _ in range(20):
            buffer.sample(method=method)
        samples, updates = [], []
        for _ in range(args.batches):
            batch = buffer.sample(method=method, beta=.7)
            samples.append(batch.sample_seconds)
            td = rng.normal(size=64)
            start = perf_counter()
            buffer.update_priorities(batch.indices, td, transition_ids=batch.transition_ids)
            updates.append(perf_counter() - start)
        output["methods"][method] = {"sampling": milliseconds(samples),
                                      "priority_update": milliseconds(updates)}
    with tempfile.TemporaryDirectory() as temp:
        with ReplayLogger(temp, run_id="benchmark", seed=args.seed, regime="synthetic",
                          synthetic=True, sample_details=True) as logger:
            times = []
            for step in range(50):
                batch, td = buffer.sample(), rng.normal(size=64)
                start = perf_counter()
                logger.log(batch, td_errors=td, unreduced_huber=huber(td), decisions=step, optimizer_step=step)
                times.append(perf_counter() - start)
        output["batch_and_per_draw_logging"] = milliseconds(times)
        output["log_bytes_per_update"] = sum(p.stat().st_size for p in Path(temp).glob("*.jsonl")) / 50
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
