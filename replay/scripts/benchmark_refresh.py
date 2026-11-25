"""Alternating paired benchmark of old per-item vs new batched priority refresh.

Retains the first version's refresh as an explicit comparison, not another sampler.
No learner, environment or evaluation outcomes are used.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from outage_replay import ReplayBuffer
from outage_replay.buffer import EPSILON


def legacy_refresh(buffer, indices, td_errors, *, transition_ids):
    """First version's independent per-item update, for valid benchmark inputs."""
    indices, ids = np.asarray(indices), np.asarray(transition_ids)
    errors = np.asarray(td_errors, dtype=np.float64)
    if indices.ndim != 1 or ids.shape != indices.shape or errors.shape != indices.shape:
        raise ValueError("matching vectors required")
    if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("integer slots/IDs required")
    if np.any(indices < 0) or np.any(indices >= len(buffer)):
        raise IndexError("invalid slot")
    if not np.array_equal(buffer._ids[indices], ids) or not np.all(np.isfinite(errors)):
        raise ValueError("stale IDs or nonfinite errors")
    unique, inverse = np.unique(indices, return_inverse=True)
    absolute = np.zeros(len(unique))
    np.maximum.at(absolute, inverse, np.abs(errors))
    for index, td_abs in zip(unique, absolute):
        raw = td_abs + EPSILON
        buffer._raw[index], buffer._td_abs[index] = raw, td_abs
        buffer._set_leaf(index, buffer._tags[index], raw)


def stats(times):
    values = np.array(times)*1000
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/improvements/cycle4_benchmark.json")
    parser.add_argument("--batches", type=int, default=300)
    args = parser.parse_args()
    rng, old = np.random.default_rng(5411), ReplayBuffer(50_000, seed=5411)
    for k in range(old.capacity):
        old.add(None, current_delivered=k % 4 != 0, next_delivered=True, terminated=False)
    new = ReplayBuffer.from_state_dict(old.state_dict())
    timings = {"legacy": [], "batched": []}
    for step in range(args.batches + 20):
        indices, td = rng.integers(old.capacity, size=64), rng.normal(size=64)
        ids = old._ids[indices].copy()
        operations = [("legacy", lambda: legacy_refresh(old, indices, td, transition_ids=ids)),
                      ("batched", lambda: new.update_priorities(indices, td, transition_ids=ids))]
        for name, operation in operations[::1 if step % 2 == 0 else -1]:
            start = perf_counter()
            operation()
            elapsed = perf_counter()-start
            if step >= 20:
                timings[name].append(elapsed)
    for key in ("_raw", "_td_abs", "_sum", "_min", "_max"):
        np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
    report = {"synthetic": True, "capacity": 50000, "batch_size": 64, "seed": 5411,
              "measured_batches": args.batches, "warmup_batches": 20,
              "platform": platform.platform(), "numpy": np.__version__,
              "legacy": stats(timings["legacy"]), "batched": stats(timings["batched"]),
              "median_speedup": float(np.median(timings["legacy"])/np.median(timings["batched"])),
              "all_priority_trees_bitwise_identical": True,
              "scope": "priority refresh only; identical inputs; alternating order; one local CPU run"}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
