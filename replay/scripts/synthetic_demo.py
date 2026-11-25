"""SYNTHETIC priorities/stream; no ViZDoom run, no agent performance estimate."""

import argparse
import json
from pathlib import Path

import numpy as np

from outage_replay import ReplayBuffer, beta_schedule
from outage_replay.buffer import METHODS
from outage_replay.diagnostics import ReplayLogger, huber
from outage_replay.frozen import analyze_frozen, export_frozen


def add_synthetic(buffer, tag, td):
    return buffer.add({"td": float(td)}, current_delivered=not bool(tag),
                      next_delivered=True, terminated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/synthetic")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    common = ReplayBuffer(1000, seed=args.seed)
    for k in range(1000):
        add_synthetic(common, k >= 800, 100 if k >= 800 else 1)
    common.update_priorities(np.arange(1000), [1]*800+[100]*200, transition_ids=np.arange(1000))
    frozen = export_frozen(common, out / "frozen", synthetic=True,
                           metadata={"seed": args.seed, "description": "800 clean TD=1, 200 outage TD=100"})
    summary = analyze_frozen(frozen, out / "frozen_analysis")
    for method in METHODS:
        buffer, stream = ReplayBuffer(256, seed=args.seed), np.random.default_rng(args.seed+1)
        for k in range(256):
            add_synthetic(buffer, k % 5 == 0, 20 if k % 5 == 0 else 1)
        buffer.update_priorities(np.arange(256), [t for t in (20 if k % 5 == 0 else 1 for k in range(256))],
                                 transition_ids=np.arange(256))
        with ReplayLogger(out / "logs", run_id=f"synthetic_{method}", seed=args.seed,
                          regime="SYNTHETIC_dynamic_stream", synthetic=True,
                          metadata={"note": "No optimizer: fixed synthetic residuals; ring stream is shared across methods"}) as logger:
            for step in range(args.steps):
                # Shared nonstationary stream, independent of replay RNG.
                for _ in range(4):
                    tag = stream.random() < (.1 if step < args.steps//2 else .45)
                    td = float(stream.lognormal(2.8 if tag else 0, .3))
                    add_synthetic(buffer, tag, td)
                decisions = 5000 + 4*step
                batch = buffer.sample(method=method, beta=beta_schedule(decisions))
                td = np.array([t["td"] for t in batch.transitions])
                logger.log(batch, td_errors=td, unreduced_huber=huber(td),
                           decisions=decisions, optimizer_step=step+1)
                buffer.update_priorities(batch.indices, td, transition_ids=batch.transition_ids)
        export_frozen(buffer, out / f"last_{method}", synthetic=True, metadata={"seed": args.seed})
    print(json.dumps({"synthetic": True, "output": str(out),
                      "frozen_outage_masses": {m: v["expected_outage_mass"] for m, v in summary["methods"].items()}}, indent=2))


if __name__ == "__main__":
    main()
