"""Paired evaluation of frozen policies.

Every policy is run greedily on the same held-out seeds (``env.eval_seeds()``,
100 seeds) in every test condition. One seed fixes both the scenario start and
the outage mask, so all methods face identical starting states and identical
outage patterns (the mask does not depend on actions; trajectories may still
differ because the policies act differently).

Checkpoints are the ``state_dict`` files written by ``train.py`` and must be
named ``ddqn_<sampler>_<train_condition>_seed<k>.pt``; the network is rebuilt
with ``networks.mlp``. Random or other agents from ``agents.py`` can be added
as sanity references with ``--agents``.

One CSV row is written per episode. Already-complete (policy, condition) cells
of an existing output file are skipped, so an interrupted run can be resumed.

Examples:
    python evaluate.py --checkpoints checkpoints/ --out results/eval.csv --workers 6
    python evaluate.py --agents random --n_seeds 20 --out results/sanity.csv
"""
import argparse
import csv
import os
import re
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from env import PredictPositionEnv, eval_seeds, make_outage

TEST_CONDITIONS = ["clean", "indep25", "burst25_L4", "burst25_L8"]
CHECKPOINT_RE = re.compile(r"^ddqn_(?P<sampler>[a-z]+)_(?P<regime>.+)_seed(?P<seed>\d+)\.pt$")

FIELDS = ["method", "train_regime", "train_seed", "condition", "eval_seed",
          "return", "hit", "outcome", "length", "missing_fraction", "fire_actions"]


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
def parse_checkpoint(path):
    """``ddqn_per_burst25_L4_seed0.pt`` -> ("per", "burst25_L4", 0)."""
    m = CHECKPOINT_RE.match(Path(path).name)
    if m is None:
        raise ValueError(f"unexpected checkpoint name: {Path(path).name}")
    return m["sampler"], m["regime"], int(m["seed"])


def find_checkpoints(paths):
    found = []
    for p in map(Path, paths):
        files = sorted(p.glob("ddqn_*_seed*.pt")) if p.is_dir() else [p]
        for f in files:
            sampler, regime, seed = parse_checkpoint(f)
            found.append({"kind": "checkpoint", "path": str(f), "method": sampler,
                          "train_regime": regime, "train_seed": seed})
    return found


def build_policy(spec):
    """Return ``act(obs) -> action`` for a policy specification."""
    if spec["kind"] == "agent":
        from agents import make_agent
        np.random.seed(0)  # RandomAgent uses the global NumPy RNG
        agent = make_agent(spec["method"])
        return agent.act

    import torch
    from networks import DEVICE, mlp, as_tensor
    from env.doom_env import N_ACTIONS, OBS_SIZE

    torch.set_num_threads(1)
    q_net = mlp(OBS_SIZE, N_ACTIONS)
    q_net.load_state_dict(torch.load(spec["path"], map_location=DEVICE))
    q_net.eval()

    @torch.no_grad()
    def act(obs):
        return int(q_net(as_tensor(obs).unsqueeze(0)).argmax(dim=1).item())

    return act


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_episode(env, act, seed):
    obs = env.reset(seed=seed)
    done, total, fire_actions = False, 0.0, 0
    while not done:
        action = act(obs)
        fire_actions += action == 3  # FIRE chosen (only the first one shoots)
        obs, reward, done, info = env.step(action)
        total += reward
    outcome = "hit" if info["hit"] else ("timeout" if info["timeout"] else "miss")
    return {"return": round(total, 6), "hit": int(info["hit"]), "outcome": outcome,
            "length": info["t"], "missing_fraction": round(info["missing_fraction"], 6),
            "fire_actions": fire_actions}


def evaluate_cell(job):
    """Evaluate one policy in one test condition on all held-out seeds."""
    spec, condition, seeds = job
    act = build_policy(spec)
    env = PredictPositionEnv(outage=make_outage(condition))
    rows = []
    try:
        for seed in seeds:
            row = {"method": spec["method"], "train_regime": spec["train_regime"],
                   "train_seed": spec["train_seed"], "condition": condition, "eval_seed": seed}
            row.update(run_episode(env, act, seed))
            rows.append(row)
    finally:
        env.close()
    return rows


def cell_key(row):
    return (row["method"], row["train_regime"], int(row["train_seed"]), row["condition"])


def completed_cells(path, n_seeds):
    if not path.exists():
        return set()
    counts = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            counts[cell_key(row)] = counts.get(cell_key(row), 0) + 1
    return {k for k, c in counts.items() if c >= n_seeds}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="checkpoint files or directories containing them")
    parser.add_argument("--agents", nargs="*", default=[],
                        help="reference agents from agents.py, e.g. random")
    parser.add_argument("--conditions", nargs="*", default=TEST_CONDITIONS,
                        choices=TEST_CONDITIONS)
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/eval.csv"))
    args = parser.parse_args()

    specs = find_checkpoints(args.checkpoints)
    specs += [{"kind": "agent", "method": a, "train_regime": "-", "train_seed": -1}
              for a in args.agents]
    if not specs:
        parser.error("give --checkpoints and/or --agents")

    seeds = eval_seeds(args.n_seeds)
    done = completed_cells(args.out, args.n_seeds)
    jobs = [(s, c, seeds) for s in specs for c in args.conditions
            if (s["method"], s["train_regime"], s["train_seed"], c) not in done]
    print(f"{len(specs)} policies x {len(args.conditions)} conditions x {len(seeds)} seeds; "
          f"{len(jobs)} cells to run ({len(done)} already in {args.out})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    start = time.perf_counter()
    with args.out.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        if args.workers > 1:
            pool = get_context("spawn").Pool(args.workers)
            results = pool.imap_unordered(evaluate_cell, jobs)
        else:
            pool, results = None, map(evaluate_cell, jobs)
        for i, rows in enumerate(results, 1):
            writer.writerows(rows)
            f.flush()
            r0 = rows[0]
            print(f"[{i}/{len(jobs)}] {r0['method']:>8} {r0['train_regime']:>11} "
                  f"seed={r0['train_seed']:<2} {r0['condition']:>11} | "
                  f"hit {np.mean([r['hit'] for r in rows]):.3f} | "
                  f"{time.perf_counter() - start:.0f}s", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    print(f"results in {args.out}")


if __name__ == "__main__":
    main()
