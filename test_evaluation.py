"""Checks of the paired evaluation and statistics (python test_evaluation.py)."""
import numpy as np

from evaluate import parse_checkpoint, run_episode
from paired_stats import compare, hierarchical_bootstrap, summary


def fake_rows(method, H, regime="burst25_L4", condition="burst25_L8", seeds=None):
    seeds = seeds if seeds is not None else range(H.shape[0])
    rows = []
    for k, row in zip(seeds, H):
        for e, h in enumerate(row):
            rows.append({"method": method, "train_regime": regime, "train_seed": k,
                         "condition": condition, "eval_seed": 9_000_000 + e, "hit": int(h),
                         "return": float(h) - 0.05, "outcome": "hit" if h else "miss",
                         "missing_fraction": 0.25})
    return rows


rng = np.random.default_rng(0)

# 1) checkpoint names written by train.py
assert parse_checkpoint("ddqn_per_burst25_L4_seed0.pt") == ("per", "burst25_L4", 0)
assert parse_checkpoint("runs/ddqn_grouped_clean_seed12.pt") == ("grouped", "clean", 12)
for bad in ("per_clean_seed0.pt", "ddqn_per_clean.pt"):
    try:
        parse_checkpoint(bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass

# 2) identical methods: every bootstrap replicate is exactly 0
H = (rng.random((3, 100)) < 0.3).astype(float)
assert np.all(hierarchical_bootstrap(H, H, n_boot=500) == 0)

# 3) a clear improvement is detected, with the right point estimate
B = rng.random((3, 100)) < 0.3
A = B | (rng.random((3, 100)) < 0.3)
c = compare(fake_rows("grouped", A) + fake_rows("per", B), "grouped", "per",
            "burst25_L4", "burst25_L8", n_boot=2000)
assert np.isclose(c["diff"], A.mean() - B.mean())
assert c["ci_low"] > 0 and c["resolved"]
assert np.allclose(c["seed_diffs"], A.mean(axis=1) - B.mean(axis=1))

# 4) no true difference: the 95 % interval covers 0 most of the time
covered = 0
for trial in range(40):
    X = (rng.random((3, 100)) < 0.3).astype(float)
    Y = (rng.random((3, 100)) < 0.3).astype(float)
    c = compare(fake_rows("grouped", X) + fake_rows("per", Y), "grouped", "per",
                "burst25_L4", "burst25_L8", n_boot=1000)
    covered += c["ci_low"] <= 0 <= c["ci_high"]
assert covered >= 32, covered  # nominal 95 %; coarse with 3 training seeds
print(f"null case: CI covered 0 in {covered}/40 trials")

# 5) unpaired inputs are refused
try:
    compare(fake_rows("grouped", A) + fake_rows("per", B[:2]), "grouped", "per",
            "burst25_L4", "burst25_L8")
    raise AssertionError("unpaired training seeds accepted")
except ValueError:
    pass

# 6) summary table
table = summary(fake_rows("grouped", A) + fake_rows("per", B))
assert {t["method"] for t in table} == {"grouped", "per"}
assert np.isclose([t for t in table if t["method"] == "per"][0]["hit_rate"], B.mean())

# 7) on the real engine, the same seed gives the same start and mask to any policy
from env import PredictPositionEnv, eval_seeds, make_outage  # noqa: E402

env = PredictPositionEnv(outage=make_outage("burst25_L8"))
seed = eval_seeds(1)[0]
r_noop = run_episode(env, lambda obs: 0, seed)
r_noop2 = run_episode(env, lambda obs: 0, seed)
assert r_noop == r_noop2, "evaluation must be deterministic for a fixed seed"
assert r_noop["outcome"] == "timeout" and r_noop["length"] == 75
env.close()
print("evaluation is deterministic for a fixed seed")

print("\nall evaluation checks passed")
