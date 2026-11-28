"""Paired comparisons and summary tables for the evaluation results.

For one (method, training regime, test condition), the results form a matrix
``H[k, e]``: training seed ``k`` x shared held-out seed ``e``. Two methods are
paired on both indices (same training seed numbers, same evaluation seeds).

* Seed-level differences: ``mean_e H_A[k, e] - mean_e H_B[k, e]`` for each k.
* Hierarchical paired bootstrap: resample training seeds with replacement, then
  evaluation seeds with replacement within each, applying the same indices to
  both methods; the 95 % percentile interval of the mean difference is reported.
  An interval that contains 0 leaves the performance claim unresolved.

With three training seeds the outer level has very few distinct resamples, so
the intervals are coarse (a limitation stated in the proposal).

Examples:
    python paired_stats.py results/eval.csv                 # primary comparison
    python paired_stats.py results/eval.csv --all           # all planned comparisons
    python paired_stats.py results/eval.csv --summary
    python paired_stats.py results/eval.csv --a grouped --b mixture --metric return
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

PRIMARY = {"a": "grouped", "b": "per", "train_regime": "burst25_L4", "condition": "burst25_L8"}
# grouped vs. each baseline, plus mixture vs. PER (is a generic reduction of
# prioritization enough to explain a benefit?)
PLANNED_PAIRS = [("grouped", "per"), ("grouped", "mixture"), ("grouped", "uniform"),
                 ("mixture", "per")]
METRICS = ("hit", "return", "miss", "timeout")


def read_results(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["train_seed"] = int(r["train_seed"])
        r["eval_seed"] = int(r["eval_seed"])
        r["hit"] = int(r["hit"])
        r["return"] = float(r["return"])
        r["missing_fraction"] = float(r["missing_fraction"])
    return rows


def metric_value(row, metric):
    if metric == "hit":
        return float(row["hit"])
    if metric == "return":
        return row["return"]
    if metric in ("miss", "timeout"):
        return float(row["outcome"] == metric)
    raise ValueError(f"unknown metric {metric}")


def outcome_matrix(rows, method, train_regime, condition, metric="hit"):
    """Return ``H[k, e]`` and the sorted training / evaluation seed lists."""
    cells = {(r["train_seed"], r["eval_seed"]): metric_value(r, metric) for r in rows
             if r["method"] == method and r["train_regime"] == train_regime
             and r["condition"] == condition}
    if not cells:
        raise ValueError(f"no results for {method} / {train_regime} / {condition}")
    tseeds = sorted({k for k, _ in cells})
    eseeds = sorted({e for _, e in cells})
    H = np.full((len(tseeds), len(eseeds)), np.nan)
    for (k, e), v in cells.items():
        H[tseeds.index(k), eseeds.index(e)] = v
    if np.isnan(H).any():
        raise ValueError(f"incomplete seed grid for {method} / {train_regime} / {condition}")
    return H, tseeds, eseeds


def hierarchical_bootstrap(a, b, n_boot=10_000, seed=0):
    """Paired two-level bootstrap replicates of ``mean(a) - mean(b)``."""
    if a.shape != b.shape:
        raise ValueError("paired matrices must have the same shape")
    rng = np.random.default_rng(seed)
    n_k, n_e = a.shape
    d = a - b
    reps = np.empty(n_boot)
    for start in range(0, n_boot, 1000):  # chunks keep memory small
        m = min(1000, n_boot - start)
        ks = rng.integers(0, n_k, size=(m, n_k))
        es = rng.integers(0, n_e, size=(m, n_k, n_e))
        reps[start:start + m] = d[ks[:, :, None], es].mean(axis=(1, 2))
    return reps


def compare(rows, a, b, train_regime, condition, metric="hit", n_boot=10_000, level=0.95):
    Ha, ka, ea = outcome_matrix(rows, a, train_regime, condition, metric)
    Hb, kb, eb = outcome_matrix(rows, b, train_regime, condition, metric)
    if ka != kb or ea != eb:
        raise ValueError(f"{a} and {b} were not evaluated on the same seeds")
    reps = hierarchical_bootstrap(Ha, Hb, n_boot)
    lo, hi = np.quantile(reps, [(1 - level) / 2, (1 + level) / 2])
    return {"a": a, "b": b, "train_regime": train_regime, "condition": condition,
            "metric": metric, "mean_a": Ha.mean(), "mean_b": Hb.mean(),
            "diff": (Ha - Hb).mean(), "ci_low": lo, "ci_high": hi,
            "resolved": bool(lo > 0 or hi < 0), "train_seeds": ka,
            "seed_diffs": Ha.mean(axis=1) - Hb.mean(axis=1), "n_eval": len(ea)}


def format_comparison(c):
    seeds = ", ".join(f"s{k}: {d:+.3f}" for k, d in zip(c["train_seeds"], c["seed_diffs"]))
    verdict = "resolved" if c["resolved"] else "unresolved (CI contains 0)"
    return (f"{c['a']} - {c['b']} | train {c['train_regime']} | test {c['condition']} | "
            f"{c['metric']}\n"
            f"  {c['a']} {c['mean_a']:.3f}  {c['b']} {c['mean_b']:.3f}  "
            f"diff {c['diff']:+.3f}  95% CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]  {verdict}\n"
            f"  seed-level differences: {seeds}  ({c['n_eval']} eval seeds)")


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["train_regime"], r["condition"])].append(r)
    table = []
    for (m, t, c), rs in sorted(groups.items()):
        table.append({
            "method": m, "train_regime": t, "condition": c, "episodes": len(rs),
            "train_seeds": len({r["train_seed"] for r in rs}),
            "hit_rate": np.mean([r["hit"] for r in rs]),
            "return": np.mean([r["return"] for r in rs]),
            "miss_rate": np.mean([r["outcome"] == "miss" for r in rs]),
            "timeout_rate": np.mean([r["outcome"] == "timeout" for r in rs]),
            "missing_fraction": np.mean([r["missing_fraction"] for r in rs]),
        })
    return table


def print_summary(table):
    cols = list(table[0])
    print("  ".join(f"{c:>16}" for c in cols))
    for t in table:
        print("  ".join(f"{t[c]:>16.3f}" if isinstance(t[c], float) else f"{t[c]:>16}"
                        for c in cols))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results")
    parser.add_argument("--summary", action="store_true", help="per-cell summary table")
    parser.add_argument("--all", action="store_true",
                        help="all planned pairs, both training regimes, all test conditions")
    parser.add_argument("--a", default=PRIMARY["a"])
    parser.add_argument("--b", default=PRIMARY["b"])
    parser.add_argument("--train_regime", default=PRIMARY["train_regime"])
    parser.add_argument("--condition", default=PRIMARY["condition"])
    parser.add_argument("--metric", default="hit", choices=METRICS)
    parser.add_argument("--n_boot", type=int, default=10_000)
    args = parser.parse_args()

    rows = read_results(args.results)
    if args.summary:
        print_summary(summary(rows))
        return
    if args.all:
        cells = defaultdict(set)  # method -> {(train_regime, condition)}
        for r in rows:
            cells[r["method"]].add((r["train_regime"], r["condition"]))
        for a, b in PLANNED_PAIRS:
            common = sorted(cells[a] & cells[b])
            if not common:
                print(f"skipping {a} - {b}: no common results")
                continue
            for t, c in common:
                print(format_comparison(compare(rows, a, b, t, c, args.metric, args.n_boot)))
        return
    print(format_comparison(compare(rows, args.a, args.b, args.train_regime, args.condition,
                                    args.metric, args.n_boot)))


if __name__ == "__main__":
    main()
