"""Plot real ReplayLogger JSONL and frozen-analysis CSV; labels follow metadata.

Never pool independent training runs implicitly. Each run gets its own figure.
Draw-level scatter uses a deterministic bounded-memory reservoir of up to 4000.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"uniform": "#475569", "per": "#e76f51", "mixture": "#8b5cf6", "grouped": "#178f8a"}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def reservoir(path, limit=4000):
    rng, sample = np.random.default_rng(7919), []
    for k, row in enumerate(read_jsonl(path)):
        if k < limit:
            sample.append(row)
        else:
            index = int(rng.integers(k+1))
            if index < limit:
                sample[index] = row
    return sample


def plot_frozen(directory, output):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    with (directory / "probabilities.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    methods = list(COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(methods, [summary["methods"][m]["expected_outage_mass"] for m in methods],
                color=[COLORS[m] for m in methods])
    axes[0].axhline(summary["buffer_outage_fraction"], color="black", linestyle="--", label="Buffer outage share")
    axes[0].set(ylabel="Expected replay outage fraction", ylim=(0, 1), title="One common frozen buffer")
    axes[0].legend()
    tags = np.array([int(r["tag"]) for r in rows])
    for method in methods:
        p = np.array([float(r[f"p_{method}"]) for r in rows])
        for c, style in [(0, "-"), (1, "--")]:
            values = np.sort(p[tags == c])
            if len(values):
                axes[1].step(values, np.arange(1, len(values)+1)/len(values), where="post",
                             color=COLORS[method], linestyle=style, label=f"{method}, c={c}")
    axes[1].set(xscale="log", xlabel="Per-transition probability P(i)", ylabel="Within-group ECDF",
                title="Probability distribution by group")
    axes[1].legend(fontsize=8, ncol=2)
    prefix = "SYNTHETIC - no ViZDoom experiment" if summary["synthetic"] else "Frozen buffer sampler analysis"
    fig.suptitle(f"{prefix} | N={summary['buffer_size']}", fontweight="bold")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "frozen_probabilities.png", dpi=170)
    plt.close(fig)


def plot_run(source, output, window=20):
    rows = list(read_jsonl(source))
    if not rows:
        raise ValueError(f"Empty batch log: {source}")
    identity = {(r["run_id"], r["method"], r["synthetic"]) for r in rows}
    if len(identity) != 1:
        raise ValueError("One file must describe one run/method/data origin")
    run_id, method, synthetic = identity.pop()
    if any(b["decisions"] < a["decisions"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Decision clock must be nondecreasing")
    windows = [rows[k:k+window] for k in range(0, len(rows), window)]
    x = np.array([g[-1]["decisions"] for g in windows])
    def total(g, key):
        return sum(row[key] for row in g)
    b = np.array([total(g, "batch_size") for g in windows])
    actual = np.array([sum(r["groups"]["1"]["sample_count"] for r in g) for g in windows])
    expected = np.array([sum(r["groups"]["1"]["expected_count_from_buffer_share"] for r in g) for g in windows])
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    axes[0, 0].plot(x, expected/b, label="Buffer share at sampling")
    axes[0, 0].plot(x, actual/b, label="Replay share", color=COLORS[method])
    if all("sampler_outage_probability" in r for r in rows):
        theoretical = np.array([sum(r["batch_size"] * r["sampler_outage_probability"] for r in g)
                                for g in windows])
        axes[0, 0].plot(x, theoretical/b, ":", color="black", label="Sampler expected share")
    axes[0, 0].set(ylabel="Outage fraction", ylim=(0, 1), title=f"Windows of {window} updates")
    axes[0, 0].legend()
    axes[0, 1].plot(x, np.cumsum(actual), label="Observed outage draws")
    axes[0, 1].plot(x, np.cumsum(expected), "--", label="Expected from time-matched buffer shares")
    axes[0, 1].set(ylabel="Cumulative draws", title="Allocation tracks a changing buffer")
    axes[0, 1].legend(fontsize=8)
    for c, color in [(0, "#3b82f6"), (1, "#d97706")]:
        numerator = np.array([sum(r["groups"][str(c)]["weighted_huber_numerator"] for r in g) for g in windows])
        count = np.array([sum(r["groups"][str(c)]["sample_count"] for r in g) for g in windows])
        axes[1, 0].plot(x, numerator/b, label=f"c={c}", color=color)
        axes[1, 1].plot(x, np.divide(numerator, count, out=np.full(len(count), np.nan), where=count>0),
                        label=f"c={c}", color=color)
    axes[1, 0].set(ylabel="Sum(weight * Huber) / batch draws", title="Group contribution to total batch loss")
    axes[1, 1].set(ylabel="Sum(weight * Huber) / group draws", title="Within-group mean weighted loss")
    axes[1, 0].legend()
    axes[1, 1].legend()
    details = Path(str(source).replace(".batches.jsonl", ".samples.jsonl"))
    if details.exists():
        points = reservoir(details)
        for c, color in [(0, "#3b82f6"), (1, "#d97706")]:
            group = [p for p in points if p["tag"] == c]
            axes[2, 0].scatter([p["replay_count_after_batch"] for p in group], [p["td_abs"] for p in group],
                               s=5, alpha=.2, label=f"c={c}", color=color)
        axes[2, 0].set(xlabel="Lifetime replay count after this batch", ylabel="Absolute sampled TD error",
                       yscale="symlog", title="Draw-weighted reservoir; repeats are retained")
        axes[2, 0].legend()
    else:
        axes[2, 0].text(.1, .5, "Per-draw log unavailable", transform=axes[2, 0].transAxes)
    axes[2, 1].plot(x, [1000*sum(r["sampler_seconds"] for r in g)/len(g) for g in windows])
    axes[2, 1].set(ylabel="Sampler ms per batch", title="Excludes logger, learner and environment")
    for ax in axes.flat:
        ax.grid(alpha=.15)
    for ax in [*axes[0], *axes[1], axes[2, 1]]:
        ax.set_xlabel("Collected decisions (includes warm-up)")
    title = "SYNTHETIC - no ViZDoom experiment" if synthetic else "Replay diagnostics"
    fig.suptitle(f"{title}\n{run_id} | {method}", fontweight="bold")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{run_id}.png", dpi=145)
    plt.close(fig)
    denominator = sum(r["batch_size"] for r in rows)
    total_actual, total_expected = float(actual.sum()), float(expected.sum())
    report = {"run_id": run_id, "method": method, "synthetic": synthetic,
              "draws": denominator, "outage_draws": total_actual,
              "expected_outage_draws_from_time_matched_occupancy": total_expected,
              "allocation_difference": (total_actual-total_expected)/denominator,
              "allocation_ratio": total_actual/total_expected if total_expected else None,
              "note": "Comparison uses sum(batch_size * buffer_share_at_sampling), not final occupancy"}
    (output / f"{run_id}.allocation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, help="Directory containing *.batches.jsonl")
    parser.add_argument("--frozen-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("window must be positive")
    if not args.logs and not args.frozen_analysis:
        parser.error("provide --logs and/or --frozen-analysis")
    if args.frozen_analysis:
        plot_frozen(args.frozen_analysis, args.output)
    if args.logs:
        paths = sorted(args.logs.glob("*.batches.jsonl"))
        if not paths:
            parser.error("no batch logs found")
        for path in paths:
            plot_run(path, args.output, args.window)
    print(f"Figures saved to {args.output}")


if __name__ == "__main__":
    main()
