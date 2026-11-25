"""Common-buffer analysis, independent of each method's collected experience."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .buffer import ALPHA, EPSILON, METHODS


def probability_vectors(raw_priority, tags):
    raw, tags = np.asarray(raw_priority, dtype=float), np.asarray(tags)
    if raw.ndim != 1 or tags.shape != raw.shape or not len(raw):
        raise ValueError("nonempty matching one-dimensional priorities/tags required")
    if not np.all(np.isfinite(raw)) or np.any(raw < EPSILON) or not np.all(np.isin(tags, [0, 1])):
        raise ValueError("invalid raw priorities or binary tags")
    u, n = raw ** ALPHA, len(raw)
    per = u / u.sum()
    grouped = np.zeros(n)
    for c in (0, 1):
        group = tags == c
        if group.any():
            grouped[group] = group.mean() * u[group] / u[group].sum()
    return {"uniform": np.full(n, 1 / n), "per": per,
            "mixture": 0.5 / n + 0.5 * per, "grouped": grouped}


def export_frozen(buffer, directory, *, synthetic: bool, metadata=None):
    """Store analysis metadata only; this is NOT a resumable learner checkpoint."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    snapshot = buffer.snapshot()
    if not len(snapshot["slot"]):
        raise ValueError("cannot export empty buffer")
    target = path / "frozen.csv"
    names = list(snapshot)
    with target.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        for row in zip(*(snapshot[k] for k in names)):
            # Unrefreshed priority has no measured TD; leave it blank, never invent one.
            writer.writerow(["" if isinstance(v, (float, np.floating)) and np.isnan(v) else v for v in row])
    info = {"schema_version": 1, "synthetic": bool(synthetic), "alpha": ALPHA,
            "epsilon": EPSILON, "buffer_size": len(buffer), "group_sizes": list(buffer.group_sizes),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "metadata": metadata or {}}
    (path / "frozen.metadata.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    return target


def analyze_frozen(source, output_directory, *, beta=1.0):
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be in [0, 1]")
    source, output = Path(source), Path(output_directory)
    info = json.loads(source.with_name("frozen.metadata.json").read_text(encoding="utf-8"))
    if info["alpha"] != ALPHA or info["epsilon"] != EPSILON:
        raise ValueError("frozen priority convention differs from this implementation")
    if hashlib.sha256(source.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("frozen CSV hash does not match metadata")
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    raw = np.array([float(r["raw_priority"]) for r in rows])
    tags = np.array([int(r["tag"]) for r in rows])
    probabilities = probability_vectors(raw, tags)
    minima = {method: p.min() for method, p in probabilities.items()}
    if len(rows) != info["buffer_size"] or list(np.bincount(tags, minlength=2)) != info["group_sizes"]:
        raise ValueError("frozen occupancy does not match metadata")
    output.mkdir(parents=True, exist_ok=True)
    fraction = float(tags.mean())
    summary = {"schema_version": 1, "synthetic": info["synthetic"],
               "source_sha256": info["sha256"], "buffer_size": len(raw),
               "buffer_outage_fraction": fraction, "beta": float(beta), "methods": {}}
    with (output / "probabilities.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["slot", "transition_id", "tag", "raw_priority", "td_abs", "replay_count"] +
                        [f"p_{m}" for m in METHODS] + [f"w_{m}" for m in METHODS])
        for k, row in enumerate(rows):
            writer.writerow([row[s] for s in ("slot", "transition_id", "tag", "raw_priority", "td_abs", "replay_count")] +
                            [probabilities[m][k] for m in METHODS] +
                            [(minima[m] / probabilities[m][k]) ** beta for m in METHODS])
    for method, p in probabilities.items():
        mass = float(p[tags == 1].sum())
        summary["methods"][method] = {
            "expected_outage_mass": mass, "allocation_difference": mass - fraction,
            "allocation_ratio": mass / fraction if fraction else None,
            "probability_min": float(minima[method]), "probability_max": float(p.max()),
            "effective_probability_support": float(1 / np.square(p).sum()),
            "probability_quantiles": np.quantile(p, [0, .25, .5, .75, 1]).tolist()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
