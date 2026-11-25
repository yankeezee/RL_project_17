"""Append-only diagnostics with sample-time occupancy and explicit loss denominators."""

import json
from pathlib import Path

import numpy as np

from .buffer import ReplayBatch


def huber(td_errors):
    """Unit-threshold Huber, matching smooth_l1_loss(..., beta=1)."""
    a = np.abs(np.asarray(td_errors, dtype=np.float64))
    return np.where(a < 1.0, 0.5 * a * a, a - 0.5)


def batch_diagnostics(batch: ReplayBatch, td_errors, unreduced_huber):
    """Use the exact pre-step TD residuals and Huber losses used by the optimizer."""
    td = np.asarray(td_errors, dtype=np.float64)
    losses = np.asarray(unreduced_huber, dtype=np.float64)
    b = len(batch.indices)
    if td.shape != (b,) or losses.shape != (b,):
        raise ValueError("TD errors and unreduced loss must have shape (batch_size,)")
    if not np.all(np.isfinite(td)) or not np.all(np.isfinite(losses)) or np.any(losses < 0):
        raise ValueError("non-finite TD/loss or negative Huber loss")
    np.testing.assert_allclose(losses, huber(td), rtol=2e-5, atol=1e-7,
                               err_msg="Expected unit-threshold unreduced Huber loss")
    weighted = batch.weights * losses
    fraction = batch.group_sizes[1] / batch.buffer_size
    sampled = float(np.mean(batch.tags))
    groups = {}
    for c in (0, 1):
        mask = batch.tags == c
        count = int(mask.sum())
        numerator = float(weighted[mask].sum())
        groups[str(c)] = {
            "sample_count": count,
            "expected_count_from_buffer_share": b * batch.group_sizes[c] / batch.buffer_size,
            "sampler_group_probability": batch.group_probabilities[c],
            "expected_count_from_sampler": b * batch.group_probabilities[c],
            "td_abs_sum": float(np.abs(td[mask]).sum()),
            "td_abs_sq_sum": float((td[mask] ** 2).sum()),
            "td_abs_mean": float(np.abs(td[mask]).mean()) if count else None,
            "replay_count_sum": int(batch.replay_counts[mask].sum()),
            "weighted_huber_numerator": numerator,
            "within_group_denominator": count,
            "within_group_weighted_huber_mean": numerator / count if count else None,
            "batch_denominator": b,
            "contribution_to_batch_mean": numerator / b,
        }
    return {
        "schema_version": 1, "method": batch.method, "beta": batch.beta,
        "buffer_size": batch.buffer_size, "group_sizes": list(batch.group_sizes),
        "batch_size": b, "buffer_outage_fraction": fraction,
        "sample_outage_fraction": sampled, "allocation_difference": sampled - fraction,
        "allocation_ratio": sampled / fraction if fraction else None,
        "sampler_outage_probability": batch.group_probabilities[1],
        "sampler_expected_allocation_difference": batch.group_probabilities[1] - fraction,
        "weighted_huber_numerator": float(weighted.sum()), "loss_denominator": b,
        "weighted_huber_batch_mean": float(weighted.mean()),
        "minimum_probability": batch.minimum_probability,
        "sampler_seconds": batch.sample_seconds, "groups": groups,
    }


class ReplayLogger:
    """One batch JSONL + optional per-draw JSONL, including repeated indices.

    Use one unique file prefix per run. Existing files are rejected to prevent
    accidental mixing of resumed/independent runs; use a fresh segment on resume.
    """

    def __init__(self, directory, *, run_id: str, seed: int, regime: str,
                 synthetic: bool, sample_details: bool = True, metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not run_id or Path(run_id).name != run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a nonempty filename component")
        self.context = {"run_id": run_id, "seed": int(seed), "regime": regime,
                        "synthetic": bool(synthetic)}
        self._closed = False
        self._batches = (self.directory / f"{run_id}.batches.jsonl").open("x", encoding="utf-8")
        self._samples = None
        try:
            if sample_details:
                self._samples = (self.directory / f"{run_id}.samples.jsonl").open("x", encoding="utf-8")
            (self.directory / f"{run_id}.metadata.json").write_text(
                json.dumps({**self.context, "schema_version": 1, "metadata": metadata or {}},
                           indent=2, allow_nan=False), encoding="utf-8")
        except Exception:
            self.close()
            raise

    def log(self, batch: ReplayBatch, *, td_errors, unreduced_huber,
            decisions: int, optimizer_step: int):
        if self._closed:
            raise ValueError("logger is closed")
        row = {**self.context, "decisions": int(decisions), "optimizer_step": int(optimizer_step),
               **batch_diagnostics(batch, td_errors, unreduced_huber)}
        self._batches.write(json.dumps(row, allow_nan=False) + "\n")
        if self._samples:
            for k in range(len(batch.indices)):
                item = {**self.context, "method": batch.method, "decisions": int(decisions),
                        "optimizer_step": int(optimizer_step), "draw": k,
                        "transition_id": int(batch.transition_ids[k]), "slot": int(batch.indices[k]),
                        "tag": int(batch.tags[k]), "probability": float(batch.probabilities[k]),
                        "weight": float(batch.weights[k]), "td_abs": float(abs(td_errors[k])),
                        "huber": float(unreduced_huber[k]),
                        "weighted_huber": float(batch.weights[k] * unreduced_huber[k]),
                        "replay_count_after_batch": int(batch.replay_counts[k])}
                self._samples.write(json.dumps(item, allow_nan=False) + "\n")
        # Durability at each update; benchmark logging separately from the sampler.
        self._batches.flush()
        if self._samples:
            self._samples.flush()
        return row

    def close(self):
        self._batches.close()
        if self._samples:
            self._samples.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
