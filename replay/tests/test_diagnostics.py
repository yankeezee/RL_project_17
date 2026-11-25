import csv
import json

import numpy as np
import pytest

from outage_replay import ReplayBuffer
from outage_replay.diagnostics import ReplayLogger, batch_diagnostics, huber
from outage_replay.frozen import analyze_frozen, export_frozen


def test_loss_group_numerators_and_denominators_and_dynamic_occupancy(tmp_path):
    buffer = ReplayBuffer(4, seed=9)
    for c in [0, 0, 0, 1]:
        buffer.add(None, current_delivered=not bool(c), next_delivered=True, terminated=False)
    batch = buffer.sample(64, beta=.7)
    # Mutate buffer after sampling. Diagnostics must retain its old 1/4 baseline.
    buffer.add(None, current_delivered=False, next_delivered=True, terminated=False)
    td = np.linspace(-2, 3, 64)
    loss = huber(td)
    with ReplayLogger(tmp_path, run_id="test", seed=9, regime="synthetic", synthetic=True) as logger:
        row = logger.log(batch, td_errors=td, unreduced_huber=loss, decisions=8, optimizer_step=1)
    assert row["buffer_outage_fraction"] == .25
    assert buffer.group_sizes == (2, 2)
    assert sum(g["contribution_to_batch_mean"] for g in row["groups"].values()) == pytest.approx(
        np.mean(batch.weights * loss))
    for c, group in row["groups"].items():
        mask = batch.tags == int(c)
        numerator = np.sum(batch.weights[mask] * loss[mask])
        assert group["weighted_huber_numerator"] == pytest.approx(numerator)
        assert group["within_group_denominator"] == int(mask.sum())
        assert group["batch_denominator"] == 64
    saved = json.loads((tmp_path / "test.batches.jsonl").read_text())
    assert saved == row
    assert len((tmp_path / "test.samples.jsonl").read_text().splitlines()) == 64
    with pytest.raises(FileExistsError):
        ReplayLogger(tmp_path, run_id="test", seed=9, regime="synthetic", synthetic=True)


def test_no_outages_ratio_and_empty_group_mean_are_null():
    buffer = ReplayBuffer(1)
    buffer.add(None, current_delivered=True, next_delivered=None, terminated=True)
    row = batch_diagnostics(buffer.sample(4), [1]*4, [.5]*4)
    assert row["allocation_ratio"] is None
    assert row["groups"]["1"]["within_group_weighted_huber_mean"] is None
    assert row["groups"]["1"]["contribution_to_batch_mean"] == 0


def test_frozen_identical_content_methods_and_hash_validation(tmp_path):
    buffer = ReplayBuffer(10)
    for k in range(10):
        buffer.add(k, current_delivered=k < 8, next_delivered=True, terminated=False)
    buffer.update_priorities(np.arange(10), [1]*8+[100]*2, transition_ids=np.arange(10))
    original = buffer.snapshot()
    source = export_frozen(buffer, tmp_path / "source", synthetic=True)
    summary = analyze_frozen(source, tmp_path / "analysis")
    assert summary["synthetic"] is True
    assert summary["methods"]["grouped"]["expected_outage_mass"] == pytest.approx(.2)
    assert summary["methods"]["per"]["expected_outage_mass"] > .7
    for key, value in original.items():
        np.testing.assert_array_equal(buffer.snapshot()[key], value)
    source.write_text(source.read_text()+"\n")
    with pytest.raises(ValueError, match="hash"):
        analyze_frozen(source, tmp_path / "bad")


def test_wrong_reduction_or_loss_is_rejected():
    buffer = ReplayBuffer(1)
    buffer.add(None, current_delivered=True, next_delivered=True, terminated=False)
    batch = buffer.sample(4)
    with pytest.raises(ValueError):
        batch_diagnostics(batch, [1]*4, .5)
    with pytest.raises(AssertionError):
        batch_diagnostics(batch, [2]*4, [4]*4)


def test_batch_only_logging_and_unmeasured_frozen_td(tmp_path):
    buffer = ReplayBuffer(1)
    buffer.add(None, current_delivered=True, next_delivered=None, terminated=True)
    with ReplayLogger(tmp_path, run_id="no_details", seed=0, regime="clean",
                      synthetic=True, sample_details=False) as logger:
        logger.log(buffer.sample(2), td_errors=[0, 0], unreduced_huber=[0, 0],
                   decisions=5000, optimizer_step=1)
    assert not (tmp_path / "no_details.samples.jsonl").exists()
    source = export_frozen(buffer, tmp_path / "source", synthetic=True)
    with source.open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["td_abs"] == ""  # no refresh, so no measured TD error
    assert int(row["replay_count"]) == 2  # both draws are retained
    summary = analyze_frozen(source, tmp_path / "analysis")
    assert summary["methods"]["per"]["expected_outage_mass"] == 0
