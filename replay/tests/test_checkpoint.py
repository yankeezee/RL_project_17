from copy import deepcopy

import numpy as np
import pytest

from outage_replay import ReplayBuffer
from outage_replay.buffer import METHODS


def add(buffer, k):
    return buffer.add({"k": k, "observation": np.array([k, 1.])},
                      current_delivered=k % 3 != 0, next_delivered=True, terminated=False)


def compare(a, b):
    for key, values in a.snapshot().items():
        np.testing.assert_array_equal(values, b.snapshot()[key])
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert a._position == b._position
    assert a._next_id == b._next_id


@pytest.mark.parametrize("insertions", [0, 1, 4, 7, 25])
def test_checkpoint_continues_same_draws_updates_and_fifo(insertions):
    original = ReplayBuffer(7, seed=547)
    for k in range(insertions):
        add(original, k)
        batch = original.sample(9, method=METHODS[k % 4])
        original.update_priorities(batch.indices, np.linspace(-2, 3, 9), transition_ids=batch.transition_ids)
    restored = ReplayBuffer.from_state_dict(original.state_dict())
    compare(original, restored)
    for k in range(insertions, insertions+20):
        assert add(original, k) == add(restored, k)
        for method in METHODS:
            a, b = original.sample(19, method=method, beta=.7), restored.sample(19, method=method, beta=.7)
            for key in ("indices", "transition_ids", "tags", "probabilities", "weights", "replay_counts"):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            for ta, tb in zip(a.transitions, b.transitions):
                assert ta["k"] == tb["k"]
                np.testing.assert_array_equal(ta["observation"], tb["observation"])
            for buffer, batch in [(original, a), (restored, b)]:
                buffer.update_priorities(batch.indices, np.linspace(-7, 4, 19), transition_ids=batch.transition_ids)
        compare(original, restored)
    restored.check_invariants()


@pytest.mark.parametrize("corruption", ["raw", "td", "tag", "count", "ids", "position", "rng", "version", "payloads"])
def test_reject_corrupt_state_without_mutating_existing_buffer(corruption):
    buffer = ReplayBuffer(4, seed=123)
    for k in range(7):
        add(buffer, k)
    before = buffer.state_dict()
    corrupt = deepcopy(before)
    edits = {
        "raw": lambda: corrupt["snapshot"]["raw_priority"].fill(np.nan),
        "td": lambda: corrupt["snapshot"]["td_abs"].fill(99),
        "tag": lambda: corrupt["snapshot"]["tag"].fill(2),
        "count": lambda: corrupt["snapshot"]["replay_count"].fill(-1),
        "ids": lambda: corrupt["snapshot"]["transition_id"].fill(3),
        "position": lambda: corrupt.update(position=0),
        "rng": lambda: corrupt.update(rng_state={}),
        "version": lambda: corrupt.update(schema_version=99),
        "payloads": lambda: corrupt.update(payloads=[]),
    }
    edits[corruption]()
    with pytest.raises((ValueError, TypeError, KeyError)):
        buffer.load_state_dict(corrupt)
    compare(buffer, ReplayBuffer.from_state_dict(before))


def test_checkpoint_payloads_and_rng_have_no_shared_mutable_storage():
    buffer = ReplayBuffer(4)
    add(buffer, 1)
    state = buffer.state_dict()
    restored = ReplayBuffer.from_state_dict(state)
    state["payloads"][0]["observation"][0] = -999
    state["snapshot"]["raw_priority"][0] = 90
    state["rng_state"]["state"]["state"] = 7
    for replay in [buffer, restored]:
        assert replay.sample(1).transitions[0]["observation"][0] == 1
        assert replay.snapshot()["raw_priority"][0] == 1
    replacement = ReplayBuffer(1)
    replacement.load_state_dict(buffer.state_dict())
    compare(buffer, replacement)
