import numpy as np
import pytest

from outage_replay import ReplayBuffer
from scripts.benchmark_refresh import legacy_refresh


@pytest.mark.parametrize("capacity", [1, 3, 17, 256])
def test_batched_refresh_bitwise_identical_to_legacy_under_ring_and_duplicates(capacity):
    a, b = ReplayBuffer(capacity, seed=6), ReplayBuffer(capacity, seed=6)
    rng = np.random.default_rng(71)
    for k in range(120):
        for buffer in (a, b):
            buffer.add(None, current_delivered=k % 3 != 0, next_delivered=True, terminated=False)
        indices = rng.integers(len(a), size=64)
        td = rng.normal(size=64) * 10 ** rng.uniform(-5, 8)
        ids = a._ids[indices]
        legacy_refresh(a, indices, td, transition_ids=ids)
        b.update_priorities(indices, td, transition_ids=ids)
        for key in ("_sum", "_min", "_max", "_raw", "_td_abs"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    a.check_invariants()
    b.check_invariants()
