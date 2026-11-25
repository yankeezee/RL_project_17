import numpy as np
import pytest

from outage_replay import ReplayBuffer
from outage_replay.buffer import METHODS
from outage_replay.diagnostics import batch_diagnostics, huber


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("tags", [[0]*7, [1]*7, [0]*5+[1]*2])
def test_exact_group_marginals_match_item_probability_sums(method, tags):
    buffer = ReplayBuffer(10)
    for c in tags:
        buffer.add(None, current_delivered=not bool(c), next_delivered=True, terminated=False)
    buffer.update_priorities(np.arange(7), [1, 2, 3, 4, 5, 100, 300], transition_ids=np.arange(7))
    p = buffer.probabilities(method)
    expected = [p[np.array(tags) == c].sum() for c in (0, 1)]
    np.testing.assert_allclose(buffer.group_probabilities(method), expected, rtol=1e-14)
    batch = buffer.sample(method=method)
    np.testing.assert_allclose(batch.group_probabilities, expected, rtol=1e-14)
    # Logging must refer to pre-update/pre-overwrite sampler state.
    for _ in range(20):
        buffer.add(None, current_delivered=True, next_delivered=True, terminated=False)
    row = batch_diagnostics(batch, np.ones(64), huber(np.ones(64)))
    assert row["sampler_outage_probability"] == pytest.approx(expected[1])
    for c in (0, 1):
        assert row["groups"][str(c)]["expected_count_from_sampler"] == pytest.approx(64*expected[c])
    # Original research diagnostic retains its original definition.
    assert row["allocation_difference"] == pytest.approx(batch.tags.mean()-np.mean(tags))


def test_per_expected_excess_does_not_depend_on_sample_noise():
    buffer = ReplayBuffer(10, seed=0)
    for k in range(10):
        buffer.add(None, current_delivered=k < 8, next_delivered=True, terminated=False)
    buffer.update_priorities(np.arange(10), [1]*8+[100]*2, transition_ids=np.arange(10))
    expected = buffer.group_probabilities("per")[1]
    assert expected > .7
    rows = [batch_diagnostics(buffer.sample(1, method="per"), [1], [.5]) for _ in range(50)]
    assert len({r["sample_outage_fraction"] for r in rows}) == 2
    assert {r["sampler_outage_probability"] for r in rows} == {expected}
