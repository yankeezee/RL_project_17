import itertools

import numpy as np
import pytest

from outage_replay import ReplayBuffer, beta_schedule, outage_tag
from outage_replay.buffer import ALPHA, EPSILON, METHODS
from outage_replay.frozen import probability_vectors


def make_buffer(tags=(0, 0, 0, 1, 1), errors=(.05, 1, 2, 10, 30), capacity=None, seed=317):
    buffer = ReplayBuffer(capacity or len(tags), seed=seed)
    slots, ids = [], []
    for k, tag in enumerate(tags):
        slot, identity = buffer.add(k, current_delivered=not bool(tag), next_delivered=True,
                                    terminated=False)
        slots.append(slot)
        ids.append(identity)
    buffer.update_priorities(slots, errors, transition_ids=ids)
    return buffer


@pytest.mark.parametrize("method", METHODS)
def test_independent_formula_probabilities_and_minimum(method):
    buffer = make_buffer(capacity=50)
    # Formula independent of the tree implementation, current N != capacity.
    u = (np.array([.05, 1, 2, 10, 30]) + EPSILON) ** ALPHA
    expected = {"uniform": np.full(5, .2), "per": u / u.sum(),
                "mixture": .1 + .5 * u / u.sum(),
                "grouped": np.r_[.6 * u[:3] / u[:3].sum(), .4 * u[3:] / u[3:].sum()]}[method]
    p = buffer.probabilities(method)
    np.testing.assert_allclose(p, expected, rtol=1e-13)
    assert (p > 0).all()
    assert p.sum() == pytest.approx(1)
    assert buffer.minimum_probability(method) == pytest.approx(p.min())
    np.testing.assert_allclose(probability_vectors(np.array([.05, 1, 2, 10, 30]) + EPSILON,
                                                  np.array([0, 0, 0, 1, 1]))[method], p)


def test_group_mass_is_occupancy_not_fixed_quota():
    buffer = make_buffer(tags=[0]*8+[1]*2, errors=[1]*8+[200, 500], capacity=50)
    p = buffer.probabilities()
    assert p[:8].sum() == pytest.approx(.8)
    assert p[8:].sum() == pytest.approx(.2)
    assert buffer.probabilities("per")[8:].sum() > .8


@pytest.mark.parametrize("tag", [0, 1])
def test_one_group_equals_per_empty_other_never_selected(tag):
    buffer = make_buffer(tags=[tag]*5)
    np.testing.assert_allclose(buffer.probabilities(), buffer.probabilities("per"))
    assert np.all(buffer.sample(2048).tags == tag)
    buffer.check_invariants()


def test_equal_priorities_become_uniform():
    buffer = make_buffer(errors=[3]*5)
    np.testing.assert_allclose(buffer.probabilities(), np.full(5, .2))


@pytest.mark.parametrize("current,next_,terminal,timeout", itertools.product([False, True], repeat=4))
def test_outage_tag_truth_table(current, next_, terminal, timeout):
    expected = int(not current or (not (terminal or timeout) and not next_))
    assert outage_tag(current_delivered=current, next_delivered=next_,
                      terminated=terminal, timeout=timeout) == expected


def test_terminal_does_not_read_next_observation_and_detection_is_not_an_input():
    assert outage_tag(current_delivered=True, next_delivered=None, terminated=True) == 0
    assert outage_tag(current_delivered=True, next_delivered=None, terminated=False, timeout=True) == 0
    with pytest.raises(TypeError):
        outage_tag(current_delivered=True, next_delivered=None, terminated=False)
    with pytest.raises(TypeError):
        outage_tag(current_delivered=1, next_delivered=True, terminated=False)


@pytest.mark.parametrize("method,beta", itertools.product(METHODS, [0, .4, 1]))
def test_actual_probability_and_global_importance_weights(method, beta):
    buffer = make_buffer(capacity=90)
    p = buffer.probabilities(method)
    batch = buffer.sample(64, method=method, beta=beta)
    raw = (len(buffer) * p) ** -beta
    np.testing.assert_allclose(batch.probabilities, p[batch.indices])
    np.testing.assert_allclose(batch.weights, (raw / raw.max())[batch.indices])
    assert np.all(batch.weights <= 1)


@pytest.mark.parametrize("method", ["per", "mixture", "grouped"])
def test_normalization_uses_transition_absent_from_batch(method):
    buffer = make_buffer()
    p = buffer.probabilities(method)
    minimum_slot = p.argmin()
    for _ in range(100):
        batch = buffer.sample(1, method=method, beta=1)
        if batch.indices[0] != minimum_slot:
            break
    assert batch.indices[0] != minimum_slot
    assert batch.weights[0] == pytest.approx(p.min() / p[batch.indices[0]])
    assert batch.weights[0] < 1  # minibatch-max normalization would wrongly give 1


def test_grouped_minimum_is_not_necessarily_global_minimum_mass():
    buffer = make_buffer(tags=[0, 0, 1, 1], errors=[1, 1000, .01, .01])
    p = buffer.probabilities("grouped")
    assert p.argmin() == 0  # smallest u is instead in group 1
    assert buffer.minimum_probability("grouped") == pytest.approx(p[0])
    batch = buffer.sample(100, beta=1)
    np.testing.assert_allclose(batch.weights, p[0]/p[batch.indices])


def test_zero_td_errors_still_have_positive_mass_and_current_max_can_reach_epsilon():
    buffer = make_buffer(errors=[0]*5)
    np.testing.assert_allclose(buffer.snapshot()["raw_priority"], EPSILON)
    buffer.add(None, current_delivered=True, next_delivered=True, terminated=False)
    assert buffer.snapshot()["raw_priority"][0] == EPSILON
    for method in METHODS:
        assert np.all(buffer.probabilities(method) > 0)
    buffer.check_invariants()


def test_overwrite_resets_lifetime_replay_count_without_affecting_other_slots():
    buffer = make_buffer(tags=[0, 1], errors=[1, 1])
    buffer.sample(100)
    old_count = buffer.snapshot()["replay_count"][1]
    buffer.add(None, current_delivered=True, next_delivered=True, terminated=False)
    assert buffer.snapshot()["replay_count"][0] == 0
    assert buffer.snapshot()["replay_count"][1] == old_count


def test_fifo_max_decrease_group_disappearance_and_reappearance():
    buffer = make_buffer(tags=[0, 1, 1], errors=[20, 2, 3], capacity=3)
    buffer.update_priorities([0], [1], transition_ids=[0])
    # Historical maximum 20 is no longer current; insertion gets current 3+eps.
    slot, identity = buffer.add("new", current_delivered=True, next_delivered=True, terminated=False)
    assert (slot, identity) == (0, 3)
    assert buffer.snapshot()["raw_priority"][slot] == pytest.approx(3 + EPSILON)
    for expected_sizes in [(2, 1), (3, 0)]:
        buffer.add("clean", current_delivered=True, next_delivered=True, terminated=False)
        assert buffer.group_sizes == expected_sizes
        buffer.check_invariants()
    buffer.add("outage", current_delivered=False, next_delivered=True, terminated=False)
    assert buffer.group_sizes == (2, 1)
    assert sorted(buffer.snapshot()["transition_id"]) == [4, 5, 6]
    buffer.check_invariants()


def test_new_priority_is_current_max_before_victim_eviction():
    buffer = make_buffer(tags=[0, 1], errors=[10, 1])
    buffer.add(None, current_delivered=False, next_delivered=None, terminated=True)
    np.testing.assert_allclose(buffer.snapshot()["raw_priority"], [10+EPSILON, 1+EPSILON])


def test_duplicate_updates_order_independent_counts_include_every_draw():
    a, b = make_buffer(), make_buffer()
    a.update_priorities([3, 3, 1], [-7, 2, 9], transition_ids=[3, 3, 1])
    b.update_priorities([1, 3, 3], [9, 2, -7], transition_ids=[1, 3, 3])
    np.testing.assert_array_equal(a.snapshot()["raw_priority"], b.snapshot()["raw_priority"])
    assert a.snapshot()["raw_priority"][3] == pytest.approx(7 + EPSILON)
    batch = a.sample(100)
    counts = np.bincount(batch.indices, minlength=len(a))
    np.testing.assert_array_equal(a.snapshot()["replay_count"], counts)
    np.testing.assert_array_equal(batch.replay_counts, counts[batch.indices])


def test_stale_and_nonfinite_updates_are_atomic():
    buffer = make_buffer(tags=[0, 1], errors=[1, 2])
    batch = buffer.sample(64)
    buffer.add("replacement", current_delivered=True, next_delivered=True, terminated=False)
    before = buffer.snapshot()["raw_priority"]
    with pytest.raises(ValueError, match="stale"):
        buffer.update_priorities(batch.indices, np.ones(64), transition_ids=batch.transition_ids)
    with pytest.raises(ValueError, match="finite"):
        buffer.update_priorities([0, 1], [10, np.nan], transition_ids=[2, 1])
    np.testing.assert_array_equal(before, buffer.snapshot()["raw_priority"])


def test_randomized_ring_against_independent_reference():
    rng, buffer = np.random.default_rng(91), ReplayBuffer(13, seed=17)
    for step in range(180):
        buffer.add(step, current_delivered=bool(rng.integers(2)),
                   next_delivered=bool(rng.integers(2)), terminated=False)
        batch = buffer.sample(17, method=METHODS[step % 4], beta=.7)
        buffer.update_priorities(batch.indices, rng.normal(size=17), transition_ids=batch.transition_ids)
        snapshot = buffer.snapshot()
        for method, p in probability_vectors(snapshot["raw_priority"], snapshot["tag"]).items():
            np.testing.assert_allclose(buffer.probabilities(method), p, rtol=2e-13)
        buffer.check_invariants()


@pytest.mark.parametrize("method", METHODS)
def test_empirical_frequencies_with_explicit_statistical_tolerance(method):
    buffer = make_buffer(seed=1037)
    n = 200_000
    batch = buffer.sample(n, method=method)
    p = buffer.probabilities(method)
    empirical = np.bincount(batch.indices, minlength=len(buffer)) / n
    # Per-category Bernstein bound, union over 5 cells at family-wise alpha=1e-6.
    # |freq-p| <= sqrt(2*p*(1-p)*log(2*k/alpha)/n) + 2*log(2*k/alpha)/(3*n).
    t = np.log(2 * len(buffer) / 1e-6)
    tolerance = np.sqrt(2 * p * (1 - p) * t / n) + 2 * t / (3 * n)
    assert np.all(np.abs(empirical - p) <= tolerance), (empirical, p, tolerance)


def test_empty_capacity_one_and_validation():
    buffer = ReplayBuffer(1)
    with pytest.raises(ValueError, match="empty"):
        buffer.sample()
    buffer.check_invariants()
    buffer.add(None, current_delivered=True, next_delivered=True, terminated=False)
    assert buffer.snapshot()["raw_priority"][0] == 1.0
    for method in METHODS:
        batch = buffer.sample(8, method=method)
        np.testing.assert_array_equal(batch.probabilities, np.ones(8))
        np.testing.assert_array_equal(batch.weights, np.ones(8))
    for kwargs in [{"batch_size": 0}, {"beta": -1}, {"beta": np.nan}, {"method": "bad"}]:
        with pytest.raises(ValueError):
            buffer.sample(**kwargs)
    with pytest.raises(ValueError):
        ReplayBuffer(0)


def test_beta_uses_total_schedule_including_warmup():
    assert beta_schedule(0) == .4
    assert beta_schedule(5000) == pytest.approx(.41)
    assert beta_schedule(150000) == pytest.approx(.7)
    assert beta_schedule(300000) == 1
    assert beta_schedule(400000) == 1
    with pytest.raises(ValueError):
        beta_schedule(-1)
