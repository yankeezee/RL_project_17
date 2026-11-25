"""FIFO replay with exact global weight normalization and two priority groups.

raw_priority = abs(td_error) + EPSILON; transformed mass u = raw_priority**ALPHA.
All random draws are independent, with replacement. Not thread-safe.
"""

from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

EPSILON = 1e-6
ALPHA = 0.6
METHODS = ("uniform", "per", "mixture", "grouped")


def _flag(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean injected delivery/task flag")
    return bool(value)


def outage_tag(*, current_delivered: bool, next_delivered: bool | None,
               terminated: bool, timeout: bool = False) -> int:
    """A timeout ends the finite task; its next observation is ignored."""
    current = _flag(current_delivered, "current_delivered")
    end = _flag(terminated, "terminated") | _flag(timeout, "timeout")
    if end:
        return int(not current)
    following = _flag(next_delivered, "next_delivered")
    return int(not current or not following)


def beta_schedule(decisions: int, total_decisions: int = 300_000) -> float:
    """Global interaction clock, including warm-up; do not restart per episode."""
    if total_decisions <= 0 or decisions < 0:
        raise ValueError("invalid training clock")
    return 0.4 + 0.6 * min(decisions / total_decisions, 1.0)


@dataclass(frozen=True)
class ReplayBatch:
    indices: np.ndarray
    transition_ids: np.ndarray
    transitions: tuple[Any, ...]
    tags: np.ndarray
    probabilities: np.ndarray
    weights: np.ndarray
    replay_counts: np.ndarray
    method: str
    beta: float
    buffer_size: int
    group_sizes: tuple[int, int]
    group_probabilities: tuple[float, float]
    minimum_probability: float
    sample_seconds: float


class ReplayBuffer:
    """O(log capacity) insertion/update and O(batch_size log capacity) sampling.

    Two sum/min trees support within-group sampling and global minimum P(i).
    A max tree tracks the *current* raw-priority maximum, allowing decreases.
    Payloads are opaque references owned by the caller, not copied internally.
    """

    def __init__(self, capacity: int = 50_000, *, seed: int = 0,
                 initial_raw_priority: float = 1.0):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if not np.isfinite(initial_raw_priority) or initial_raw_priority < EPSILON:
            raise ValueError("initial raw priority must be finite and >= epsilon")
        self.capacity = capacity
        self.initial_raw_priority = float(initial_raw_priority)
        self.rng = np.random.default_rng(seed)
        self._base = 1 << (capacity - 1).bit_length()
        self._sum = np.zeros((2, 2 * self._base), dtype=np.float64)
        self._min = np.full((2, 2 * self._base), np.inf, dtype=np.float64)
        self._max = np.zeros(2 * self._base, dtype=np.float64)
        self._raw = np.zeros(capacity, dtype=np.float64)
        self._tags = np.zeros(capacity, dtype=np.uint8)
        self._ids = np.zeros(capacity, dtype=np.int64)
        self._counts = np.zeros(capacity, dtype=np.int64)
        self._td_abs = np.full(capacity, np.nan)
        self._payloads = [None] * capacity
        self._group_sizes = np.zeros(2, dtype=np.int64)
        self._size = self._position = self._next_id = 0

    def __len__(self):
        return self._size

    @property
    def group_sizes(self):
        return tuple(int(x) for x in self._group_sizes)

    @property
    def metadata_bytes(self):
        """Exact ndarray storage; excludes payloads and Python/container overhead."""
        return sum(a.nbytes for a in (self._sum, self._min, self._max, self._raw,
                                     self._tags, self._ids, self._counts, self._td_abs,
                                     self._group_sizes))

    def _set_leaf(self, index, tag, raw):
        leaf = self._base + index
        u = raw ** ALPHA
        self._sum[:, leaf] = 0.0
        self._min[:, leaf] = np.inf
        self._sum[tag, leaf] = self._min[tag, leaf] = u
        self._max[leaf] = raw
        node = leaf // 2
        while node:
            left, right = 2 * node, 2 * node + 1
            self._sum[:, node] = self._sum[:, left] + self._sum[:, right]
            self._min[:, node] = np.minimum(self._min[:, left], self._min[:, right])
            self._max[node] = max(self._max[left], self._max[right])
            node //= 2

    def _set_leaves(self, indices, tags, raw):
        """Unique leaves, then each affected ancestor once at each tree level."""
        leaves = self._base + indices
        # Retain scalar exponent evaluation to match the original updater bitwise.
        u = np.fromiter((r ** ALPHA for r in raw), dtype=np.float64, count=len(raw))
        self._sum[:, leaves] = 0.0
        self._min[:, leaves] = np.inf
        self._sum[tags, leaves] = self._min[tags, leaves] = u
        self._max[leaves] = raw
        nodes = np.unique(leaves // 2)
        nodes = nodes[nodes > 0]
        while len(nodes):
            left, right = 2 * nodes, 2 * nodes + 1
            self._sum[:, nodes] = self._sum[:, left] + self._sum[:, right]
            self._min[:, nodes] = np.minimum(self._min[:, left], self._min[:, right])
            self._max[nodes] = np.maximum(self._max[left], self._max[right])
            nodes = np.unique(nodes // 2)
            nodes = nodes[nodes > 0]

    def add(self, transition: Any, *, current_delivered: bool,
            next_delivered: bool | None, terminated: bool,
            timeout: bool = False) -> tuple[int, int]:
        """Return (slot, lifetime ID). New raw priority is max just before eviction."""
        tag = outage_tag(current_delivered=current_delivered, next_delivered=next_delivered,
                         terminated=terminated, timeout=timeout)
        raw = float(self._max[1]) if self._size else self.initial_raw_priority
        slot = self._position
        if self._size == self.capacity:
            self._group_sizes[self._tags[slot]] -= 1
        else:
            self._size += 1
        self._group_sizes[tag] += 1
        self._raw[slot], self._tags[slot] = raw, tag
        self._ids[slot] = self._next_id
        self._counts[slot], self._td_abs[slot] = 0, np.nan
        self._payloads[slot] = transition
        self._set_leaf(slot, tag, raw)
        result = (slot, self._next_id)
        self._next_id += 1
        self._position = (slot + 1) % self.capacity
        return result

    def _require_nonempty(self, method):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not self._size:
            raise ValueError("cannot sample/analyze an empty buffer")

    def _probabilities_at(self, indices, method):
        u = self._sum[self._tags[indices], self._base + indices]
        if method == "uniform":
            return np.full(len(indices), 1.0 / self._size)
        per = u / self._sum[:, 1].sum()
        if method == "per":
            return per
        if method == "mixture":
            return 0.5 / self._size + 0.5 * per
        tags = self._tags[indices]
        return (self._group_sizes[tags] / self._size) * u / self._sum[tags, 1]

    def minimum_probability(self, method="grouped"):
        self._require_nonempty(method)
        active = self._group_sizes > 0
        if method == "uniform":
            return 1.0 / self._size
        minimum_per = self._min[active, 1].min() / self._sum[:, 1].sum()
        if method == "per":
            return float(minimum_per)
        if method == "mixture":
            return float(0.5 / self._size + 0.5 * minimum_per)
        return float(np.min(self._group_sizes[active] / self._size *
                            self._min[active, 1] / self._sum[active, 1]))

    def probabilities(self, method="grouped"):
        """Diagnostic O(N) vector in physical slot order, never used by sampling."""
        self._require_nonempty(method)
        return self._probabilities_at(np.arange(self._size), method)

    def group_probabilities(self, method="grouped"):
        """Exact sampler marginal probabilities of groups 0/1, in O(1) time."""
        self._require_nonempty(method)
        share = self._group_sizes / self._size
        if method in ("uniform", "grouped"):
            result = share
        else:
            per = self._sum[:, 1] / self._sum[:, 1].sum()
            result = per if method == "per" else 0.5 * share + 0.5 * per
        return tuple(float(p) for p in result)

    def _draw_tree(self, groups):
        # Vectorized descent. Empty groups never enter this method.
        nodes = np.ones(len(groups), dtype=np.int64)
        masses = self.rng.random(len(groups)) * self._sum[groups, 1]
        while np.any(nodes < self._base):
            left = 2 * nodes
            left_mass = self._sum[groups, left]
            right = masses >= left_mass
            masses = np.where(right, masses - left_mass, masses)
            nodes = left + right
        return nodes - self._base

    def sample(self, batch_size: int = 64, *, method="grouped", beta=0.4):
        self._require_nonempty(method)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch size must be a positive integer")
        if not np.isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        start = perf_counter()
        if method == "uniform":
            indices = self.rng.integers(self._size, size=batch_size)
        elif method == "grouped":
            # Select group by occupancy, then priority within the selected group.
            groups = (self.rng.random(batch_size) < self._group_sizes[1] / self._size).astype(int)
            indices = self._draw_tree(groups)
        else:
            totals = self._sum[:, 1]
            # Decomposing global PER by each group's mass is exactly global PER.
            groups = (self.rng.random(batch_size) < totals[1] / totals.sum()).astype(int)
            indices = self._draw_tree(groups)
            if method == "mixture":
                uniform_branch = self.rng.random(batch_size) < 0.5
                indices[uniform_branch] = self.rng.integers(self._size, size=uniform_branch.sum())
        probabilities = self._probabilities_at(indices, method)
        minimum = self.minimum_probability(method)
        # Algebraically equals (N*P)^(-beta) / (N*min(P))^(-beta).
        weights = (minimum / probabilities) ** beta
        np.add.at(self._counts, indices, 1)
        values = dict(indices=indices, transition_ids=self._ids[indices].copy(),
                      transitions=tuple(self._payloads[i] for i in indices),
                      tags=self._tags[indices].copy(), probabilities=probabilities,
                      weights=weights, replay_counts=self._counts[indices].copy(),
                      method=method, beta=float(beta), buffer_size=self._size,
                      group_sizes=self.group_sizes,
                      group_probabilities=self.group_probabilities(method), minimum_probability=minimum)
        return ReplayBatch(**values, sample_seconds=perf_counter() - start)

    def update_priorities(self, indices, td_errors, *, transition_ids):
        """Apply after optimizer.step using detached TD residuals from that loss.

        Duplicate IDs reduce by max(abs(td)); order-independent. Validate the
        entire request before mutation. Stale overwritten slots are rejected.
        """
        indices = np.asarray(indices)
        ids = np.asarray(transition_ids)
        errors = np.asarray(td_errors, dtype=np.float64)
        if indices.ndim != 1 or ids.shape != indices.shape or errors.shape != indices.shape:
            raise ValueError("indices, transition_ids and TD errors must be matching vectors")
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("indices and transition IDs must be integers")
        if np.any(indices < 0) or np.any(indices >= self._size):
            raise IndexError("slot is not currently occupied")
        if not np.array_equal(self._ids[indices], ids):
            raise ValueError("stale batch: a sampled transition has been overwritten")
        if not np.all(np.isfinite(errors)):
            raise ValueError("TD errors must be finite")
        unique, inverse = np.unique(indices, return_inverse=True)
        absolute = np.zeros(len(unique))
        np.maximum.at(absolute, inverse, np.abs(errors))
        raw = absolute + EPSILON
        self._raw[unique], self._td_abs[unique] = raw, absolute
        self._set_leaves(unique, self._tags[unique], raw)

    def snapshot(self):
        """Independent arrays; payloads deliberately excluded from analysis export."""
        n = self._size
        return {"slot": np.arange(n), "transition_id": self._ids[:n].copy(),
                "tag": self._tags[:n].copy(), "raw_priority": self._raw[:n].copy(),
                "td_abs": self._td_abs[:n].copy(), "replay_count": self._counts[:n].copy()}

    def state_dict(self):
        """Independent replay checkpoint including payloads and RNG; no learner state.

        Payloads must support deepcopy. Serialize only with a trusted checkpoint
        mechanism owned by the learner; no pickle loader is exposed here.
        """
        return {"schema_version": 1, "alpha": ALPHA, "epsilon": EPSILON,
                "capacity": self.capacity, "initial_raw_priority": self.initial_raw_priority,
                "position": self._position, "next_id": self._next_id,
                "snapshot": self.snapshot(), "payloads": deepcopy(self._payloads[:self._size]),
                "rng_state": deepcopy(self.rng.bit_generator.state)}

    @classmethod
    def from_state_dict(cls, state):
        """Validate before constructing live state; exact replay within the same RNG/runtime."""
        def integer(value, name):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            return int(value)

        if state["schema_version"] != 1 or state["alpha"] != ALPHA or state["epsilon"] != EPSILON:
            raise ValueError("incompatible replay state schema or priority convention")
        capacity = integer(state["capacity"], "capacity")
        restored = cls(capacity, initial_raw_priority=state["initial_raw_priority"])
        snap = state["snapshot"]
        arrays = {key: np.asarray(snap[key]) for key in
                  ("slot", "transition_id", "tag", "raw_priority", "td_abs", "replay_count")}
        n = len(arrays["slot"])
        if not 0 <= n <= capacity or any(a.shape != (n,) for a in arrays.values()):
            raise ValueError("invalid replay array dimensions")
        for key in ("slot", "transition_id", "tag", "replay_count"):
            if not np.issubdtype(arrays[key].dtype, np.integer):
                raise ValueError(f"{key} must contain integers")
        raw = arrays["raw_priority"]
        td = arrays["td_abs"]
        if not np.all(np.isfinite(raw)) or np.any(raw < EPSILON):
            raise ValueError("invalid raw priorities")
        if not np.all(np.isnan(td) | (np.isfinite(td) & (td >= 0))):
            raise ValueError("invalid measured TD errors")
        measured = ~np.isnan(td)
        if not np.array_equal(raw[measured], td[measured] + EPSILON):
            raise ValueError("measured TD errors disagree with raw priorities")
        if not np.all(np.isin(arrays["tag"], [0, 1])) or np.any(arrays["replay_count"] < 0):
            raise ValueError("invalid tags or replay counts")
        position = integer(state["position"], "position")
        next_id = integer(state["next_id"], "next_id")
        if not n <= next_id <= np.iinfo(np.int64).max or position != next_id % capacity:
            raise ValueError("invalid FIFO cursor or next transition ID")
        if n < capacity and next_id != n:
            raise ValueError("partially filled replay has an inconsistent insertion count")
        slots = np.arange(n)
        expected_ids = next_id - 1 - ((next_id - 1 - slots) % capacity)
        if not np.array_equal(arrays["slot"], slots) or not np.array_equal(arrays["transition_id"], expected_ids):
            raise ValueError("transition IDs disagree with FIFO slot order")
        if not isinstance(state["payloads"], (list, tuple)) or len(state["payloads"]) != n:
            raise ValueError("payload count disagrees with replay size")
        restored.rng.bit_generator.state = deepcopy(state["rng_state"])
        restored._size, restored._position, restored._next_id = n, position, next_id
        restored._raw[:n] = raw
        restored._td_abs[:n] = td
        restored._tags[:n] = arrays["tag"]
        restored._ids[:n] = arrays["transition_id"]
        restored._counts[:n] = arrays["replay_count"]
        restored._payloads[:n] = deepcopy(state["payloads"])
        restored._group_sizes[:] = np.bincount(restored._tags[:n], minlength=2)
        restored._set_leaves(np.arange(n), restored._tags[:n], restored._raw[:n])
        return restored

    def load_state_dict(self, state):
        """Atomically replace replay state; a failed load leaves this buffer intact."""
        restored = self.from_state_dict(state)
        self.__dict__.update(restored.__dict__)

    def check_invariants(self):
        """Expensive independent check for tests/pilots, not the training hot path."""
        n, base = self._size, self._base
        assert self._group_sizes.sum() == n
        assert np.array_equal(self._group_sizes, np.bincount(self._tags[:n], minlength=2))
        assert np.all(self._raw[:n] >= EPSILON)
        assert len(np.unique(self._ids[:n])) == n
        for group in (0, 1):
            expected = np.zeros(base)
            occupied = np.flatnonzero(self._tags[:n] == group)
            expected[occupied] = self._raw[occupied] ** ALPHA
            np.testing.assert_allclose(self._sum[group, base:], expected, rtol=1e-13)
            expected_min = np.where(expected > 0, expected, np.inf)
            np.testing.assert_allclose(self._min[group, base:], expected_min, rtol=1e-13)
            for node in range(base - 1, 0, -1):
                np.testing.assert_allclose(self._sum[group, node], self._sum[group, 2*node:2*node+2].sum(), rtol=1e-13)
                assert self._min[group, node] == self._min[group, 2*node:2*node+2].min()
        assert self._max[1] == (self._raw[:n].max() if n else 0)
        for method in METHODS:
            if n:
                p = self.probabilities(method)
                assert np.all(p > 0)
                np.testing.assert_allclose(p.sum(), 1.0, rtol=1e-13)
                np.testing.assert_allclose(p.min(), self.minimum_probability(method), rtol=1e-13)
