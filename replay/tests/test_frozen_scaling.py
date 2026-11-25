import numpy as np

from outage_replay import ReplayBuffer
from outage_replay import frozen


def test_frozen_minimum_reductions_are_constant_not_per_row(tmp_path, monkeypatch):
    buffer = ReplayBuffer(200)
    for k in range(200):
        buffer.add(None, current_delivered=k % 3 != 0, next_delivered=True, terminated=False)
    source = frozen.export_frozen(buffer, tmp_path / "source", synthetic=True)
    original = frozen.probability_vectors

    class Counter(np.ndarray):
        calls = 0

        def min(self, *args, **kwargs):
            Counter.calls += 1
            return super().min(*args, **kwargs)

    def counted(raw, tags):
        return {m: p.view(Counter) for m, p in original(raw, tags).items()}

    monkeypatch.setattr(frozen, "probability_vectors", counted)
    frozen.analyze_frozen(source, tmp_path / "output")
    # One per method, reused for weights and summary, independent of buffer N.
    assert Counter.calls == 4, f"Full-vector minimum reductions: {Counter.calls}"
