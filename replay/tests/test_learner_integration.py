import numpy as np
import pytest

torch = pytest.importorskip("torch")

from examples.learner_integration import Transition, learner_step
from outage_replay import ReplayBuffer
from outage_replay.diagnostics import huber


@pytest.mark.parametrize("method", ["uniform", "per", "mixture", "grouped"])
def test_real_optimizer_step_weights_priorities_and_no_target_grad(method):
    torch.manual_seed(418)
    buffer = ReplayBuffer(32, seed=90)
    for k in range(20):
        end = k % 3 == 0
        timeout = k % 5 == 0
        t = Transition(np.array([k / 20, 1], dtype=np.float32), k % 2, float(k % 4),
                       None if end or timeout else np.array([(k+1)/20, 1], dtype=np.float32),
                       end, timeout)
        buffer.add(t, current_delivered=k % 4 != 0, next_delivered=None if end or timeout else True,
                   terminated=end, timeout=timeout)
    online = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(),
                                 torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    target = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(),
                                 torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in online.parameters()]
    result = learner_step(buffer=buffer, online=online, target=target, optimizer=optimizer,
                          decisions=5000, optimizer_step=1, method=method)
    batch, td = result["batch"], result["td_errors"]
    assert result["loss"] == pytest.approx(np.mean(batch.weights * huber(td)), rel=2e-6)
    for index in np.unique(batch.indices):
        assert buffer.snapshot()["raw_priority"][index] == pytest.approx(abs(td[batch.indices == index]).max()+1e-6)
    assert any(not torch.equal(old, new) for old, new in zip(before, online.parameters()))
    assert all(p.grad is None for p in target.parameters())


@pytest.mark.parametrize("timeout", [False, True])
def test_terminal_and_timeout_have_reward_only_target(timeout):
    buffer = ReplayBuffer(1)
    transition = Transition(np.ones(2, dtype=np.float32), 0, 2., None, not timeout, timeout)
    buffer.add(transition, current_delivered=True, next_delivered=None,
               terminated=not timeout, timeout=timeout)
    online, target = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    with torch.no_grad():
        online.weight.zero_()
        online.bias.zero_()
        target.weight.fill_(float("nan"))
        target.bias.fill_(float("nan"))
    result = learner_step(buffer=buffer, online=online, target=target,
                          optimizer=torch.optim.Adam(online.parameters(), lr=1e-4),
                          decisions=5000, optimizer_step=1)
    np.testing.assert_array_equal(result["td_errors"], np.full(64, 2.))
    assert result["loss"] == 1.5
