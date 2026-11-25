import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from examples.learner_integration import Transition, learner_step
from outage_replay import ReplayBuffer


class ScaledLinear(torch.nn.Linear):
    def forward(self, inputs):
        return 2 * super().forward(inputs)


@pytest.mark.parametrize("failure", ["nan_reward", "gradient_overflow"])
def test_invalid_update_never_mutates_parameters_adam_or_priorities(failure):
    online, target = ScaledLinear(1, 1), torch.nn.Linear(1, 1)
    with torch.no_grad():
        online.weight.zero_()
        online.bias.zero_()
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    buffer = ReplayBuffer(1)
    # finite forward/loss but accumulated backward gradient overflows for large input.
    observation = np.array([3e38 if failure == "gradient_overflow" else 1.], dtype=np.float32)
    transition = Transition(observation, 0, float("nan") if failure == "nan_reward" else 2.,
                            None, True, False)
    buffer.add(transition, current_delivered=True, next_delivered=None, terminated=True)
    parameters = [p.detach().clone() for p in online.parameters()]
    state = copy.deepcopy(optimizer.state_dict())
    priorities = buffer.snapshot()["raw_priority"].copy()
    with pytest.raises(ValueError, match="finite"):
        learner_step(buffer=buffer, online=online, target=target, optimizer=optimizer,
                     decisions=5000, optimizer_step=1)
    for old, new in zip(parameters, online.parameters()):
        assert torch.equal(old, new)
    assert optimizer.state_dict() == state
    np.testing.assert_array_equal(buffer.snapshot()["raw_priority"], priorities)
