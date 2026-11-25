"""Minimal Double DQN update adapter using the team's networks and optimizer.

API verified against PyTorch 2.10 official documentation; see docs/INTEGRATION.md.
Payload = Transition, observation is a caller-owned immutable numeric array.
No architecture, reward, observation, schedule, or exploration policy is defined here.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from outage_replay import beta_schedule


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray | None
    terminated: bool
    timeout: bool


def learner_step(*, buffer, online, target, optimizer, decisions, optimizer_step,
                 method="grouped", total_decisions=300_000, batch_size=64,
                 gamma=.99, logger=None):
    """Call only when the shared learner's update schedule authorizes an update."""
    batch = buffer.sample(batch_size, method=method,
                          beta=beta_schedule(decisions, total_decisions))
    device = next(online.parameters()).device
    states = torch.as_tensor(np.stack([t.observation for t in batch.transitions]),
                             device=device, dtype=torch.float32)
    actions = torch.tensor([t.action for t in batch.transitions], device=device, dtype=torch.long)
    rewards = torch.tensor([t.reward for t in batch.transitions], device=device, dtype=torch.float32)
    active = np.array([not (t.terminated or t.timeout) for t in batch.transitions])
    # Do not evaluate terminal/reset observations: absent or NaN terminal frames
    # cannot contaminate the target through the expression 0 * NaN.
    with torch.no_grad():
        targets = rewards.clone()
        if active.any():
            next_states = torch.as_tensor(np.stack([t.next_observation for t, a in
                                                    zip(batch.transitions, active) if a]),
                                          device=device, dtype=torch.float32)
            chosen_actions = online(next_states).argmax(dim=1)
            next_values = target(next_states).gather(1, chosen_actions[:, None]).squeeze(1)
            targets[torch.as_tensor(active, device=device)] += gamma * next_values
    predictions = online(states).gather(1, actions[:, None]).squeeze(1)
    td_errors = targets - predictions
    unreduced = F.smooth_l1_loss(predictions, targets, reduction="none", beta=1.0)
    weights = torch.as_tensor(batch.weights, device=device, dtype=unreduced.dtype)
    loss = (weights * unreduced).mean()
    td_cpu = td_errors.detach().cpu().numpy().copy()
    huber_cpu = unreduced.detach().cpu().numpy().copy()
    if not (np.all(np.isfinite(td_cpu)) and np.all(np.isfinite(huber_cpu))
            and bool(torch.isfinite(loss))):
        raise ValueError("TD errors and loss must be finite before optimizer.step")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
           for group in optimizer.param_groups for p in group["params"]):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("Gradients must be finite before optimizer.step")
    optimizer.step()
    # Refresh after the step using the residuals from this exact loss; do not
    # apply alpha in the learner or recompute unrelated post-step residuals.
    buffer.update_priorities(batch.indices, td_cpu, transition_ids=batch.transition_ids)
    if logger is not None:
        logger.log(batch, td_errors=td_cpu, unreduced_huber=huber_cpu,
                   decisions=decisions, optimizer_step=optimizer_step)
    return {"loss": float(loss.detach().cpu()), "batch": batch, "td_errors": td_cpu,
            "unreduced_huber": huber_cpu}
