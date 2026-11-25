# Integration with the current team repository

The original `env/doom_env.py` and `env/outage.py` are from main commit `0398f7b`.
This branch adds `replay/` and `demo/` without rewriting that environment or
changing rewards, observations, learner architecture or the main protocol.

Earlier standalone audit documents predate access to this repository. Their
notes about unavailable team code describe that stage; this file is the current handoff.

## Capture the correct flags

`env.delivered` is the current injected delivery flag. Capture it **before**
stepping. The returned `info['delivered']` is a newly generated candidate next flag.

```python
import numpy as np
from outage_replay import ReplayBuffer, beta_schedule

replay = ReplayBuffer(capacity=50_000, seed=replay_seed)
observation = env.reset(seed=scenario_seed)

current_delivered = bool(env.delivered)
next_observation, reward, done, info = env.step(action)
timeout = bool(done and env.game.is_episode_timeout_reached())
terminated = bool(done and not timeout)

# Adapt the opaque payload to the shared learner's existing transition type.
payload = {
    'observation': np.array(observation, copy=True),
    'action': action,
    'reward': reward,
    'next_observation': None if done else np.array(next_observation, copy=True),
    'terminated': terminated,
    'timeout': timeout,
}
slot, transition_id = replay.add(
    payload,
    current_delivered=current_delivered,
    next_delivered=None if done else bool(info['delivered']),
    terminated=terminated,
    timeout=timeout,
)
observation = next_observation
```

The buffer computes the tag. Terminal next flags are ignored, including timeout.
Detection is never the grouping signal. Copy arrays before they can be reused.
The optimizer example uses a `Transition` dataclass instead of this illustrative
dictionary; replay accepts either type, but the common learner must use one contract.

## Learner hooks

```python
batch = replay.sample(64, method=method,
                      beta=beta_schedule(total_collected_decisions, 300_000))
# Double DQN targets: no next-state bootstrap at any finite task end.
# unreduced_huber.shape == weights.shape == (64,)
loss = (weights * unreduced_huber).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
replay.update_priorities(batch.indices, detached_td_errors,
                         transition_ids=batch.transition_ids)
```

Use full returned probabilities and global weights from sampling time. Do not
normalize by batch max or sum(weights), or apply alpha a second time. Repeated
draws remain in the loss; duplicate refreshes use max absolute TD residual.

`replay/examples/learner_integration.py` is a tested optimizer adapter with finite
TD/loss/gradient checks and terminal masking. It accepts the team's networks and
optimizer; it is not a training loop. Further logging/checkpoint examples are in
`replay/docs/INTEGRATION.md`.

## Environment issues before main training

- **Timeout:** current `info['timeout']` uses `self.t >= 75`, while engine start
  is tic 16 and timeout is tic 300. Use the engine timeout predicate; define one
  actual-remaining-time encoding consistently for all methods. At terminal the
  engine clock can reset to zero: do not interpret that as a fresh full budget.
- **Terminal channel count:** step generates an outage flag even when the task
  has ended. Do not use that candidate flag to tag a nonexistent next observation
  or count it as a real delivered/lost frame. Clarify the missingness denominator.
- **Seeds:** reset currently uses one value for engine and channel RNG. Add/record
  explicit paired scenario/mask seeds and disjoint pilot/train/held-out sets.
- **Context:** verify normalization bounds, first history fill, remaining time,
  detection/initialization masks and autoreset handling before the common pilot.

The existing wrapper is preserved so Leon and Areg can review environment changes
and apply them to every sampler together.

## Demo scope

The real engine demo uses an untrained heuristic with delivered label positions
and own heading/spawn location. It is not the team's four-step RL vector policy.
Observer truth, outcomes and hidden velocity do not enter its current observation.
Do not treat these 12 episodes as main training/evaluation data or tune the
research protocol using their outcomes.

Ready: replay, checks, diagnostics, frozen analysis, plots, timings, Methods and demo.
Still needed: shared learner/training, environment fixes, real training runs/logs,
final checkpoints, paired evaluation and uncertainty estimates.
