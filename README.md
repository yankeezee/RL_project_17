# RL_project_17

Does prioritized replay oversample observation outages in ViZDoom?
(see `Research_Proposal_RL2026.md`)

## Installation

Python 3.12, no virtual environment:

```
pip install -r requirements.txt
```

## Structure

```
env/
  doom_env.py   predict_position environment (observations from labels + 4-step history)
  outage.py     observation loss models (clean, independent, bursts)
  seeds.py      disjoint seed ranges (training / pilot / learning curve / held-out)
agents.py       agents (only a random agent for now)
test_env.py     checks the environment on the real engine (about 2.5 min)
watch.py        watch an agent play in the ViZDoom window
play.py         play the scenario yourself
```

## Test

```
python test_env.py
```

## Watch an agent / play yourself

```
python watch.py --agent random --condition burst25_L4 --episodes 5
python play.py --episodes 5
```

In `play.py`: left/right arrows (or mouse) to turn, Ctrl (or left click) to fire.

To add an agent, write a class with an `act(obs)` method in `agents.py`
and register it in the `AGENTS` dictionary. It then becomes available in `watch.py --agent`.

## Observation conditions

| name         | description                                    |
|--------------|------------------------------------------------|
| `clean`      | no loss                                        |
| `indep25`    | 25% of frames lost independently               |
| `burst25_L4` | 25% loss in bursts of mean length 4            |
| `burst25_L8` | 25% loss in bursts of mean length 8            |

## Environment interface

```python
from env import PredictPositionEnv, make_outage, train_episode_seed

env = PredictPositionEnv(outage=make_outage("burst25_L4"))
obs = env.reset(seed=train_episode_seed(train_seed=0, episode=0))  # obs: float32[49]
current_delivered = env.delivered          # flag of the current observation
next_obs, reward, done, info = env.step(action)
```

- `done` ends the task: kill, missed rocket (ends the episode early) or the
  scenario timeout (75 decisions). After `done` there is no next observation:
  do not bootstrap, and `info["delivered"]` is `None`.
- Outage tag of a transition: `not current_delivered or (not done and not info["delivered"])`.
- `info["hit"]` (set on the terminal step) is true if the target was killed at
  any point of the episode; the kill reward can arrive one decision before the end.
- `info["timeout"]` uses the engine's timeout predicate and is never true for a hit.
- `info["missing_fraction"]` = lost frames / observation frames actually produced
  (the reset frame and every non-terminal next frame).
- One seed fixes both the scenario start and the outage mask; the mask does not
  depend on actions. Use the ranges of `env/seeds.py`, which never overlap:

| use | seeds |
|---|---|
| training episodes | `train_episode_seed(train_seed, episode)` = 1,000,000 + 100,000·train_seed + episode |
| pilot / tuning | `pilot_seeds(n)` = 5,000,000 + k |
| learning-curve evaluation | `curve_eval_seeds(n)` = 7,000,000 + k |
| held-out evaluation (shared by all methods) | `eval_seeds()` = 9,000,000 + k, k < 100 |

### Observation (4 history entries of 12 values + remaining time)

| values | content |
|---|---|
| 2 | cos / sin of the player heading |
| 4 | previous action, one-hot (all zeros before the first action) |
| 1 | delivered flag (injected by the outage process) |
| 1 | target detected (delivered and target label visible) |
| 2 | last measured target position, normalized to [-1, 1] with fixed bounds x in [-96, 256], y in [-640, 640] |
| 1 | age of that measurement (steps / 20, saturated; 0 before the first detection) |
| 1 | initialized flag (0 until the first detection) |

Missing positions are forward-filled. An outage (`delivered = 0`) is encoded
differently from a delivered frame without a visible target (`delivered = 1, detected = 0`).
