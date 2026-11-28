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
evaluate.py     paired evaluation of frozen policies (4 test conditions x 100 shared seeds)
paired_stats.py seed-level differences, hierarchical paired bootstrap, summary table
test_evaluation.py  checks of the evaluation and statistics
watch.py        watch an agent play in the ViZDoom window
play.py         play the scenario yourself
```

## Test

```
python test_env.py
python test_evaluation.py
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

## Paired evaluation

Every frozen policy is run greedily on the same 100 held-out seeds
(`eval_seeds()`) in the four test conditions `clean`, `indep25`, `burst25_L4`
and `burst25_L8`. Methods therefore face identical starts and identical outage
masks. Checkpoints are the `state_dict` files written by `train.py`, named
`ddqn_<sampler>_<train_condition>_seed<k>.pt`.

```
python evaluate.py --checkpoints checkpoints/ --out results/eval.csv --workers 6
python evaluate.py --agents random --out results/eval.csv        # sanity reference
```

One CSV row per episode: method, training regime, training seed, test condition,
evaluation seed, return, hit, outcome (`hit` / `miss` / `timeout`), length,
realized missing fraction and number of FIRE actions. Cells already complete in
the output file are skipped, so an interrupted evaluation can be resumed.
The full grid (24 policies x 4 conditions x 100 seeds = 9,600 episodes) takes
roughly 40 min with 6 workers on a 12-core laptop.

```
python paired_stats.py results/eval.csv            # primary: grouped - PER, train burst25_L4, test burst25_L8
python paired_stats.py results/eval.csv --all      # grouped vs PER / mixture / uniform, mixture vs PER
python paired_stats.py results/eval.csv --summary  # hit rate, return, misses, timeouts per cell
python paired_stats.py results/eval.csv --metric return --a grouped --b mixture
```

For each comparison: the difference of means, the seed-level differences
(one per training seed) and a 95 % hierarchical paired bootstrap interval
(training seeds, then evaluation seeds, same indices for both methods).
An interval containing 0 is reported as unresolved. With three training seeds,
the intervals are coarse.
