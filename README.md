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
agents.py       agents (only a random agent for now)
test_env.py     tests the environment with a random agent
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
