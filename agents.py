"""Agents that can be plugged into PredictPositionEnv.

Every agent has an `act(obs)` method that returns an action index.
To add a new agent, write its class here and register it in AGENTS.
"""
import numpy as np

from env.doom_env import N_ACTIONS


class RandomAgent:
    def act(self, obs):
        return np.random.randint(N_ACTIONS)


AGENTS = {
    "random": RandomAgent,
}


def make_agent(name):
    if name not in AGENTS:
        raise ValueError(f"unknown agent: {name} (available: {list(AGENTS)})")
    return AGENTS[name]()
