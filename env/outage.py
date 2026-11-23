"""Observation outage processes.

At each step, every model returns a boolean `delivered`:
True = the frame arrived, False = the observation was lost.
"""
import numpy as np


class NoOutage:
    """Observations are always delivered ('clean' condition)."""

    def reset(self, rng):
        self.rng = rng
        return True

    def step(self):
        return True


class IndependentOutage:
    """Each frame is lost independently with probability p."""

    def __init__(self, p=0.25):
        self.p = p

    def reset(self, rng):
        self.rng = rng
        return True  # the channel is available at the start of the episode

    def step(self):
        return bool(self.rng.random() >= self.p)


class BurstOutage:
    """Burst losses (2-state Markov chain).

    p = stationary fraction of lost frames, L = mean burst length.
    - recovery probability (outage -> ok): 1 / L
    - outage onset probability (ok -> outage): p / (L * (1 - p))
    """

    def __init__(self, p=0.25, L=4):
        self.p = p
        self.L = L
        self.p_recover = 1.0 / L
        self.p_onset = p / (L * (1.0 - p))

    def reset(self, rng):
        self.rng = rng
        self.available = True  # the channel is available at the start of the episode
        return True

    def step(self):
        if self.available:
            if self.rng.random() < self.p_onset:
                self.available = False
        else:
            if self.rng.random() < self.p_recover:
                self.available = True
        return self.available


def make_outage(name):
    """Small helper to build an outage model from a condition name."""
    if name == "clean":
        return NoOutage()
    if name == "indep25":
        return IndependentOutage(p=0.25)
    if name == "burst25_L4":
        return BurstOutage(p=0.25, L=4)
    if name == "burst25_L8":
        return BurstOutage(p=0.25, L=8)
    raise ValueError(f"unknown condition: {name}")
