"""Disjoint episode seed ranges.

One episode seed fixes both the scenario initialization and the outage mask
(see ``PredictPositionEnv.reset``). Training, pilot and held-out evaluation
episodes must never share a seed, and every method must be evaluated on the
same held-out seeds so that comparisons are paired.

| use                        | seeds                                       |
|----------------------------|---------------------------------------------|
| training episodes          | 1_000_000 + 100_000 * train_seed + episode  |
| pilot / tuning episodes    | 5_000_000 + k                               |
| learning-curve evaluation  | 7_000_000 + k                               |
| held-out evaluation        | 9_000_000 + k, k < 100                      |
"""

TRAIN_SEED_BASE = 1_000_000
EPISODES_PER_TRAIN_SEED = 100_000
PILOT_SEED_BASE = 5_000_000
CURVE_SEED_BASE = 7_000_000
EVAL_SEED_BASE = 9_000_000
N_EVAL_SEEDS = 100


def train_episode_seed(train_seed, episode):
    """Seed of the ``episode``-th training episode of run ``train_seed``."""
    if not 0 <= episode < EPISODES_PER_TRAIN_SEED:
        raise ValueError("episode index outside the reserved training range")
    if not 0 <= train_seed < (PILOT_SEED_BASE - TRAIN_SEED_BASE) // EPISODES_PER_TRAIN_SEED:
        raise ValueError("train_seed outside the reserved training range")
    return TRAIN_SEED_BASE + EPISODES_PER_TRAIN_SEED * train_seed + episode


def pilot_seeds(n):
    return [PILOT_SEED_BASE + k for k in range(n)]


def curve_eval_seeds(n):
    return [CURVE_SEED_BASE + k for k in range(n)]


def eval_seeds(n=N_EVAL_SEEDS):
    """The shared held-out seeds used by every method in every test condition."""
    if n > N_EVAL_SEEDS:
        raise ValueError(f"only {N_EVAL_SEEDS} held-out seeds are reserved")
    return [EVAL_SEED_BASE + k for k in range(n)]
