"""Checks of the environment on the real ViZDoom engine (python test_env.py)."""
import numpy as np

from env import (BurstOutage, PredictPositionEnv, make_outage, curve_eval_seeds,
                 eval_seeds, pilot_seeds, train_episode_seed)
from env.doom_env import ENTRY_SIZE, N_ACTIONS

N_EPISODES = 20
LAST = slice((4 - 1) * ENTRY_SIZE, 4 * ENTRY_SIZE)  # newest history entry


def run_episode(env, seed, policy):
    obs = env.reset(seed=seed)
    flags = [env.delivered]
    done, R, killed, t = False, 0.0, False, 0
    while not done:
        obs, r, done, info = env.step(policy(t, obs))
        R += r
        killed |= r > 0.5
        flags.append(info["delivered"])
        t += 1
    return R, killed, flags, info


# 1) summary per condition with a random agent
rng = np.random.default_rng(0)
random_policy = lambda t, obs: int(rng.integers(N_ACTIONS))
for cond in ["clean", "indep25", "burst25_L4", "burst25_L8"]:
    env = PredictPositionEnv(outage=make_outage(cond))
    returns, hits, missing, lengths = [], [], [], []
    for ep in range(N_EPISODES):
        R, killed, _, info = run_episode(env, pilot_seeds(N_EPISODES)[ep], random_policy)
        returns.append(R)
        hits.append(info["hit"])
        missing.append(info["missing_fraction"])
        lengths.append(info["t"])
    env.close()
    print(f"{cond:12s} | mean return {np.mean(returns):+.3f} | hit rate {np.mean(hits):.2f} "
          f"| length {np.mean(lengths):.1f} | lost frames {np.mean(missing):.2f}")

env = PredictPositionEnv(outage=make_outage("burst25_L4"))

# 2) reset observation: right size, no previous action, full remaining time
obs = env.reset(seed=pilot_seeds(1)[0])
assert obs.shape == (env.obs_size,)
assert (obs[LAST][2:2 + N_ACTIONS] == 0).all(), "no previous action at reset"
assert obs[-1] == 1.0

# 3) a no-op episode reaches the engine timeout after 75 decisions
_, killed, flags, info = run_episode(env, pilot_seeds(2)[1], lambda t, obs: 0)
assert info["t"] == env.max_steps == 75, info["t"]
assert info["timeout"] and not info["hit"] and not killed
assert flags[-1] is None, "no outage flag is drawn for the nonexistent terminal frame"
assert all(f is not None for f in flags[:-1])
print("\nno-op episode: 75 decisions, timeout flag set, no terminal outage flag")

# 4) every kill is counted as a hit, even when the reward arrives before the end
n_kill = n_hit = 0
positions = []
for seed in pilot_seeds(150):
    obs = env.reset(seed=seed)
    done, killed = False, False
    while not done:
        obs, r, done, info = env.step(int(rng.integers(N_ACTIONS)))
        killed |= r > 0.5
        positions.append(obs[LAST][8:10])
    n_kill += killed
    n_hit += info["hit"]
    assert info["hit"] == killed
    assert not (info["hit"] and info["timeout"])
assert n_kill > 0, "no kill in 150 random episodes; increase the sample"
positions = np.array(positions)
assert np.abs(positions).max() <= 1.0
print(f"random episodes: {n_kill} kills, {n_hit} counted as hits")
print(f"normalized positions within [{positions.min():.2f}, {positions.max():.2f}]")

# 5) missing fraction counts real frames only
R, killed, flags, info = run_episode(env, pilot_seeds(3)[2], random_policy)
real = flags[:-1]
assert np.isclose(info["missing_fraction"], real.count(False) / len(real))

# 6) same seed -> same scenario start and same mask, whatever the policy
seed = eval_seeds(1)[0]
_, _, m1, _ = run_episode(env, seed, lambda t, obs: 0)
_, _, m2, _ = run_episode(env, seed, lambda t, obs: (1, 2)[t % 2])
n = min(len(m1), len(m2)) - 1
assert m1[:n] == m2[:n], "outage mask must not depend on actions"
assert (env.reset(seed=seed) == env.reset(seed=seed)).all()
env.close()
print("same seed gives the same start and the same mask for different policies")

# 7) seed ranges are disjoint
train = {train_episode_seed(k, e) for k in range(3) for e in range(0, 100_000, 7)}
assert train.isdisjoint(pilot_seeds(10_000))
assert train.isdisjoint(curve_eval_seeds(10_000))
assert train.isdisjoint(eval_seeds())
assert set(curve_eval_seeds(10_000)).isdisjoint(eval_seeds())
print("training / pilot / curve / held-out seed ranges are disjoint")

# 8) the burst process on a long sequence
o = BurstOutage(p=0.25, L=4)
o.reset(np.random.default_rng(0))
seq = np.array([o.step() for _ in range(200_000)])
print(f"\nBurstOutage(p=0.25, L=4): lost fraction = {1 - seq.mean():.3f} (expected 0.25)")
assert abs((1 - seq.mean()) - 0.25) < 0.01
print("\nall environment checks passed")
