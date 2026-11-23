"""Quick test of the environment with a random agent."""
import numpy as np

from env import PredictPositionEnv, BurstOutage, make_outage

N_EPISODES = 20

for cond in ["clean", "indep25", "burst25_L4", "burst25_L8"]:
    env = PredictPositionEnv(outage=make_outage(cond))
    returns, hits, missing, lengths = [], [], [], []
    for ep in range(N_EPISODES):
        obs = env.reset(seed=ep)
        assert obs.shape == (env.obs_size,)
        done, R = False, 0.0
        while not done:
            action = np.random.randint(env.n_actions)
            obs, r, done, info = env.step(action)
            R += r
        returns.append(R)
        hits.append(info["hit"])
        missing.append(info["missing_fraction"])
        lengths.append(info["t"])
    env.close()
    print(f"{cond:12s} | mean return {np.mean(returns):+.3f} | hit rate {np.mean(hits):.2f} "
          f"| length {np.mean(lengths):.1f} | lost frames {np.mean(missing):.2f}")

# check the burst process on a long sequence
o = BurstOutage(p=0.25, L=4)
o.reset(np.random.default_rng(0))
seq = np.array([o.step() for _ in range(200_000)])
print(f"\nBurstOutage(p=0.25, L=4): lost fraction = {1 - seq.mean():.3f} (expected 0.25)")
