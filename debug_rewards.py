import numpy as np
from env import PredictPositionEnv, make_outage


def test_policy(name, policy, n_episodes=100):
    env = PredictPositionEnv(outage=make_outage("clean"))

    returns, hits, lengths = [], [], []

    for ep in range(n_episodes):
        obs = env.reset(seed=10000 + ep)
        rng = np.random.default_rng(ep)

        done = False
        R = 0
        t = 0

        while not done:
            action = policy(obs, t, rng)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1

        returns.append(R)
        hits.append(info["hit"])
        lengths.append(t)

    env.close()

    print(
        f"{name:15s} | "
        f"Return: {np.mean(returns):+.4f} | "
        f"Hit rate: {np.mean(hits):.3f} | "
        f"Length: {np.mean(lengths):.2f}"
    )


test_policy("Random", lambda obs, t, rng: rng.integers(4))
test_policy("Always fire", lambda obs, t, rng: 3)
test_policy("Always wait", lambda obs, t, rng: 0)
test_policy("Wait then fire", lambda obs, t, rng: 0 if t < 20 else 3)
test_policy(
    "Alternate fire",
    lambda obs, t, rng: 3 if t % 2 == 0 else 0
)

test_policy(
    "Wait 5 then fire",
    lambda obs, t, rng: 0 if t < 5 else 3
)

test_policy(
    "Wait 5 alternate",
    lambda obs, t, rng: 0 if t < 5 else (
        3 if t % 2 == 0 else 0
    )
)