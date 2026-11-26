import numpy as np
import torch

from env import PredictPositionEnv, make_outage
from networks import mlp, as_tensor, DEVICE

env = PredictPositionEnv(
    outage=make_outage("clean")
)

q_net = mlp(env.obs_size, env.n_actions)

q_net.load_state_dict(
    torch.load(
        "ddqn_uniform_clean_seed0.pt",
        map_location=DEVICE
    )
)

q_net.eval()

action_counts = np.zeros(4)
first_fire_steps = []
returns = []
hits = []
timeouts = []
detection_rates = []
lengths = []

for ep in range(100):
    obs = env.reset(seed=10000 + ep)

    done = False
    R = 0
    t = 0
    first_fire = None
    detections = []

    while not done:
        with torch.no_grad():
            q_values = q_net(
                as_tensor(obs).unsqueeze(0)
            )
            action = q_values.argmax(dim=1).item()

        # Detection flag of the most recent history entry
        detections.append(obs[43])

        action_counts[action] += 1

        if action == 3 and first_fire is None:
            first_fire = t

        obs, reward, done, info = env.step(action)

        R += reward
        t += 1

    if first_fire is not None:
        first_fire_steps.append(first_fire)

    returns.append(R)
    hits.append(info["hit"])
    timeouts.append(info["timeout"])
    detection_rates.append(np.mean(detections))
    lengths.append(t)

env.close()

print("Action distribution:")
for name, count in zip(
    ["No-op", "Left", "Right", "Fire"],
    action_counts
):
    print(f"{name}: {count / action_counts.sum():.2%}")

print("Mean return:", np.mean(returns))
print("Hit rate:", np.mean(hits))
print("Timeout rate:", np.mean(timeouts))
print("Mean episode length:", np.mean(lengths))
print("Detection rate:", np.mean(detection_rates))
print("Episodes with fire action:", len(first_fire_steps))

print(
    "Mean first fire step:",
    np.mean(first_fire_steps) if first_fire_steps else None
)