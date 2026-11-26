import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mlp(in_dim, out_dim, hidden=(64, 64)):
    layers = []
    last = in_dim

    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h

    layers.append(nn.Linear(last, out_dim))

    return nn.Sequential(*layers).to(DEVICE)


def as_tensor(x, dtype=torch.float32):
    return torch.as_tensor(
        np.asarray(x), dtype=dtype, device=DEVICE
    )


@torch.no_grad()
def dqn_target(batch, q_net, target_net, gamma=0.99):
    best = q_net(batch["next_obs"]).argmax(dim=1)

    next_q = target_net(batch["next_obs"]).gather(
        1, best.view(-1, 1)
    ).squeeze(1)

    return (
        batch["rewards"]
        + gamma * (1 - batch["dones"]) * next_q
    )