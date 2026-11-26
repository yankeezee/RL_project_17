import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, sampler="uniform",
                 alpha=0.6):
        self.capacity = capacity
        self.sampler = sampler
        self.alpha = alpha

        self.obs = np.zeros(
            (capacity, obs_dim), dtype=np.float32
        )
        self.next_obs = np.zeros_like(self.obs)

        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.priorities = np.ones(capacity, dtype=np.float32)
        self.outages = np.zeros(capacity, dtype=bool)

        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, obs, action, reward, next_obs, done,
             outage=False):

        i = self.position

        max_priority = (
            self.priorities[:self.size].max()
            if self.size > 0 else 1.0
        )

        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)
        self.outages[i] = outage
        self.priorities[i] = max_priority

        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def probabilities(self):
        if self.sampler == "uniform":
            return np.full(self.size, 1.0 / self.size)

        if self.sampler == "per":
            priorities = (
                self.priorities[:self.size] + 1e-6
            ) ** self.alpha

            return priorities / priorities.sum()

        raise ValueError(f"Unknown sampler: {self.sampler}")

    def sample(self, batch_size, rng, beta=0.4):
        probs = self.probabilities()

        indices = rng.choice(
            self.size,
            size=batch_size,
            replace=True,
            p=probs
        )

        weights = (self.size * probs[indices]) ** (-beta)

        max_weight = (self.size * probs.min()) ** (-beta)
        weights /= max_weight

        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
            "outages": self.outages[indices],
            "indices": indices,
            "weights": weights.astype(np.float32),
        }

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = np.abs(td_errors)