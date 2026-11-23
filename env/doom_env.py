"""ViZDoom 'predict_position' environment with intermittent observations.

The agent does not see pixels: it receives a vector built from the ViZDoom
labels (target position when it is visible), with a history of the last
4 steps.
"""
import os
from collections import deque

import numpy as np
import vizdoom as vzd

from env.outage import NoOutage

# actions: no-op, turn left, turn right, fire
# buttons in the cfg: TURN_LEFT, TURN_RIGHT, ATTACK
ACTIONS = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
]
N_ACTIONS = len(ACTIONS)
FRAME_SKIP = 4       # each action is held for 4 tics
HISTORY_LEN = 4
TARGET_NAME = "Cacodemon"

# fixed bound used to normalize positions (approx. map size)
POS_SCALE = 600.0
MAX_AGE = 20         # age (in steps) after which we saturate
EPISODE_TICS = 300   # episode_timeout from the cfg
MAX_STEPS = EPISODE_TICS // FRAME_SKIP

# size of one history entry:
# heading (cos, sin) + previous action one-hot (4) + delivered + detected
# + last position (x, y) + age + initialized
ENTRY_SIZE = 2 + N_ACTIONS + 1 + 1 + 2 + 1 + 1
OBS_SIZE = HISTORY_LEN * ENTRY_SIZE + 1  # +1 for the remaining time


class PredictPositionEnv:
    def __init__(self, outage=None, render=False):
        self.outage = outage if outage is not None else NoOutage()

        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, "predict_position.cfg"))
        self.game.set_window_visible(render)
        self.game.set_labels_buffer_enabled(True)   # needed to get state.labels
        self.game.add_available_game_variable(vzd.GameVariable.ANGLE)
        self.game.init()

        self.n_actions = N_ACTIONS
        self.obs_size = OBS_SIZE

    def reset(self, seed=None):
        if seed is not None:
            self.game.set_seed(int(seed))
        self.rng = np.random.default_rng(seed)
        self.game.new_episode()

        self.t = 0
        self.prev_action = 0
        self.last_pos = np.zeros(2)   # zero position until the first detection
        self.age = 0
        self.initialized = False
        self.history = deque(maxlen=HISTORY_LEN)
        self.n_missing = 0

        delivered = self.outage.reset(self.rng)
        entry = self._make_entry(delivered)
        # fill the history with the first entry
        for _ in range(HISTORY_LEN):
            self.history.append(entry)
        self.delivered = delivered
        return self._get_obs()

    def step(self, action):
        reward = self.game.make_action(ACTIONS[action], FRAME_SKIP)
        self.t += 1
        self.prev_action = action
        done = self.game.is_episode_finished()

        delivered = self.outage.step()
        if not delivered:
            self.n_missing += 1

        # once the episode is over there is no state anymore, keep the old obs
        if not done:
            self.history.append(self._make_entry(delivered))
        self.delivered = delivered

        info = {
            "delivered": delivered,
            "hit": done and reward > 0.5,  # positive reward when the target is killed
            "timeout": done and self.t >= MAX_STEPS,
            "t": self.t,
        }
        if done:
            info["missing_fraction"] = self.n_missing / self.t
        return self._get_obs(), reward, done, info

    def close(self):
        self.game.close()

    def _make_entry(self, delivered):
        state = self.game.get_state()
        heading = np.deg2rad(state.game_variables[0])

        detected = False
        if delivered:
            for label in state.labels:
                if label.object_name == TARGET_NAME:
                    self.last_pos = np.array([label.object_position_x,
                                              label.object_position_y]) / POS_SCALE
                    detected = True
                    break

        if detected:
            self.age = 0
            self.initialized = True
        else:
            self.age += 1   # forward-fill the position and count how old it is

        action_onehot = np.zeros(N_ACTIONS)
        action_onehot[self.prev_action] = 1.0

        entry = np.concatenate([
            [np.cos(heading), np.sin(heading)],
            action_onehot,
            [float(delivered), float(detected)],
            self.last_pos,
            [min(self.age, MAX_AGE) / MAX_AGE, float(self.initialized)],
        ])
        return entry.astype(np.float32)

    def _get_obs(self):
        time_left = 1.0 - self.t / MAX_STEPS
        return np.concatenate(list(self.history) + [[time_left]]).astype(np.float32)
