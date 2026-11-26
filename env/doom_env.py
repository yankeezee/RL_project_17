"""ViZDoom 'predict_position' environment with intermittent observations.

The agent does not see pixels: it receives a vector built from the ViZDoom
labels (target position when it is visible), with a history of the last
4 steps.

One integer seed passed to ``reset`` fixes both the scenario initialization
and the outage mask of the episode. Use the disjoint seed ranges of
``env/seeds.py`` for training, pilot and held-out evaluation episodes.
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
KILL_REWARD_THRESHOLD = 0.5  # kill reward is +1, the time penalty is -0.001 per tic

# Fixed scenario bounds used to normalize target positions to [-1, 1].
# Measured on 160 random-action pilot episodes: visible target positions lie in
# x in [-55, 225] and y in [-607, 607] (the player stands at (-544, 0)). The
# bounds below are rounded outwards with a margin; values outside are clipped.
X_BOUNDS = (-96.0, 256.0)
Y_BOUNDS = (-640.0, 640.0)
MAX_AGE = 20         # age (in steps) after which we saturate

# size of one history entry:
# heading (cos, sin) + previous action one-hot (4) + delivered + detected
# + last position (x, y) + age + initialized
ENTRY_SIZE = 2 + N_ACTIONS + 1 + 1 + 2 + 1 + 1
OBS_SIZE = HISTORY_LEN * ENTRY_SIZE + 1  # +1 for the remaining time


def normalize_position(x, y):
    """Map a world-plane position to [-1, 1]^2 with the fixed scenario bounds."""
    nx = 2.0 * (x - X_BOUNDS[0]) / (X_BOUNDS[1] - X_BOUNDS[0]) - 1.0
    ny = 2.0 * (y - Y_BOUNDS[0]) / (Y_BOUNDS[1] - Y_BOUNDS[0]) - 1.0
    return np.clip(np.array([nx, ny]), -1.0, 1.0)


class PredictPositionEnv:
    def __init__(self, outage=None, render=False):
        self.outage = outage if outage is not None else NoOutage()

        self.game = vzd.DoomGame()
        self.game.load_config(os.path.join(vzd.scenarios_path, "predict_position.cfg"))
        self.game.set_window_visible(render)
        self.game.set_labels_buffer_enabled(True)   # needed to get state.labels
        self.game.add_available_game_variable(vzd.GameVariable.ANGLE)
        self.game.init()

        # The engine's timeout (300 tics) is counted from the episode start,
        # so an episode lasts at most 300 / 4 = 75 decisions.
        self.max_steps = int(np.ceil(self.game.get_episode_timeout() / FRAME_SKIP))
        self.n_actions = N_ACTIONS
        self.obs_size = OBS_SIZE

    def reset(self, seed=None):
        if seed is not None:
            self.game.set_seed(int(seed))
        # The mask RNG has its own stream, fully determined by the seed and
        # independent of the agent's actions.
        self.rng = np.random.default_rng(None if seed is None else [int(seed), 0])
        self.game.new_episode()

        self.t = 0
        self.prev_action = None       # no action taken yet
        self.last_pos = np.zeros(2)   # zero position until the first detection
        self.age = 0
        self.initialized = False
        self.history = deque(maxlen=HISTORY_LEN)
        self.killed = False
        self.n_frames = 0             # observation frames actually produced
        self.n_missing = 0            # ... and how many of them were lost

        delivered = self.outage.reset(self.rng)
        entry = self._make_entry(delivered)
        # fill the history with the first entry
        for _ in range(HISTORY_LEN):
            self.history.append(entry)
        self.delivered = delivered
        return self._get_obs()

    def step(self, action):
        """Returns ``(next_obs, reward, done, info)``.

        ``done`` marks a task-ending transition: kill, missed rocket or scenario
        timeout. When ``done`` is True there is no next observation: the
        returned observation must not be bootstrapped, no outage flag is drawn
        and ``info["delivered"]`` is None. ``env.delivered`` read *before* the
        step is the flag of the current observation.
        """
        reward = self.game.make_action(ACTIONS[action], FRAME_SKIP)
        self.t += 1
        self.prev_action = action
        done = self.game.is_episode_finished()
        # The kill reward can arrive one decision before the episode ends, so
        # the hit is remembered over the whole episode.
        self.killed = self.killed or reward > KILL_REWARD_THRESHOLD

        if done:
            # Terminal: no next frame exists, so no outage flag is drawn and
            # nothing is counted. Keep the previous observation.
            delivered = None
        else:
            delivered = self.outage.step()
            self.history.append(self._make_entry(delivered))
            self.delivered = delivered

        info = {
            "delivered": delivered,
            "hit": bool(done and self.killed),
            "timeout": bool(done and not self.killed and self.game.is_episode_timeout_reached()),
            "t": self.t,
        }
        if done:
            info["missing_fraction"] = self.n_missing / self.n_frames
        return self._get_obs(), reward, done, info

    def close(self):
        self.game.close()

    def _make_entry(self, delivered):
        state = self.game.get_state()
        heading = np.deg2rad(state.game_variables[0])
        self.n_frames += 1
        if not delivered:
            self.n_missing += 1

        detected = False
        if delivered:
            for label in state.labels:
                if label.object_name == TARGET_NAME:
                    self.last_pos = normalize_position(label.object_position_x,
                                                       label.object_position_y)
                    detected = True
                    break

        if detected:
            self.age = 0
            self.initialized = True
        elif self.initialized:
            self.age += 1   # forward-fill the position and count how old it is

        action_onehot = np.zeros(N_ACTIONS)   # all zeros before the first action
        if self.prev_action is not None:
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
        time_left = max(0.0, 1.0 - self.t / self.max_steps)
        return np.concatenate(list(self.history) + [[time_left]]).astype(np.float32)
