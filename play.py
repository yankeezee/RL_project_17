"""Play predict_position yourself.

Controls: left/right arrows (or mouse) to turn, Ctrl (or left click) to fire.
You only have one rocket per episode.

    python play.py --episodes 5
"""
import argparse
import os

import vizdoom as vzd

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=5)
args = parser.parse_args()

game = vzd.DoomGame()
game.load_config(os.path.join(vzd.scenarios_path, "predict_position.cfg"))
game.set_window_visible(True)
game.set_mode(vzd.Mode.ASYNC_SPECTATOR)   # the human plays in real time
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.init()

hits = 0
for ep in range(args.episodes):
    game.new_episode()
    while not game.is_episode_finished():
        game.advance_action()
    total = game.get_total_reward()
    hit = total > 0.5
    hits += hit
    print(f"episode {ep}: {'hit' if hit else 'miss'} | return {total:+.3f}")

print(f"\nhit rate: {hits}/{args.episodes}")
game.close()
