"""Check the actual installed engine before constructing the demo."""
import sys
from pathlib import Path
import json
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
from engine_runtime import engine_runtime
from PIL import Image

runtime = engine_runtime()
vzd = runtime.__enter__()
print("ViZDoom", vzd.__version__, flush=True)
print("package", vzd.__file__, flush=True)
cfg = Path(vzd.scenarios_path) / "predict_position.cfg"
print(cfg.read_text(), flush=True)
game = vzd.DoomGame()
try:
    game.load_config(str(cfg))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.set_seed(20261125)
    game.init()
    print("ENGINE INITIALIZED", flush=True)
    state = game.get_state()
    Image.fromarray(state.screen_buffer).save(ROOT / "output" / "probe.png")
    report = {"version": vzd.__version__, "buttons": [str(b) for b in game.get_available_buttons()],
              "timeout": game.get_episode_timeout(), "tic": game.get_episode_time(),
              "screen_shape": list(state.screen_buffer.shape),
              "labels": [{"name": l.object_name, "x": l.x, "y": l.y, "w": l.width,
                          "h": l.height, "wx": l.object_position_x, "wy": l.object_position_y}
                         for l in state.labels],
              "variables": {name: game.get_game_variable(getattr(vzd.GameVariable, name))
                            for name in ["KILLCOUNT", "SELECTED_WEAPON_AMMO", "ANGLE", "POSITION_X", "POSITION_Y"]}}
    for _ in range(3):
        game.make_action([1, 0, 0], 4)
    game.make_action([0, 0, 1], 4)
    while not game.is_episode_finished():
        game.make_action([0, 0, 0], 4)
    report["finished"] = {"return": game.get_total_reward(), "tic": game.get_episode_time(),
                          "kills": game.get_game_variable(vzd.GameVariable.KILLCOUNT),
                          "ammo": game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO),
                          "timeout": game.is_episode_timeout_reached()}
    (ROOT / "output" / "probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
finally:
    game.close()
    runtime.__exit__(None, None, None)
