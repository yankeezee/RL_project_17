"""Real ViZDoom predict_position; untrained visual-label heuristic; no learning."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from engine_runtime import ROOT, engine_runtime

sys.stdout.reconfigure(encoding="utf-8")
TICS_PER_DECISION = 4
FPS = 35
ACTIONS = {"wait": [0, 0, 0], "left": [1, 0, 0], "right": [0, 1, 0], "fire": [0, 0, 1]}
TARGET = "Cacodemon"
WIDTH, HEIGHT = 704, 416
BG, INK, MUTED = "#111a28", "#edf3fa", "#96a9be"
GREEN, RED, BLUE = "#51d9a4", "#ff877b", "#77bafa"


def font(size, bold=False):
    path = Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


FONTS = {s: font(s, s >= 18) for s in [12, 14, 16, 18, 21]}


def burst_mask(length, seed, p=.25, mean_length=4):
    """Pre-generated, action-independent Markov channel; starts available."""
    rng = np.random.default_rng(seed)
    missing = False
    result = []
    for _ in range(length):
        result.append(not missing)
        missing = rng.random() >= 1/mean_length if missing else rng.random() < p/(mean_length*(1-p))
    return result


def visible_target(state):
    return next((label for label in state.labels if label.object_name == TARGET), None)


class Heuristic:
    """No training; constant lead from two delivered position measurements only.

    Own fixed spawn position and heading are proprioception. No engine target
    velocity, hidden labels, truth image or outcome variables enter this policy.
    """
    def __init__(self, own_position):
        self.own_position = np.asarray(own_position, dtype=float)
        self.position = None
        self.velocity_estimate = np.zeros(2)
        self.last_detection_tic = None
        self.last_frame_tic = None
        self.delivered_frame = None
        self.box = None
        self.fired = False

    def observe(self, *, delivered, frame, label, tic):
        self.box = None
        if delivered:
            self.last_frame_tic = tic
            self.delivered_frame = frame.copy()
            if label is not None:
                measured = np.array([label.object_position_x, label.object_position_y])
                if self.position is not None and tic > self.last_detection_tic:
                    self.velocity_estimate = (measured-self.position)/(tic-self.last_detection_tic)
                self.position, self.last_detection_tic = measured, tic
                self.box = (int(label.x), int(label.y), int(label.width), int(label.height))

    def choose(self, *, heading, tic, decision):
        if self.fired:
            return "wait"
        if self.position is None:
            return "left"
        predicted = self.position + self.velocity_estimate * (tic-self.last_detection_tic+24)
        vector = predicted-self.own_position
        bearing = math.degrees(math.atan2(vector[1], vector[0]))
        error = (bearing-heading+180) % 360 - 180
        if abs(error) > 10:
            return "left" if error > 0 else "right"
        if decision >= 16:  # fixed observation period; chosen before the demo runs
            self.fired = True
            return "fire"
        return "wait"


def dashboard(*, truth, policy, mode, episode, seed, tic, decision, delivered,
              detected, action, shots, outcome=None):
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    mode_title = "CLEAN OBSERVATIONS" if mode == "clean" else "BURST OUTAGES | p=25%, mean L=4 decisions"
    draw.text((24, 10), mode_title, fill=INK, font=FONTS[18])
    draw.text((24, 39), f"UNTRAINED HEURISTIC | episode {episode+1} | scenario seed {seed}", fill=MUTED, font=FONTS[12])
    draw.text((24, 59), "ENVIRONMENT / observer only", fill=BLUE, font=FONTS[14])
    draw.text((360, 59), "AGENT / delivered information", fill=GREEN, font=FONTS[14])
    canvas.paste(Image.fromarray(truth), (24, 80))
    if delivered:
        canvas.paste(Image.fromarray(policy.delivered_frame), (360, 80))
        if policy.box:
            x, y, w, h = policy.box
            draw.rectangle((360+x, 80+y, 360+x+w, 80+y+h), outline=GREEN, width=2)
    else:
        draw.rectangle((360, 80, 679, 319), fill="#241e2b")
        draw.text((445, 157), "OUTAGE", fill=RED, font=FONTS[21])
        draw.text((386, 193), "No new image or target labels", fill=MUTED, font=FONTS[14])
        draw.text((389, 216), "Only the previous position is retained", fill=MUTED, font=FONTS[12])
    age = max(0, tic-policy.last_frame_tic) if policy.last_frame_tic is not None else 0
    target_age = tic-policy.last_detection_tic if policy.last_detection_tic is not None else None
    status = "YES" if delivered else "NO - injected outage"
    detection = "YES" if detected else ("NO - target not visible" if delivered else "UNKNOWN - no frame")
    draw.text((24, 329), f"Action: {action.upper()}  |  actual rockets fired: {shots}", fill=INK, font=FONTS[14])
    draw.text((24, 353), f"Decision {decision:02d} | game tic {tic:03d}", fill=MUTED, font=FONTS[14])
    draw.text((360, 329), f"Frame delivered: {status}", fill=GREEN if delivered else RED, font=FONTS[14])
    draw.text((360, 352), f"Target detected: {detection}", fill=INK, font=FONTS[12])
    draw.text((360, 373), f"Frame age: {age/4:g} decisions / {age/35:.2f}s", fill=MUTED, font=FONTS[12])
    target_text = "not initialized" if target_age is None else f"age {target_age/4:g} decisions"
    draw.text((24, 378), f"Remembered target position: {target_text}", fill=MUTED, font=FONTS[12])
    if outcome:
        draw.rounded_rectangle((37, 269, 331, 308), radius=7, fill="#111a28")
        draw.text((48, 276), f"RESULT: {outcome.upper()} | last valid frame", fill=GREEN if outcome == "hit" else RED, font=FONTS[14])
    return np.array(canvas)


def run_episode(game, vzd, *, mode, episode, scenario_seed, mask_seed, output, video):
    game.set_seed(scenario_seed)
    game.new_episode()
    state = game.get_state()
    first_hash = hashlib.sha256(state.screen_buffer.tobytes()).hexdigest()
    own = [game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(vzd.GameVariable.POSITION_Y)]
    policy = Heuristic(own)
    masks = burst_mask(1000, mask_seed) if mode == "burst" else [True]*1000
    initial_ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    shots = decision = tics = missing = delivered_without_target = 0
    rows, frames = [], []
    start_tic = int(state.tic)
    final_image = None
    while not game.is_episode_finished():
        state = game.get_state()
        truth = state.screen_buffer.copy()
        tic = int(state.tic)
        delivered = bool(masks[decision])
        truth_label = visible_target(state)
        label = truth_label if delivered else None
        detected = label is not None
        # This boundary gates all current target/pixel input to the heuristic.
        policy.observe(delivered=delivered, frame=truth if delivered else None, label=label, tic=tic)
        heading = game.get_game_variable(vzd.GameVariable.ANGLE)
        action = policy.choose(heading=heading, tic=tic, decision=decision)
        row = {"mode": mode, "episode": episode, "scenario_seed": scenario_seed, "mask_seed": mask_seed,
               "decision": decision, "tic": tic, "delivered": delivered,
               "agent_target_detected": detected, "target_status": "detected" if detected else
               ("not_visible" if delivered else "unavailable"),
               "frame_age_decisions": (tic-policy.last_frame_tic)/4,
               "target_age_decisions": (tic-policy.last_detection_tic)/4 if policy.last_detection_tic is not None else None,
               "remembered_position": policy.position.tolist() if policy.position is not None else None,
               "observed_velocity_estimate": policy.velocity_estimate.tolist(), "heading": heading,
               "action": action, "truth_target_visible_observer_only": truth_label is not None}
        missing += not delivered
        delivered_without_target += delivered and not detected
        game.set_action(ACTIONS[action])
        advanced, reward = 0, 0.
        for _ in range(TICS_PER_DECISION):
            # Same action held four tics; per-tic rendering does not re-observe or re-decide.
            game.advance_action(1)
            advanced += 1
            tics += 1
            reward += game.get_last_reward()
            shots = int(round(initial_ammo-game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)))
            next_state = game.get_state()
            if next_state is not None:
                truth, display_tic = next_state.screen_buffer.copy(), int(next_state.tic)
            else:
                display_tic = tic+advanced
            final_image = dashboard(truth=truth, policy=policy, mode=mode, episode=episode,
                                    seed=scenario_seed, tic=display_tic, decision=decision,
                                    delivered=delivered, detected=detected, action=action, shots=shots)
            video.append_data(final_image)
            if episode < 2:
                frames.append(final_image)
            if not delivered and not (output / f"{mode}_outage.png").exists():
                Image.fromarray(final_image).save(output / f"{mode}_outage.png")
            if game.is_episode_finished():
                break
        row.update(tics_advanced=advanced, reward=reward, shots_so_far=shots)
        rows.append(row)
        decision += 1
    hit = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT) > 0)
    timeout = bool(game.is_episode_timeout_reached())
    outcome = "hit" if hit else "timeout" if timeout else "miss" if shots else "other_terminal"
    missed_shots = max(0, shots-hit) if not timeout else 0
    unresolved = max(0, shots-hit) if timeout else 0
    result = {"mode": mode, "episode": episode, "scenario_seed": scenario_seed, "mask_seed": mask_seed,
              "policy": "untrained_heuristic", "hit": hit, "missed_shots": missed_shots,
              "timeout": int(timeout), "unresolved_shots_at_timeout": unresolved,
              "actual_shots": shots, "outcome": outcome, "return": game.get_total_reward(),
              "decisions": decision, "executed_tics": tics, "start_tic": start_tic,
              "missing_decisions": missing, "realized_missing_fraction": missing/decision,
              "delivered_without_target_decisions": delivered_without_target,
              "initial_frame_sha256": first_hash}
    (output / f"{mode}_{episode:02d}.decisions.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False)+"\n" for r in rows), encoding="utf-8")
    end_frame = dashboard(truth=truth, policy=policy, mode=mode, episode=episode,
                          seed=scenario_seed, tic=start_tic+tics, decision=decision-1,
                          delivered=delivered, detected=detected, action=action, shots=shots, outcome=outcome)
    for _ in range(28):
        video.append_data(end_frame)
        if episode < 2:
            frames.append(end_frame)
    Image.fromarray(end_frame).save(output / f"{mode}_{episode:02d}.result.png")
    print(json.dumps(result), flush=True)
    return result, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "run1")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261125)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    with engine_runtime() as vzd:
        import imageio.v2 as imageio
        import imageio_ffmpeg
        config = Path(vzd.scenarios_path)/"predict_position.cfg"
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [config, config.with_suffix(".wad"), Path(vzd.install_path)/"freedoom2.wad"]}
        (output / "predict_position.cfg").write_bytes(config.read_bytes())
        game = vzd.DoomGame()
        records, previews = [], {}
        try:
            game.load_config(str(config))
            game.set_window_visible(False)
            game.set_sound_enabled(False)
            game.set_screen_format(vzd.ScreenFormat.RGB24)
            game.set_labels_buffer_enabled(True)
            game.set_seed(args.seed)
            game.init()
            print("REAL VIZDOOM ENGINE INITIALIZED", flush=True)
            expected_buttons = [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK]
            assert game.get_available_buttons() == expected_buttons
            for mode in ["clean", "burst"]:
                with imageio.get_writer(str(output/f"{mode}.mp4"), fps=FPS, codec="libx264", quality=7,
                                        macro_block_size=1, ffmpeg_log_level="error") as video:
                    for episode in range(args.episodes):
                        result, frames = run_episode(game, vzd, mode=mode, episode=episode,
                            scenario_seed=args.seed+episode, mask_seed=args.seed+10000+episode,
                            output=output, video=video)
                        records.append(result)
                        if frames:
                            previews[mode, episode] = frames
            with imageio.get_writer(str(output/"comparison.mp4"), fps=FPS, codec="libx264", quality=7,
                                    macro_block_size=1, ffmpeg_log_level="error") as video:
                gif_frames = []
                for episode in range(min(2, args.episodes)):
                    clean, burst = previews["clean", episode], previews["burst", episode]
                    for k in range(max(len(clean), len(burst))):
                        combined = np.concatenate([clean[min(k, len(clean)-1)], burst[min(k, len(burst)-1)]], axis=0)
                        video.append_data(combined)
                        if k % 4 == 0:
                            gif_frames.append(Image.fromarray(combined))
                gif_frames[0].save(output/"comparison.gif", save_all=True, append_images=gif_frames[1:],
                                   duration=114, loop=0, optimize=False)
        finally:
            game.close()
        for e in range(args.episodes):
            pair = [r for r in records if r["episode"] == e]
            assert pair[0]["initial_frame_sha256"] == pair[1]["initial_frame_sha256"], "scenario seed pairing failed"
        with (output/"episodes.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        summary = {"real_vizdoom": True, "trained_agent": False, "scenario": "predict_position",
                   "vizdoom": vzd.__version__, "numpy": np.__version__, "python": sys.version,
                   "platform": platform.platform(), "imageio_ffmpeg": imageio_ffmpeg.__version__,
                   "seed": args.seed, "episodes_per_mode": args.episodes,
                   "burst": {"p": .25, "L_decisions": 4, "recovery": .25, "onset": 1/12,
                             "episode_start_available": True, "masks_pregenerated_independent_of_actions": True},
                   "action_hold_tics": 4, "engine_fps": 35, "timeout_tics": game.get_episode_timeout(),
                   "asset_sha256": hashes, "preview_episodes": [0, 1][:args.episodes],
                   "policy": {"type": "untrained heuristic", "lead_tics": 24, "aim_tolerance_deg": 10,
                              "minimum_fire_decision": 16, "learned_parameters": False,
                              "inputs": "delivered visible-target positions; previous delivered measurements; own heading and fixed spawn position",
                              "does_not_read": "hidden target positions, engine target velocity, observer truth image during outage, outcome counters"},
                   "results": {}}
        for mode in ["clean", "burst"]:
            rows = [r for r in records if r["mode"] == mode]
            summary["results"][mode] = {"episodes": len(rows), "hits": sum(r["hit"] for r in rows),
                "missed_shots": sum(r["missed_shots"] for r in rows), "timeouts": sum(r["timeout"] for r in rows),
                "shots": sum(r["actual_shots"] for r in rows), "outcomes": dict(Counter(r["outcome"] for r in rows)),
                "missing_decisions": sum(r["missing_decisions"] for r in rows),
                "decisions": sum(r["decisions"] for r in rows),
                "delivered_without_target_decisions": sum(r["delivered_without_target_decisions"] for r in rows),
                "mean_return": float(np.mean([r["return"] for r in rows]))}
        summary["wall_seconds"] = perf_counter()-started
        (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
