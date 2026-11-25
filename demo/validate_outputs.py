"""Audit captured real episodes and decode every MP4 frame and GIF frame."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import cv2
import numpy as np
from PIL import Image

from run_demo import burst_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory.resolve()
    summary = json.loads((path/"summary.json").read_text(encoding="utf-8"))
    with (path/"episodes.csv").open(encoding="utf-8", newline="") as file:
        episodes = list(csv.DictReader(file))
    assert summary["real_vizdoom"] is True and summary["trained_agent"] is False
    max_gap, invalid_leaks = 0, 0
    for episode in episodes:
        mode, index = episode["mode"], int(episode["episode"])
        rows = [json.loads(line) for line in (path/f"{mode}_{index:02d}.decisions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == int(episode["decisions"])
        assert sum(r["tics_advanced"] for r in rows) == int(episode["executed_tics"])
        assert np.isclose(sum(r["reward"] for r in rows), float(episode["return"]))
        # The unmodified scenario has +1 kill reward and -.001 per executed tic.
        assert np.isclose(float(episode["return"]), int(episode["hit"])-.001*int(episode["executed_tics"]))
        assert int(episode["actual_shots"]) == int(episode["hit"])+int(episode["missed_shots"])+int(episode["unresolved_shots_at_timeout"])
        flags = burst_mask(1000, int(episode["mask_seed"])) if mode == "burst" else [True]*1000
        assert [r["delivered"] for r in rows] == flags[:len(rows)]
        last_delivery = None
        streak = 0
        for k, row in enumerate(rows):
            assert row["decision"] == k
            if row["delivered"]:
                last_delivery = row["tic"]
                streak = 0
            else:
                streak += 1
                max_gap = max(max_gap, streak)
                assert row["target_status"] == "unavailable" and not row["agent_target_detected"]
                if k:
                    assert row["remembered_position"] == rows[k-1]["remembered_position"]
                    assert row["observed_velocity_estimate"] == rows[k-1]["observed_velocity_estimate"]
            assert row["frame_age_decisions"] == (row["tic"]-last_delivery)/4
    for index in range(summary["episodes_per_mode"]):
        pair = [r for r in episodes if int(r["episode"]) == index]
        assert len(pair) == 2 and pair[0]["initial_frame_sha256"] == pair[1]["initial_frame_sha256"]
    decoded = {}
    for name in ["clean", "burst", "comparison"]:
        video = cv2.VideoCapture(str(path/f"{name}.mp4"))
        assert video.isOpened()
        fps = video.get(cv2.CAP_PROP_FPS)
        count, dimensions = 0, None
        while True:
            ok, frame = video.read()
            if not ok:
                break
            count += 1
            dimensions = list(frame.shape[:2])
        video.release()
        if name in ("clean", "burst"):
            expected = sum(int(r["executed_tics"])+28 for r in episodes if r["mode"] == name)
        else:
            expected = sum(max(int(r["executed_tics"])+28 for r in episodes if int(r["episode"]) == index)
                           for index in summary["preview_episodes"])
        assert count == expected, (name, count, expected)
        assert dimensions == ([416, 704] if name != "comparison" else [832, 704])
        assert fps == 35
        decoded[name] = {"frames": count, "fps": fps, "seconds": count/fps, "dimensions": dimensions}
    with Image.open(path/"comparison.gif") as gif:
        assert gif.size == (704, 832) and gif.n_frames > 10
        duration = 0
        for k in range(gif.n_frames):
            gif.seek(k)
            gif.convert("RGB").load()
            duration += gif.info["duration"]
        decoded["gif"] = {"frames": gif.n_frames, "duration_seconds": duration/1000, "size": list(gif.size)}
    report = {"real_episodes_checked": len(episodes), "scenario_pairs_same_initial_pixels": True,
              "reward_tics_hit_conservation_passed": True, "no_target_memory_updates_during_outages": True,
              "max_observed_burst_decisions": max_gap, "media_decode": decoded}
    (path/"validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
