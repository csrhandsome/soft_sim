"""Check synchronized raw RGB/state frames and decode all four episode MP4s."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def validate(folder):
    folder = Path(folder)
    episode = json.loads((folder / "episode.json").read_text())
    assert episode["status"] == "complete", "Episode is incomplete"
    rows = episode["frames"]
    assert len(rows) == episode["frame_count"] > 0
    np.testing.assert_allclose(np.diff([row["time_s"] for row in rows]),
                               1 / episode["fps"], atol=1e-7, rtol=0)
    minima = {name: float("inf") for name in episode["camera_order"]}
    for index, row in enumerate(rows):
        assert row["index"] == index
        with np.load(folder / row["file"], allow_pickle=False) as frame:
            assert abs(float(frame["time_s"]) - row["time_s"]) < 1e-8
            for name in minima:
                rgb = frame[name]
                resolution = episode["calibration"]["external_resolution" if name == "external_rgb" else "finger_resolution"]
                assert rgb.shape == (resolution[1], resolution[0], 3)
                assert rgb.dtype == np.uint8 and rgb.std() > 1.
                minima[name] = min(minima[name], float(rgb.std()))
            for name in ("joint_positions", "joint_velocities", "commanded_joint_positions",
                         "object_position", "object_orientation_wxyz", "finger_camera_to_world"):
                assert np.isfinite(frame[name]).all(), f"Non-finite state: {name}"
            assert frame["joint_positions"].shape == (9,)
            assert frame["finger_camera_to_world"].shape == (2, 4, 4)
    videos = {}
    for name in (*episode["camera_order"], "episode"):
        video = cv2.VideoCapture(str(folder / (name + ".mp4")))
        assert video.isOpened(), f"Cannot open {name} MP4"
        fps = video.get(cv2.CAP_PROP_FPS)
        assert abs(fps - episode["fps"]) < 1e-6
        size = [int(video.get(key)) for key in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
        decoded = 0
        while True:
            ok, image = video.read()
            if not ok:
                break
            assert image.shape[:2] == (size[1], size[0])
            decoded += 1
        video.release()
        assert decoded == len(rows), f"{name}: decoded {decoded}, expected {len(rows)}"
        videos[name] = dict(frames=decoded, fps=fps, resolution=size)
    report = dict(valid=True, data_integrity_valid=True,
                  camera_projection_valid=None,
                  camera_motion_sync_valid=None,
                  camera_mount_valid=None,
                  clamp_valid=None,
                  physics_initialization_valid=None,
                  hold_projected_motion_valid=None,
                  camera_rgb_hold_stability_valid=None,
                  frame_count=len(rows), duration_s=episode["duration_s"],
                  phases=list(dict.fromkeys(row["phase"] for row in rows)),
                  minimum_rgb_std=minima, videos=videos,
                  grasp_success=episode["result"]["success"])
    projection_report = folder / "camera_projection_validation.json"
    if projection_report.is_file():
        camera_checks = json.loads(projection_report.read_text())
        report["camera_projection_valid"] = camera_checks.get("initial_projection_valid", camera_checks["valid"])
        report["camera_motion_sync_valid"] = camera_checks.get("motion_sync_valid")
        report["valid"] = camera_checks["valid"]
    motion_report = folder / "finger_motion_validation.json"
    if motion_report.is_file():
        motion_checks = json.loads(motion_report.read_text())
        report["camera_mount_valid"] = motion_checks["camera_mount_valid"]
        report["clamp_valid"] = motion_checks.get("clamp_valid")
        report["physics_initialization_valid"] = motion_checks.get("physics_initialization_valid")
        report["hold_projected_motion_valid"] = motion_checks.get("hold_projected_motion_valid")
        report["valid"] = report["valid"] and motion_checks.get("valid", report["camera_mount_valid"])
    stability_report = folder / "camera_stability_validation.json"
    if stability_report.is_file():
        stability_checks = json.loads(stability_report.read_text())
        report["camera_rgb_hold_stability_valid"] = stability_checks["valid"]
        report["valid"] = report["valid"] and stability_checks["valid"]
    (folder / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="OUTPUT/recording directory")
    args = parser.parse_args()
    report = validate(args.folder)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["valid"] else 1)
