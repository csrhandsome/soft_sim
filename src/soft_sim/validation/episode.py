"""Check RGB/state integrity and require the independent sensor/physics audits."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from soft_sim.recording.reader import open_frame, _dataset


def validate(folder):
    folder = Path(folder)
    episode = json.loads((folder / "episode.json").read_text())
    assert episode["status"] in ("committed", "complete"), "Episode is incomplete"
    rows = episode["frames"]
    assert len(rows) == episode["frame_count"] > 0
    np.testing.assert_allclose(np.diff([row["time_s"] for row in rows]),
                               1 / episode["fps"], atol=1e-7, rtol=0)
    minima = {name: float("inf") for name in episode["camera_order"]}
    for index, row in enumerate(rows):
        assert row["index"] == index
        with open_frame(folder, row) as frame:
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
            joint_count = len(episode["calibration"].get("joint_names", range(9)))
            assert frame["joint_positions"].shape == (joint_count,)
            assert frame["finger_camera_to_world"].shape == (2, 4, 4)
    videos = {}
    if episode.get("storage_format") == "lerobot_v3":
        dataset, _ = _dataset(str(folder.resolve()))
        assert dataset.meta.info["codebase_version"] == "v3.0"
        entry = dataset.meta.episodes[episode["episode_index"]]
        assert int(entry["length"]) == len(rows)
        video_paths = {name: dataset.root / dataset.meta.get_video_file_path(
            episode["episode_index"], "observation.images." + name) for name in episode["camera_order"]}
    else:
        video_paths = {name: folder / (name + ".mp4") for name in (*episode["camera_order"], "episode")}
    for name, path in video_paths.items():
        video = cv2.VideoCapture(str(path))
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
        # v3 shards may contain several episodes. Every image of this episode
        # was already decoded at its metadata timestamp by open_frame above.
        assert decoded >= len(rows) if episode.get("storage_format") == "lerobot_v3" else decoded == len(rows)
        videos[name] = dict(frames=decoded, fps=fps, resolution=size)
    report = dict(valid=True, data_integrity_valid=True,
                  camera_projection_valid=None,
                  camera_motion_sync_valid=None,
                  camera_fem_projection_valid=None,
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
    fem_projection_report = folder / "fem_camera_projection_validation.json"
    if fem_projection_report.is_file():
        fem_projection = json.loads(fem_projection_report.read_text())
        report["camera_fem_projection_valid"] = fem_projection["valid"]
        report["valid"] = report["valid"] and fem_projection["valid"]
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
    required = ("camera_projection_valid", "camera_motion_sync_valid", "camera_fem_projection_valid",
                "camera_mount_valid", "clamp_valid", "physics_initialization_valid",
                "hold_projected_motion_valid", "camera_rgb_hold_stability_valid")
    report["missing_quality_checks"] = [name for name in required if report[name] is None]
    report["quality_checks_complete"] = not report["missing_quality_checks"]
    report["valid"] = bool(report["valid"] and all(report[name] is True for name in required))
    (folder / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="OUTPUT/recording directory")
    args = parser.parse_args()
    report = validate(args.folder)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["valid"] else 1)
