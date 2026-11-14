"""Measure raw RGB feature motion during settled holds, without stabilizing it.

Reports image motion, which can come from camera drift, FEM motion, object
motion or rendering. Use check_finger_motion.py to separate mounting and FEM.
The pixel limit applies at the recording's native resolution and frame rate.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from soft_sim.execution import settled_phase_pairs
from soft_sim.recording.reader import open_frame


def feature_motion(previous, current):
    previous = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    points = cv2.goodFeaturesToTrack(previous, maxCorners=300, qualityLevel=.01, minDistance=5,
                                     blockSize=5)
    if points is None:
        return None
    following, forward, error = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if following is None:
        return None
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, following, None)
    if returned is None:
        return None
    # Repetitive cage bars can produce false large matches. Check both
    # directions and brightness residual instead of treating them as motion.
    valid = ((forward.ravel() == 1) & (backward.ravel() == 1)
             & (np.linalg.norm((returned - points).reshape(-1, 2), axis=1) < .5)
             & (error.ravel() < 15.))
    xy = following.reshape(-1, 2)
    valid &= np.all((xy >= 0) & (xy < [current.shape[1], current.shape[0]]), axis=1)
    if valid.sum() < max(20, .7 * len(points)):
        return None
    motion = np.linalg.norm((following - points).reshape(-1, 2), axis=1)[valid]
    return dict(tracked_features=int(valid.sum()), detected_features=len(points),
                median_motion_px=float(np.median(motion)), p95_motion_px=float(np.percentile(motion, 95)))


def check(recording, maximum_p95_px=.5, settle_s=.25):
    if not np.isfinite(maximum_p95_px) or maximum_p95_px <= 0 or not np.isfinite(settle_s) or settle_s < 0:
        raise ValueError("Pixel limit must be positive and settling duration nonnegative; both must be finite")
    recording = Path(recording)
    episode = json.loads((recording / "episode.json").read_text())
    if episode["status"] != "complete":
        raise ValueError("Episode is incomplete")
    reports = {}
    for phase in ("hold", "verify_hold"):
        rows = [row for row in episode["frames"] if row["phase"] == phase]
        if not rows:
            continue
        pairs = settled_phase_pairs(episode["frames"], phase, settle_s)
        samples = {name: [] for name in ("left_rgb", "right_rgb", "external_rgb")}
        for a, b in pairs:
            with open_frame(recording, episode["frames"][a]) as frame:
                previous = {name: frame[name] for name in samples}
            row = episode["frames"][b]
            with open_frame(recording, row) as frame:
                current = {name: frame[name] for name in samples}
            for name in samples:
                result = feature_motion(previous[name], current[name])
                samples[name].append(dict(time_s=row["time_s"], measurement=result))
        reports[phase] = {}
        for name, frames in samples.items():
            measured = [row for row in frames if row["measurement"] is not None]
            worst = max(measured, key=lambda row: row["measurement"]["p95_motion_px"]) if measured else None
            complete = bool(frames) and len(measured) == len(frames)
            reports[phase][name] = dict(checked_pairs=len(frames), measured_pairs=len(measured),
                maximum_frame_p95_motion_px=worst["measurement"]["p95_motion_px"] if worst else None,
                worst_frame=worst,
                passed=bool(complete and worst["measurement"]["p95_motion_px"] <= maximum_p95_px),
                frames=frames)
    report = dict(valid=bool(reports) and all(camera["passed"] for cameras in reports.values()
                                            for camera in cameras.values()),
                  fps=episode["fps"], maximum_allowed_frame_p95_motion_px=maximum_p95_px,
                  excluded_initial_hold_s=settle_s,
                  minimum_tracked_features=20, minimum_tracked_fraction=.7,
                  scope="Raw RGB, native resolution, forward/backward checked sparse optical flow during holds; "
                        "does not attribute motion to camera vs scene and can miss motion between sampled frames",
                  phases=reports)
    (recording / "camera_stability_validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--maximum-p95-px", type=float, default=.5)
    parser.add_argument("--settle-s", type=float, default=.25)
    args = parser.parse_args()
    report = check(args.recording, args.maximum_p95_px, args.settle_s)
    summary = {**report, "phases": {phase: {name: {key: value for key, value in camera.items() if key != "frames"}
                                              for name, camera in cameras.items()}
                                   for phase, cameras in report["phases"].items()}}
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if report["valid"] else 1)
