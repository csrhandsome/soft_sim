"""Measure repeated renders at one physical state, without advancing simulation."""
import json
from pathlib import Path

import numpy as np

from soft_sim.validation.camera_stability import feature_motion


def probe(capture, read_state, directory, count=12, maximum_p95_px=.5):
    """Callbacks return RGB camera dictionaries and copied physical-state arrays."""
    if count < 2:
        raise ValueError("Need at least two frozen renders")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    initial = read_state()
    previous = None
    measurements = {}
    state_delta = {name: 0. for name in initial}
    for index in range(count):
        images = capture()
        state = read_state()
        for name, value in initial.items():
            state_delta[name] = max(state_delta[name], float(np.max(np.abs(state[name] - value))))
        np.savez_compressed(directory / f"frame_{index:03d}.npz", **images, **state)
        if previous is not None:
            for name, current in images.items():
                measurements.setdefault(name, []).append(feature_motion(previous[name], current))
        previous = images
    cameras = {}
    for name, rows in measurements.items():
        measured = [row for row in rows if row is not None]
        worst = max((row["p95_motion_px"] for row in measured), default=None)
        cameras[name] = dict(measured_pairs=len(measured), checked_pairs=len(rows),
                             maximum_frame_p95_motion_px=worst,
                             passed=len(measured) == len(rows) and worst is not None and worst <= maximum_p95_px)
    # State arrays include world time, nodes, mounts, camera poses, arm and object.
    unchanged = all(value == 0. for value in state_delta.values())
    report = dict(physical_state_unchanged=unchanged, maximum_state_delta=state_delta,
                  maximum_allowed_frame_p95_motion_px=maximum_p95_px, cameras=cameras,
                  valid=unchanged and bool(cameras) and all(row["passed"] for row in cameras.values()))
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    return report
