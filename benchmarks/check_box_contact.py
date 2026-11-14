"""Check recorded collision vertices against an independently evaluated rigid box.

Vertex penetration is a lower bound on mesh intersection: this check can reject
deep penetration but cannot rule out intersections confined to edges or faces.
"""
import argparse
import json
from pathlib import Path

import numpy as np
try:
    from benchmarks.episode_reader import open_frame
except ModuleNotFoundError:
    from episode_reader import open_frame
from scipy.spatial.transform import Rotation


def box_signed_distances(points, center, orientation_wxyz, size):
    points, center, quat, size = [np.asarray(value, dtype=float)
                                  for value in (points, center, orientation_wxyz, size)]
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or center.shape != (3,) or quat.shape != (4,) or size.shape != (3,)
            or not all(np.isfinite(value).all() for value in (points, center, quat, size))
            or np.any(size <= 0) or np.linalg.norm(quat) == 0):
        raise ValueError("Expected finite collision points and a rigid box pose with positive dimensions")
    rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()
    local = (points - center) @ rotation
    distance = np.abs(local) - size / 2
    return np.linalg.norm(np.maximum(distance, 0), axis=1) + np.minimum(np.max(distance, axis=1), 0)


def check(recording, maximum_penetration_mm=.5):
    if not np.isfinite(maximum_penetration_mm) or maximum_penetration_mm <= 0:
        raise ValueError("Penetration limit must be positive and finite")
    recording = Path(recording)
    episode = json.loads((recording / "episode.json").read_text())
    if episode["status"] != "complete":
        raise ValueError("Episode is incomplete")
    result = episode.get("result") or {}
    if result.get("object_asset") != "primitive:box" or not result.get("object_size_m"):
        raise ValueError("This diagnostic requires the recorded analytic rigid box dimensions")
    rows = []
    for row in episode["frames"]:
        with open_frame(recording, row) as frame:
            distances = [box_signed_distances(frame["finger_collision_nodes_" + side],
                frame["object_position"], frame["object_orientation_wxyz"], result["object_size_m"])
                for side in ("left", "right")]
        rows.append(dict(time_s=row["time_s"], phase=row["phase"],
            maximum_vertex_penetration_mm=[float(np.maximum(-d, 0).max() * 1000) for d in distances],
            minimum_signed_vertex_distance_mm=[float(d.min() * 1000) for d in distances]))
    worst = np.max([row["maximum_vertex_penetration_mm"] for row in rows], axis=0).tolist() if rows else None
    report = dict(valid=bool(rows) and max(worst) <= maximum_penetration_mm,
        maximum_allowed_vertex_penetration_mm=maximum_penetration_mm,
        maximum_vertex_penetration_mm=worst, box_size_m=result["object_size_m"], frame_count=len(rows),
        scope=__doc__.strip(), frames=rows)
    (recording / "box_contact_validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--maximum-penetration-mm", type=float, default=.5)
    args = parser.parse_args()
    report = check(args.recording, args.maximum_penetration_mm)
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, indent=2))
    raise SystemExit(0 if report["valid"] else 1)
