"""Compare recorded finger depth to the same frame's deformed FEM boundary.

Uses independent CPU ray intersections with volume boundary triangles, rather
than the renderer's STL embedding or an undeformed reference cage. The near
field avoids object/background occlusion; it does not certify those objects'
rendering or the external camera.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from soft_sim.recording.reader import open_frame
from soft_sim.mesh import surface_faces
from soft_sim.validation.collisions import finger_reference_for_arm, resolve_operation_arm


def projected_depth(vertices, faces, camera_pose, intrinsics, pixels):
    rays = np.c_[(pixels[:, 0]-intrinsics[0, 2])/intrinsics[0, 0],
                 (pixels[:, 1]-intrinsics[1, 2])/intrinsics[1, 1], np.ones(len(pixels))]
    rays = rays@camera_pose[:3, :3].T
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    hits, indices, _ = mesh.ray.intersects_location(
        np.repeat(camera_pose[None, :3, 3], len(pixels), axis=0), rays, multiple_hits=False)
    depth = np.full(len(pixels), np.nan)
    if len(indices):
        depth[indices] = ((hits-camera_pose[:3, 3])@camera_pose[:3, :3])[:, 2]
    return depth


def check(folder):
    folder = Path(folder)
    episode = json.loads((folder / "episode.json").read_text())
    if episode["status"] not in ("committed", "complete"):
        raise ValueError("A complete recording is required")
    width, height = episode["calibration"]["finger_resolution"]
    intrinsics = np.asarray(episode["calibration"]["finger_intrinsics"])
    x, y = np.meshgrid(np.arange(8, width-8, 8), np.arange(8, height-8, 8))
    pixels = np.c_[x.ravel(), y.ravel()]
    plans = (episode.get("result") or {}).get("executed_plans") or []
    faces_by_arm = {}
    results = []
    with np.load(folder / "finger_reference.npz", allow_pickle=False) as reference:
        for row in episode["frames"]:
            sides = []
            with open_frame(folder, row, include_images=False) as frame:
                op_index = int(np.asarray(frame["operation_index"]).reshape(-1)[0]) if "operation_index" in frame else 0
                op_arm = resolve_operation_arm(episode, plans, op_index)
                if op_arm not in faces_by_arm:
                    faces_by_arm[op_arm] = [
                        surface_faces(finger_reference_for_arm(reference, op_arm, name)[2])
                        for name in ("left", "right")
                    ]
                faces = faces_by_arm[op_arm]
                for side, name in enumerate(("left", "right")):
                    expected = projected_depth(frame[f"finger_nodes_{name}"], faces[side],
                        frame["finger_camera_to_world"][side], intrinsics[side], pixels)
                    near = np.isfinite(expected) & (expected > .001) & (expected < .02)
                    actual = frame["finger_depth_m"][side, pixels[near, 1], pixels[near, 0]]
                    errors = np.abs(actual-expected[near])
                    agreement = float(np.mean(np.isfinite(actual) & (errors < .002))) if len(actual) else 0.
                    sides.append(dict(camera=name, rays=int(near.sum()), fraction_within_2mm=agreement,
                        median_depth_error_mm=float(np.nanmedian(errors)*1000) if np.isfinite(errors).any() else None,
                        passed=bool(near.sum() >= 50 and agreement > .85)))
            results.append(dict(index=row["index"], time_s=row["time_s"], phase=row["phase"], cameras=sides))
    report = dict(valid=bool(results) and all(side["passed"] for row in results for side in row["cameras"]),
        frame_count=len(results), maximum_error_m=.002, minimum_agreement=.85,
        maximum_optical_depth_m=.02, minimum_rays=50,
        scope="All recorded frames; near-field finger depth versus independent deformed volume boundary rays. Does not validate object/background rendering, RGB stability, or camera attachment to the robot.",
        frames=results)
    (folder / "fem_camera_projection_validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    report = check(args.recording)
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    raise SystemExit(0 if report["valid"] else 1)
