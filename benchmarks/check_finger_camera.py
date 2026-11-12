"""Check initial projection and pre-contact motion against source-STL rays.

This catches renderer projection errors that a non-black-image check misses.
Run on a recorded episode whose first frame is the undeformed settle phase.
For planned grasps, also check the near-base cage during pregrasp movement.
This catches stale rendered geometry paired with an updated camera transform.
Do not compare contact-loaded fingers against an undeformed STL.
No Kit or GPU is required.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.config import CameraConfig
from shared.geometry import camera_intrinsics, mounted_camera_pose
from shared.mesh import load_stl


def check(folder):
    folder = Path(folder)
    recording = folder / "recording"
    result = json.loads((folder / "result.json").read_text())
    episode = json.loads((recording / "episode.json").read_text())
    first = episode["frames"][0]
    if first["phase"] != "settle" or first["time_s"] > .1:
        raise ValueError("Projection check needs an initial settle frame at t <= 0.1 s")
    config = result["config"]
    camera = CameraConfig(**config["camera"])
    mesh_config = config["mesh"]
    vertices, faces = load_stl(ROOT / mesh_config["stl"], mesh_config["scale_to_m"])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    local_camera = mounted_camera_pose(camera, np.zeros(3), np.eye(3))
    x, y = np.meshgrid(np.arange(8, camera.width - 8, 8), np.arange(8, camera.height - 8, 8))
    pixels = np.column_stack([x.ravel(), y.ravel()])
    reports, motion_reports = [], []
    motion_rows = [row for row in episode["frames"] if row["phase"] == "pregrasp"]
    with np.load(recording / first["file"], allow_pickle=False) as frame:
        for side, name in enumerate(("left_rgb", "right_rgb")):
            intrinsics = np.asarray(episode["calibration"]["finger_intrinsics"])[side]
            # Also validate serialized calibration against the physical lens.
            projection = episode["calibration"].get("finger_projection")
            if projection:
                np.testing.assert_allclose(intrinsics, camera_intrinsics(camera, projection[side]["physical_focal_length_mm"]))
            optical_rays = np.column_stack([(pixels[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0],
                                             (pixels[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1],
                                             np.ones(len(pixels))])
            rays = optical_rays @ local_camera[:3, :3].T
            origins = np.tile(local_camera[:3, 3], (len(rays), 1))
            hits, indices, _ = mesh.ray.intersects_location(origins, rays, multiple_hits=False)
            expected = ((hits - local_camera[:3, 3]) @ local_camera[:3, :3])[:, 2]
            indices = indices[expected > camera.near_m]
            expected = expected[expected > camera.near_m]
            actual = frame["finger_depth_m"][side, pixels[indices, 1], pixels[indices, 0]]
            errors = np.abs(actual - expected)
            agreement = float(np.mean(np.isfinite(actual) & (errors < .002)))
            rgb = frame[name].astype(float)
            center = rgb[camera.height // 4:3 * camera.height // 4, camera.width // 4:3 * camera.width // 4]
            spatial_std = float(center.std(axis=(0, 1)).mean())
            reports.append(dict(camera=name, sampled_mesh_hits=len(expected),
                                fraction_within_2mm_of_stl=agreement,
                                median_depth_error_mm=float(np.nanmedian(errors) * 1000),
                                center_spatial_rgb_std=spatial_std,
                                passed=agreement > .85 and spatial_std > 5.))
            if motion_rows:
                # The near-base cage changes little during the unloaded motion;
                # distal bars can bend under gravity and are excluded. The 2 mm
                # tolerance allows elastic motion without accepting the several
                # millimeter alternating depth errors from stale Fabric state.
                near = expected < .04
                if near.sum() < 50:
                    raise ValueError("Too few near-base STL rays for the motion check")
                samples = []
                for row in motion_rows:
                    with np.load(recording / row["file"], allow_pickle=False) as moving:
                        actual = moving["finger_depth_m"][side, pixels[indices[near], 1], pixels[indices[near], 0]]
                    error = np.abs(actual - expected[near])
                    samples.append(dict(time_s=row["time_s"],
                        fraction_within_2mm=float(np.mean(np.isfinite(actual) & (error < .002))),
                        median_depth_error_mm=float(np.nanmedian(error) * 1000)))
                worst = min(samples, key=lambda row: row["fraction_within_2mm"])
                motion_reports.append(dict(camera=name, checked_frames=len(samples),
                    sampled_mesh_hits=int(near.sum()), worst_frame=worst,
                    maximum_median_depth_error_mm=max(row["median_depth_error_mm"] for row in samples),
                    passed=all(row["fraction_within_2mm"] > .85 for row in samples)))
    initial_valid = all(row["passed"] for row in reports)
    motion_valid = all(row["passed"] for row in motion_reports) if motion_reports else None
    report = dict(valid=initial_valid and motion_valid is not False,
                  initial_projection_valid=initial_valid, motion_sync_valid=motion_valid,
                  time_s=first["time_s"],
                  check="Rendered cage depth vs independent source-STL rays; center spatial RGB variation",
                  scope="Initial projection and near-base unloaded pregrasp motion; excludes contact-loaded deformation",
                  cameras=reports, pre_contact_motion=motion_reports)
    (recording / "camera_projection_validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    report = check(args.episode)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["valid"] else 1)
