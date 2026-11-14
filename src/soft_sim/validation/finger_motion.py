"""Measure full-episode FEM motion separately from the rigid camera mount.

Requires recorded cooked FEM nodes and finger_reference.npz. Displacements
are measured in each moving finger base, so arm travel and jaw closure do not
count as elastic deformation. Physical vibration is reported, not filtered.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from soft_sim.execution import settled_phase_pairs
from soft_sim.recording.reader import open_frame


def project_nodes(points, camera_to_world, intrinsics):
    """Project physical nodes in the optical camera; no image stabilization."""
    optical = (points - camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    pixels = optical @ intrinsics.T
    in_front = optical[:, 2] > 1e-6
    uv = np.full((len(points), 2), np.nan)
    uv[in_front] = pixels[in_front, :2] / pixels[in_front, 2:3]
    return uv


def check(recording, maximum_unloaded_mm=2., maximum_hold_p95_px=.5, maximum_pin_error_mm=.01):
    if not all(np.isfinite(value) and value > 0 for value in (maximum_unloaded_mm, maximum_hold_p95_px, maximum_pin_error_mm)):
        raise ValueError("Motion limits must be finite and positive")
    recording = Path(recording)
    episode = json.loads((recording / "episode.json").read_text())
    if episode["status"] != "complete":
        raise ValueError("Episode is incomplete")
    with np.load(recording / "finger_reference.npz", allow_pickle=False) as reference:
        rest = [reference[f"finger_reference_{side}"] for side in ("left", "right")]
        pins = [reference[f"finger_pin_{side}"] for side in ("left", "right")]
    calibration = episode.get("calibration", {})
    intrinsics = calibration.get("finger_intrinsics")
    resolution = calibration.get("finger_resolution")
    projected_positions = []
    rows, local_positions, camera_in_mount = [], [], []
    for row in episode["frames"]:
        with open_frame(recording, row, include_images=False) as frame:
            mounts, cameras = frame["finger_mounts"], frame["finger_camera_to_world"]
            local = [(frame[f"finger_nodes_{name}"] - mounts[side, :3, 3]) @ mounts[side, :3, :3]
                     for side, name in enumerate(("left", "right"))]
            if any(points.shape != expected.shape or not np.isfinite(points).all()
                   for points, expected in zip(local, rest)):
                raise ValueError("Invalid or inconsistent cooked FEM nodes")
            camera_in_mount.append(np.linalg.inv(mounts) @ cameras)
            if intrinsics is not None and resolution is not None:
                projected_positions.append([project_nodes(frame[f"finger_nodes_{name}"], cameras[side],
                                                         np.asarray(intrinsics[side]))
                                            for side, name in enumerate(("left", "right"))])
        local_positions.append(local)
        displacement = [np.linalg.norm(points - expected, axis=1) * 1000
                        for points, expected in zip(local, rest)]
        rows.append(dict(time_s=row["time_s"], phase=row["phase"],
                         max_displacement_mm=[float(d.max()) for d in displacement],
                         p95_displacement_mm=[float(np.percentile(d, 95)) for d in displacement],
                         max_pin_error_mm=[float(d[pin].max()) for d, pin in zip(displacement, pins)]))
    camera_in_mount = np.asarray(camera_in_mount)
    reference_camera = camera_in_mount[0]
    translation_error = float(np.linalg.norm(camera_in_mount[:, :, :3, 3] - reference_camera[:, :3, 3], axis=-1).max() * 1000)
    rotation_error = float(np.abs(camera_in_mount[:, :, :3, :3] - reference_camera[:, :3, :3]).max())
    phases = {}
    baseline = next((i for i in range(len(rows) - 1, -1, -1) if rows[i]["phase"] == "approach"), None)
    if baseline is None:
        baseline = next((i for i in range(len(rows) - 1, -1, -1) if rows[i]["phase"] == "settle"), None)
    for phase in dict.fromkeys(row["phase"] for row in rows):
        indices = [i for i, row in enumerate(rows) if row["phase"] == phase]
        stats = dict(frames=len(indices),
                     maximum_displacement_mm=np.max([rows[i]["max_displacement_mm"] for i in indices], axis=0).tolist())
        if baseline is not None:
            stats["maximum_change_from_precontact_mm"] = [float(max(
                np.linalg.norm(local_positions[i][side] - local_positions[baseline][side], axis=1).max() * 1000
                for i in indices)) for side in (0, 1)]
        # Exclude initial settling after a hold starts. These are physical node
        # changes in the base frame, not camera motion or a visual stabilizer.
        if phase in ("hold", "verify_hold"):
            pairs = settled_phase_pairs(rows, phase)
            stats["settled_checked_pairs"] = len(pairs)
            if pairs:
                deltas = [np.stack([np.linalg.norm(local_positions[b][side] - local_positions[a][side], axis=1) * 1000
                                   for a, b in pairs]) for side in (0, 1)]
                stats["settled_max_frame_node_delta_mm"] = [float(d.max()) for d in deltas]
                stats["settled_p95_frame_node_delta_mm"] = [float(np.percentile(d, 95)) for d in deltas]
                if projected_positions:
                    # A submillimeter change near an internal lens can move
                    # bars several pixels. Use each frame's P95, then take the
                    # worst frame; pooling frames would hide short bursts.
                    motion_px = [[], []]
                    for a, b in pairs:
                        for side in (0, 1):
                            previous, current = projected_positions[a][side], projected_positions[b][side]
                            in_frame = np.all(np.isfinite(previous) & np.isfinite(current)
                                & (previous >= 0) & (current >= 0)
                                & (previous < resolution) & (current < resolution), axis=1)
                            delta = np.linalg.norm(current[in_frame] - previous[in_frame], axis=1)
                            if len(delta):
                                motion_px[side].append(np.percentile(delta, 95))
                    stats["settled_max_frame_p95_projected_node_delta_px"] = [
                        float(max(values)) if len(values) == len(pairs) else None
                        for values in motion_px]
        phases[phase] = stats
    maximum_pin_error = float(np.max([row["max_pin_error_mm"] for row in rows]))
    clamp_valid = maximum_pin_error <= maximum_pin_error_mm
    camera_valid = bool(translation_error < .001 and rotation_error < 1e-6)
    unloaded = phases.get("settle")
    initialization_valid = bool(max(unloaded["maximum_displacement_mm"]) <= maximum_unloaded_mm) if unloaded else None
    hold_pixel_motion = [stats.get("settled_max_frame_p95_projected_node_delta_px")
                         for phase, stats in phases.items() if phase in ("hold", "verify_hold")]
    hold_motion_valid = (all(value <= maximum_hold_p95_px for values in hold_pixel_motion for value in values)
                         if hold_pixel_motion and all(values is not None and None not in values
                                                      for values in hold_pixel_motion) else None)
    report = dict(valid=clamp_valid and camera_valid and initialization_valid is not False and hold_motion_valid is not False,
                  fps=episode.get("fps"),
                  finger_resolution=resolution,
                  camera_mount_valid=camera_valid,
                  physics_initialization_valid=initialization_valid,
                  hold_projected_motion_valid=hold_motion_valid,
                  maximum_allowed_hold_frame_p95_projected_node_delta_px=maximum_hold_p95_px,
                  maximum_allowed_unloaded_displacement_mm=maximum_unloaded_mm,
                  maximum_camera_mount_translation_error_mm=translation_error,
                  maximum_camera_mount_rotation_matrix_error=rotation_error,
                  maximum_pinned_node_error_mm=maximum_pin_error,
                  clamp_valid=clamp_valid, maximum_allowed_pin_error_mm=maximum_pin_error_mm,
                  precontact_reference_phase=rows[baseline]["phase"] if baseline is not None else None,
                  scope="Rigid camera mounting, FEM deformation and in-frustum node projection motion at recorded FPS; "
                        "projection includes occluded nodes and does not validate rendered RGB stability or calibrated damping",
                  phases=phases, frames=rows)
    (recording / "finger_motion_validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--maximum-unloaded-mm", type=float, default=2.,
                        help="Unloaded displacement limit for the current TPU cage; reassess for other materials")
    parser.add_argument("--maximum-hold-p95-px", type=float, default=.5,
                        help="Worst settled hold frame's P95 node projection motion limit at recorded FPS")
    parser.add_argument("--maximum-pin-error-mm", type=float, default=.01)
    args = parser.parse_args()
    report = check(args.recording, args.maximum_unloaded_mm, args.maximum_hold_p95_px, args.maximum_pin_error_mm)
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    raise SystemExit(0 if report["valid"] else 1)
