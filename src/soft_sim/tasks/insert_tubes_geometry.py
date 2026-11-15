"""Real-geometry hole extraction and placement for ``insert_tubes``."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
import trimesh


def _pose_matrix(position, quaternion_wxyz):
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    q = np.asarray(quaternion_wxyz, dtype=float)
    matrix[:3, :3] = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
    return matrix


def extract_slot_holes(
    slot_vertices,
    slot_faces,
    slot_position,
    slot_quaternion_wxyz,
    *,
    grid_step_m=0.002,
    min_depth_m=0.015,
    min_area_m2=1e-4,
):
    """Extract real vertical-axis holes from the slot rack mesh in its local frame."""
    slot_pose = _pose_matrix(slot_position, slot_quaternion_wxyz)
    slot_rot = slot_pose[:3, :3]
    slot_pos = slot_pose[:3, 3]

    local_v = (np.asarray(slot_vertices, dtype=float) - slot_pos) @ slot_rot
    slot_local = trimesh.Trimesh(local_v, np.asarray(slot_faces, dtype=np.int32), process=False)
    bmin, bmax = slot_local.bounds

    xs = np.arange(bmin[0] + grid_step_m * 0.5, bmax[0], grid_step_m)
    ys = np.arange(bmin[1] + grid_step_m * 0.5, bmax[1], grid_step_m)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Slot mesh has degenerate XY bounds")
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    origins = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, bmax[2] + 0.02)])
    dirs = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    locs, idx_ray, _ = slot_local.ray.intersects_location(origins, dirs, multiple_hits=False)

    flat_z = np.full(len(origins), np.nan)
    flat_z[idx_ray] = locs[:, 2]
    z_map = flat_z.reshape(gx.shape)

    hole_mask = np.isfinite(z_map) & (z_map < bmax[2] - min_depth_m)
    labels, n_comp = ndimage.label(hole_mask)

    holes = []
    for cid in range(1, n_comp + 1):
        mask = labels == cid
        area = float(mask.sum()) * (grid_step_m**2)
        if area < min_area_m2:
            continue
        cx = float(gx[mask].mean())
        cy = float(gy[mask].mean())
        z_floor = float(np.median(z_map[mask]))
        depth_m = float(bmax[2] - z_floor)
        rx = float(gx[mask].max() - gx[mask].min()) * 0.5 + grid_step_m * 0.5
        ry = float(gy[mask].max() - gy[mask].min()) * 0.5 + grid_step_m * 0.5
        equiv_r = float(np.sqrt(area / np.pi))
        world_floor = slot_pos + slot_rot @ np.array([cx, cy, z_floor])
        world_top = slot_pos + slot_rot @ np.array([cx, cy, bmax[2]])
        holes.append(
            dict(
                component_id=int(cid),
                local_center_xy_m=[cx, cy],
                local_z_floor_m=z_floor,
                depth_m=depth_m,
                area_m2=area,
                radius_x_m=rx,
                radius_y_m=ry,
                equiv_radius_m=equiv_r,
                world_floor_center_m=world_floor.tolist(),
                world_top_center_m=world_top.tolist(),
                axis_world=(slot_rot @ np.array([0.0, 0.0, 1.0])).tolist(),
            )
        )
    summary = dict(
        grid_step_m=float(grid_step_m),
        min_depth_m=float(min_depth_m),
        rack_local_bounds_min_m=bmin.tolist(),
        rack_local_bounds_max_m=bmax.tolist(),
        total_components=int(n_comp),
        valid_holes_count=len(holes),
    )
    return holes, summary


def measure_tube_profile(tube_vertices, tube_position, tube_quaternion_wxyz, goal_rotation):
    """Measure upright tube height, origin-to-bottom offset, and shaft cross-section radius."""
    cur_pose = _pose_matrix(tube_position, tube_quaternion_wxyz)
    local_v = (np.asarray(tube_vertices, dtype=float) - cur_pose[:3, 3]) @ cur_pose[:3, :3]
    goal_rotation = np.asarray(goal_rotation, dtype=float)
    upright_v = local_v @ goal_rotation.T

    umin = upright_v.min(axis=0)
    umax = upright_v.max(axis=0)
    height_m = float(umax[2] - umin[2])
    z_from_bottom = upright_v[:, 2] - umin[2]
    shaft_mask = (z_from_bottom >= 0.05 * height_m) & (z_from_bottom <= 0.70 * height_m)
    shaft_v = upright_v[shaft_mask] if shaft_mask.any() else upright_v
    shaft_center_xy = (shaft_v[:, :2].min(axis=0) + shaft_v[:, :2].max(axis=0)) * 0.5
    shaft_radius_m = float(np.linalg.norm(shaft_v[:, :2] - shaft_center_xy, axis=1).max())
    full_radius_m = float(np.linalg.norm(upright_v[:, :2] - shaft_center_xy, axis=1).max())

    return dict(
        height_m=height_m,
        origin_to_bottom_m=float(-umin[2]),
        upright_center_xy_offset_m=shaft_center_xy.tolist(),
        shaft_radius_m=shaft_radius_m,
        full_radius_m=full_radius_m,
    )


def select_insert_tubes_holes(holes, tube_shaft_radius_m, *, count=3, required_depth_m=0.045):
    """Select ``count`` unambiguous qualifying holes that fit the tube shaft and depth."""
    qualifying = [
        h
        for h in holes
        if h["equiv_radius_m"] >= tube_shaft_radius_m * 0.92 and h["depth_m"] >= required_depth_m
    ]
    if len(qualifying) < count:
        raise ValueError(
            f"Insufficient qualifying slot holes ({len(qualifying)} < {count}) for "
            f"tube_shaft_radius_m={tube_shaft_radius_m:.4f}, required_depth_m={required_depth_m:.4f}"
        )
    # Sort deterministically by (local_y, local_x) so front-row holes come first,
    # then pick `count` well-separated holes across the qualifying row.
    front_y = min(h["local_center_xy_m"][1] for h in qualifying)
    front_row = [h for h in qualifying if abs(h["local_center_xy_m"][1] - front_y) <= 0.015]
    pool = sorted(front_row if len(front_row) >= count else qualifying, key=lambda h: (
        round(h["local_center_xy_m"][0], 4),
        round(h["local_center_xy_m"][1], 4),
    ))
    if len(pool) == count:
        return pool
    # Spread evenly across available holes in the pool (e.g. indices [1, 2, 3] or [0, 2, 4] of 5)
    if len(pool) >= count + 2:
        indices = np.linspace(1, len(pool) - 2, count, dtype=int).tolist()
    else:
        indices = np.linspace(0, len(pool) - 1, count, dtype=int).tolist()
    if len(set(indices)) != count:
        raise ValueError(f"Ambiguous hole selection indices {indices} from {len(pool)} holes")
    return [pool[i] for i in indices]


def insert_tubes_placement(
    scene,
    scene_dir,
    op,
    *,
    goal_rotation,
    gap_m=0.002,
    floor_clearance_m=0.002,
):
    """Compute real hole-aligned ``world_from_goal_object`` and provenance for ``insert_tubes``."""
    scene_dir = Path(scene_dir)
    objects = scene["task_objects"]
    slot_meta = objects["slot"]
    tube_meta = objects[op.target]

    slot_path = scene_dir / slot_meta["geometry_file"]
    tube_path = scene_dir / tube_meta["geometry_file"]
    with np.load(slot_path) as slot_npz:
        holes, rack_summary = extract_slot_holes(
            slot_npz["vertices"],
            slot_npz["faces"],
            slot_meta["position"],
            slot_meta["quaternion_wxyz"],
        )
    with np.load(tube_path) as tube_npz:
        profile = measure_tube_profile(
            tube_npz["vertices"],
            tube_meta["position"],
            tube_meta["quaternion_wxyz"],
            goal_rotation,
        )

    selected_holes = select_insert_tubes_holes(
        holes,
        profile["shaft_radius_m"],
        count=op.count,
    )
    chosen = selected_holes[op.slot]
    axis_world = np.asarray(chosen["axis_world"], dtype=float)
    floor_world = np.asarray(chosen["world_floor_center_m"], dtype=float)
    target_bottom_world = floor_world + axis_world * (floor_clearance_m + gap_m)

    # Align the upright tube's shaft XY center and bottom Z to target_bottom_world.
    goal = np.eye(4, dtype=float)
    goal[:3, :3] = np.asarray(goal_rotation, dtype=float)
    offset_xy = np.asarray(profile["upright_center_xy_offset_m"], dtype=float)
    goal[:2, 3] = target_bottom_world[:2] - offset_xy
    goal[2, 3] = target_bottom_world[2] + profile["origin_to_bottom_m"]

    report = dict(
        algorithm="raycast_hole_extraction_v1",
        slot_geometry_file=str(slot_meta["geometry_file"]),
        slot_geometry_sha256=hashlib.sha256(slot_path.read_bytes()).hexdigest(),
        tube_geometry_file=str(tube_meta["geometry_file"]),
        tube_geometry_sha256=hashlib.sha256(tube_path.read_bytes()).hexdigest(),
        rack_summary=rack_summary,
        tube_profile=profile,
        selected_holes=selected_holes,
        chosen_slot_index=int(op.slot),
        chosen_hole=chosen,
        floor_clearance_m=float(floor_clearance_m),
        gap_m=float(gap_m),
        target_bottom_world_m=target_bottom_world.tolist(),
        world_from_goal_object=goal.tolist(),
    )
    return goal, report
