"""Motion-planning commands and task phases, independent of Isaac/Kit."""
import numpy as np


def placement_transform_from_pose(scene, goal_pose):
    """Rigid world displacement mapping the actual object pose to its goal."""
    from scipy.spatial.transform import Rotation
    goal = np.asarray(goal_pose, dtype=float)
    if (goal.shape != (4, 4) or not np.isfinite(goal).all()
            or not np.allclose(goal[3], [0, 0, 0, 1])
            or not np.allclose(goal[:3, :3].T @ goal[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(goal[:3, :3]), 1., atol=1e-6)):
        raise ValueError("Placement goal must be a finite rigid SE(3) transform")
    quaternion = np.asarray(scene["initial_object_orientation_wxyz"])
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_matrix()
    initial[:3, 3] = scene["initial_object_position"]
    return goal @ np.linalg.inv(initial)


def layout_target_parameters(layout, label):
    """Read either saved or manager-enriched records without mutating them."""
    return next(({"category": category, **record}
        for group in layout.values() if isinstance(group, dict)
        for category, records in group.items() if isinstance(records, list)
        for record in records if isinstance(record, dict) and record.get("label") == label), {})


def pickup_phases(plan, paths):
    opened, closed = plan["candidate"]["opened"], plan["candidate"]["closed"]
    release_width = plan["candidate"].get(
        "release_width",
        plan.get("placement", {}).get("release_width_m", opened),
    )
    sample_dt = plan.get("sample_dt_s", .02)
    initial = np.repeat(np.asarray(plan["initial_q"][:-2])[None], 2, axis=0)
    grasp = paths["approach"][-1:][[0, 0]]
    phases = [("settle", .3, initial, opened),
              ("pregrasp", max(.5, (len(paths["pregrasp"])-1)*sample_dt), paths["pregrasp"], opened),
              ("approach", max(1., (len(paths["approach"])-1)*sample_dt), paths["approach"], opened),
              ("close", 2., grasp, closed), ("hold", 1., grasp, closed),
              ("probe_lift", 1.5, paths["probe"], closed),
              ("lift", max(2., (len(paths["lift"])-1)*sample_dt), paths["lift"], closed)]
    if "placement" not in plan:
        return phases + [("verify_hold", 1., paths["lift"][-1:][[0, 0]], closed)]
    phases.extend((name, max(minimum, (len(paths[key])-1)*sample_dt), paths[key], width)
                  for name, key, minimum, width in (
                      ("transfer", "transfer", 2., closed),
                      ("place", "place", 2., closed),
                      ("release", "release", 1.5, release_width),
                      ("retreat", "retreat", 1., release_width),
                      ("return_home", "return_home", 2., opened)))
    phases.append(("verify_hold", 1., paths["return_home"][-1:][[0, 0]], opened))
    return phases


def hand_from_object_transform(hand_position, hand_rotation, object_position, object_quat_wxyz):
    """Compute 4x4 homogeneous transform of the object in the hand/mount frame."""
    from soft_sim.tasks.adapters import rotation

    world_from_hand = np.eye(4, dtype=float)
    world_from_hand[:3, :3] = np.asarray(hand_rotation, dtype=float)
    world_from_hand[:3, 3] = np.asarray(hand_position, dtype=float)

    world_from_obj = np.eye(4, dtype=float)
    world_from_obj[:3, :3] = rotation(object_quat_wxyz)
    world_from_obj[:3, 3] = np.asarray(object_position, dtype=float)
    return np.linalg.inv(world_from_hand) @ world_from_obj


def grasp_envelope_status(
    object_position,
    mount_center,
    *,
    initial_hand_from_object=None,
    current_hand_from_object=None,
    max_mount_distance_m=0.16,
    max_slip_m=0.08,
):
    """Check whether the gripped object remains within the finger grasp envelope."""
    obj_pos = np.asarray(object_position, dtype=float)
    mnt_pos = np.asarray(mount_center, dtype=float)
    mount_distance_m = float(np.linalg.norm(obj_pos - mnt_pos))
    slip_m = 0.0
    slip_deg = 0.0
    if initial_hand_from_object is not None and current_hand_from_object is not None:
        init_t = np.asarray(initial_hand_from_object, dtype=float)
        curr_t = np.asarray(current_hand_from_object, dtype=float)
        slip_m = float(np.linalg.norm(curr_t[:3, 3] - init_t[:3, 3]))
        rel_rot = init_t[:3, :3].T @ curr_t[:3, :3]
        cos_theta = float(np.clip((np.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0))
        slip_deg = float(np.degrees(np.arccos(cos_theta)))
    in_envelope = (mount_distance_m <= float(max_mount_distance_m)) and (
        slip_m <= float(max_slip_m)
    )
    return dict(
        in_envelope=bool(in_envelope),
        mount_distance_m=mount_distance_m,
        slip_m=slip_m,
        slip_deg=slip_deg,
    )


def trajectory_command(trajectory, duration, elapsed, previous_width, width):
    """Sample a TOPP path and its velocity, with smooth jaw motion in SI units."""
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or not len(trajectory) or not np.isfinite(trajectory).all():
        raise ValueError("Expected a finite, nonempty joint trajectory")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Phase duration must be positive")
    u = np.clip(elapsed / duration, 0., 1.)
    if len(trajectory) == 1:
        arm, velocity = trajectory[0], np.zeros(trajectory.shape[1])
    else:
        location = u * (len(trajectory)-1)
        lower = min(int(location), len(trajectory)-2)
        fraction = location-lower
        arm = (1-fraction)*trajectory[lower]+fraction*trajectory[lower+1]
        velocity = (trajectory[lower+1]-trajectory[lower])*(len(trajectory)-1)/duration
    jaw = previous_width + u*u*(3-2*u)*(width-previous_width)
    jaw_velocity = 6*u*(1-u)*(width-previous_width)/duration
    return np.r_[arm, jaw/2, jaw/2], np.r_[velocity, jaw_velocity/2, jaw_velocity/2], float(jaw)


def phase_commands(trajectory, steps, timestep_s, previous_width, width):
    """Prepare a phase once, using the same sampling epochs as trajectory_command.

    The host can transfer these small arrays to the robot device once per phase
    instead of validating the entire path and allocating two tensors at 1 kHz.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or not len(trajectory) or not np.isfinite(trajectory).all():
        raise ValueError("Expected a finite, nonempty joint trajectory")
    if not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("Phase steps must be a positive integer")
    if not np.isfinite(timestep_s) or timestep_s <= 0:
        raise ValueError("Phase timestep must be positive")
    duration = steps * timestep_s
    u = np.clip((np.arange(steps) + 1) * timestep_s / duration, 0., 1.)
    positions = np.empty((steps, trajectory.shape[1] + 2))
    velocities = np.empty_like(positions)
    if len(trajectory) == 1:
        positions[:, :-2] = trajectory[0]
        velocities[:, :-2] = 0.
    else:
        location = u * (len(trajectory) - 1)
        lower = np.minimum(location.astype(np.int64), len(trajectory) - 2)
        fraction = (location - lower)[:, None]
        positions[:, :-2] = (1 - fraction) * trajectory[lower] + fraction * trajectory[lower + 1]
        velocities[:, :-2] = (trajectory[lower + 1] - trajectory[lower]) * (len(trajectory) - 1) / duration
    jaw = previous_width + u*u*(3 - 2*u)*(width - previous_width)
    jaw_velocity = 6*u*(1 - u)*(width - previous_width) / duration
    positions[:, -2:] = jaw[:, None] / 2
    velocities[:, -2:] = jaw_velocity[:, None] / 2
    return positions, velocities, jaw


def settled_phase_pairs(rows, phase, settle_s=.25):
    """Only compare adjacent captures inside each occurrence of a hold phase."""
    pairs, start = [], None
    for index, row in enumerate(rows):
        if row["phase"] != phase:
            start = None
            continue
        if start is None:
            start = row["time_s"]
        if (index and rows[index-1]["phase"] == phase
                and rows[index-1]["time_s"] >= start+settle_s):
            pairs.append((index-1, index))
    return pairs


def support_placement_translation(target_vertices, support_vertices, gap_m=.002):
    """Center the unchanged payload over a support and leave a small release gap."""
    if not np.isfinite(gap_m) or gap_m < 0:
        raise ValueError("Release gap must be finite and nonnegative")
    target, support = np.asarray(target_vertices), np.asarray(support_vertices)
    if any(x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all()
           for x in (target, support)):
        raise ValueError("Placement needs finite target and support vertices")
    delta = (support.min(0)+support.max(0)-target.min(0)-target.max(0))/2
    delta[2] = support[:, 2].max()-target[:, 2].min()+gap_m
    return delta
