"""RoboDojo-aligned cuRobo GPU motion planner for soft-finger grasp and placement."""

from pathlib import Path
import json
import sys
import tempfile
import numpy as np

from soft_sim.config import ROOT
from soft_sim.execution import support_placement_translation
from soft_sim.geometry import matrix_to_quat_wxyz
from soft_sim.planning.candidates import generate_candidates
from soft_sim.planning.world_collision import get_scene_planner


_PLANNER_CACHE = {}
_PATCHED_YML_DIR = None


def _ensure_robodojo_path():
    robodojo_root = str((ROOT / "third_party/RoboDojo").resolve())
    if robodojo_root not in sys.path:
        sys.path.insert(0, robodojo_root)


def _patched_curobo_yml(robot_name: str) -> str:
    import yaml

    global _PATCHED_YML_DIR
    robot_dir = (ROOT / "third_party/RoboDojo/Assets/Robots" / robot_name).resolve()
    yml_src = robot_dir / "curobo.yml"
    if not yml_src.is_file():
        raise FileNotFoundError(f"Missing cuRobo config for robot '{robot_name}': {yml_src}")
    data = yaml.safe_load(yml_src.read_text())
    kin = data["robot_cfg"]["kinematics"]
    kin["asset_root_path"] = str(robot_dir)
    urdf_name = Path(kin.get("urdf_path", f"{robot_name.upper()}A.urdf")).name
    if not (robot_dir / urdf_name).is_file():
        urdfs = sorted(robot_dir.glob("*.urdf"))
        if not urdfs:
            raise FileNotFoundError(f"No URDF found in {robot_dir}")
        urdf_name = urdfs[0].name
    kin["urdf_path"] = str(robot_dir / urdf_name)
    if _PATCHED_YML_DIR is None:
        _PATCHED_YML_DIR = Path(tempfile.mkdtemp(prefix="soft_sim_curobo_"))
    out_yml = _PATCHED_YML_DIR / f"{robot_name}_curobo.yml"
    out_yml.write_text(yaml.safe_dump(data))
    return str(out_yml)


def get_curobo_planner(robot_name: str, arm_joints: list[str], table_height: float, dt: float = 0.02):
    key = (robot_name, round(float(table_height), 3), round(float(dt), 4))
    if key in _PLANNER_CACHE:
        return _PLANNER_CACHE[key]
    _ensure_robodojo_path()
    from env.planner_manager.curobo_planner import CuroboPlanner

    planner = CuroboPlanner(
        robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        active_joints_name=list(arm_joints),
        all_joints=list(arm_joints),
        dt=float(dt),
        yml_path=_patched_curobo_yml(robot_name),
        table_height=float(table_height),
    )
    _PLANNER_CACHE[key] = planner
    return planner


def _cartesian_ik_segment(
    planner,
    q_start: np.ndarray,
    pos_start: np.ndarray,
    pos_end: np.ndarray,
    hand_quat: list[float],
    base_pose: list[float],
    steps: int = 6,
    max_step_rad: float = 0.75,
):
    """Solve a short Cartesian segment via seeded cuRobo IK waypoints."""
    q_curr = np.asarray(q_start, dtype=float).copy()
    waypoints = [q_curr]
    for s in np.linspace(0.0, 1.0, steps)[1:]:
        pos = (1.0 - s) * pos_start + s * pos_end
        ik = planner.solve_ik_to_joint(
            q_curr.tolist(),
            pos.tolist() + list(hand_quat),
            real_robot_pose=base_pose,
        )
        if ik.get("status") != "Success":
            return dict(status=f"Cartesian IK failed at s={s:.2f}")
        q_next = np.asarray(ik["joint_value"], dtype=float)
        if np.max(np.abs(q_next - q_curr)) > max_step_rad:
            return dict(status="Cartesian IK joint jump")
        waypoints.append(q_next)
        q_curr = q_next
    # Subsample/densify slightly for smooth PhysX tracking.
    dense = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        dense.append(0.5 * (a + b))
        dense.append(b)
    return dict(status="Success", position=np.asarray(dense, dtype=float))


def plan_scene_curobo(
    scene_dir,
    out,
    *,
    contact_height=None,
    friction=0.7,
    indent=0.003,
    max_attempts=24,
    lift_height=0.10,
    place_on=None,
    release_gap=0.002,
    sample_dt=0.02,
    place_pose=None,
):
    """Plan grasp + optional placement using RoboDojo's CuroboPlanner."""
    scene_dir = Path(scene_dir).resolve()
    out = Path(out).resolve()
    info = generate_candidates(
        scene_dir,
        out,
        contact_height=contact_height,
        friction=friction,
        indent=indent,
    )
    scene = info["scene"]
    contact_mesh = info["contact_mesh"]
    object_mesh = info["object_mesh"]
    arm = info["arm"]
    tcp_origin_m = info["tcp_origin_m"]
    tool_forward_m = info["tool_forward_m"]
    height = info["height"]
    radius = info["radius"]
    hand_to_tcp_m = info["hand_to_tcp_m"]
    candidates = info["candidates"]

    if place_on is not None and place_pose is not None:
        raise ValueError("Use only one placement destination")
    placement_transform = None
    if place_on:
        with np.load(place_on) as support:
            placement_transform = np.eye(4)
            placement_transform[:3, 3] = support_placement_translation(
                contact_mesh.vertices, support["vertices"], release_gap
            )
    if place_pose:
        from soft_sim.execution import placement_transform_from_pose

        goal_pose = json.loads(Path(place_pose).read_text())["world_from_object"]
        placement_transform = placement_transform_from_pose(scene, goal_pose)

    world_from_base = np.asarray(scene.get("world_from_robot_base", np.eye(4)), dtype=float)
    if "world_from_robot_base" not in scene:
        world_from_base[:3, 3] = [0.0, 0.0, scene["table_z"]]
    base_translation = world_from_base[:3, 3]
    base_quat = list(matrix_to_quat_wxyz(world_from_base[:3, :3]))
    base_pose = list(base_translation) + base_quat

    n_arm = len(arm.arm_joints)
    q_init = np.asarray(scene["initial_q"], dtype=float)
    q0_arm = q_init[:n_arm]

    # RoboDojo sets table_height = 0.74 - root_pose[2] = table_z - base_z - 0.025
    # so base_link's bottom collision sphere stays clear of the table top.
    table_height = float(scene["table_z"]) - float(base_translation[2]) - 0.025
    planner = get_scene_planner(
        scene_dir,
        arm.name,
        list(arm.arm_joints),
        table_height,
        sample_dt=sample_dt,
    )

    def to_hand_pose(tcp_pos, tcp_rot):
        hand_r = tcp_rot @ arm.rotation.T
        hand_pos = tcp_pos - hand_r @ (
            tcp_origin_m + arm.growth_axis_in_hand * (tool_forward_m + height)
        )
        return hand_pos, list(matrix_to_quat_wxyz(hand_r))

    attempts = []
    for index, candidate in enumerate(candidates[:max_attempts]):
        tcp = np.asarray(candidate["tcp"], dtype=float)
        r = np.asarray(candidate["rotation"], dtype=float)
        hand_pos, hand_quat = to_hand_pose(tcp, r)
        pregrasp_tcp = tcp - r[:, 2] * 0.075
        pregrasp_pos, _ = to_hand_pose(pregrasp_tcp, r)

        pregrasp = planner.plan_path(
            q0_arm.tolist(),
            pregrasp_pos.tolist() + hand_quat,
            real_robot_pose=base_pose,
        )
        if pregrasp.get("status") != "Success":
            attempts.append(dict(index=index, phase="pregrasp", status=pregrasp.get("status", "Fail")))
            continue

        q_pre = np.asarray(pregrasp["position"][-1], dtype=float)
        approach = _cartesian_ik_segment(
            planner, q_pre, pregrasp_pos, hand_pos, hand_quat, base_pose, steps=6
        )
        if approach.get("status") != "Success":
            attempts.append(dict(index=index, phase="approach", status=approach["status"]))
            continue

        q_grasp = approach["position"][-1]
        probe_pos, _ = to_hand_pose(tcp + [0.0, 0.0, 0.012], r)
        probe = _cartesian_ik_segment(
            planner, q_grasp, hand_pos, probe_pos, hand_quat, base_pose, steps=3
        )
        if probe.get("status") != "Success":
            attempts.append(dict(index=index, phase="probe", status=probe["status"]))
            continue

        q_probe = probe["position"][-1]
        lift_pos, _ = to_hand_pose(tcp + [0.0, 0.0, lift_height], r)
        lift = _cartesian_ik_segment(
            planner, q_probe, probe_pos, lift_pos, hand_quat, base_pose, steps=6
        )
        if lift.get("status") != "Success":
            lift = planner.plan_path(
                q_probe.tolist(), lift_pos.tolist() + hand_quat, real_robot_pose=base_pose
            )
        if lift.get("status") != "Success":
            attempts.append(dict(index=index, phase="lift", status=lift.get("status", "Fail")))
            continue

        trajectories = {
            "pregrasp": np.asarray(pregrasp["position"], dtype=float),
            "approach": approach["position"],
            "probe": probe["position"],
            "lift": np.asarray(lift["position"], dtype=float),
        }

        placement = None
        if placement_transform is not None:
            goal = placement_transform[:3, :3] @ tcp + placement_transform[:3, 3]
            goal_rotation = placement_transform[:3, :3] @ r
            above = goal + [0.0, 0.0, 0.09]
            above_pos, goal_hand_quat = to_hand_pose(above, goal_rotation)
            goal_pos, _ = to_hand_pose(goal, goal_rotation)
            retreat_pos, _ = to_hand_pose(goal - goal_rotation[:, 2] * 0.09, goal_rotation)

            q_lift = trajectories["lift"][-1]
            transfer = planner.plan_path(
                q_lift.tolist(),
                above_pos.tolist() + goal_hand_quat,
                real_robot_pose=base_pose,
            )
            if transfer.get("status") != "Success":
                attempts.append(dict(index=index, phase="transfer", status=transfer.get("status", "Fail")))
                continue

            q_above = np.asarray(transfer["position"][-1], dtype=float)
            place = _cartesian_ik_segment(
                planner, q_above, above_pos, goal_pos, goal_hand_quat, base_pose, steps=6
            )
            if place.get("status") != "Success":
                place = planner.plan_path(
                    q_above.tolist(), goal_pos.tolist() + goal_hand_quat, real_robot_pose=base_pose
                )
            if place.get("status") != "Success":
                attempts.append(dict(index=index, phase="place", status=place.get("status", "Fail")))
                continue

            q_place = np.asarray(place["position"][-1], dtype=float)
            retreat = _cartesian_ik_segment(
                planner, q_place, goal_pos, retreat_pos, goal_hand_quat, base_pose, steps=6
            )
            if retreat.get("status") != "Success":
                retreat = planner.plan_path(
                    q_place.tolist(), retreat_pos.tolist() + goal_hand_quat, real_robot_pose=base_pose
                )
            if retreat.get("status") != "Success":
                attempts.append(dict(index=index, phase="retreat", status=retreat.get("status", "Fail")))
                continue

            q_retreat = np.asarray(retreat["position"][-1], dtype=float)
            home = planner.plan_joint(q_retreat.tolist(), q0_arm.tolist())
            if home.get("status") != "Success":
                attempts.append(dict(index=index, phase="return_home", status=home.get("status", "Fail")))
                continue

            trajectories.update(
                transfer=np.asarray(transfer["position"], dtype=float),
                place=np.asarray(place["position"], dtype=float),
                release=np.repeat(np.asarray(place["position"], dtype=float)[-1:], 2, axis=0),
                retreat=np.asarray(retreat["position"], dtype=float),
                return_home=np.asarray(home["position"], dtype=float),
            )
            placement = dict(
                world_displacement=placement_transform.tolist(),
                tcp_goal_m=goal.tolist(),
                tcp_goal_rotation=goal_rotation.tolist(),
                release_gap_m=release_gap,
                support_geometry=str(Path(place_on).resolve()) if place_on else None,
                goal_pose_source=str(Path(place_pose).resolve()) if place_pose else None,
                final_object_position=(
                    placement_transform[:3, :3] @ np.asarray(scene["initial_object_position"])
                    + placement_transform[:3, 3]
                ).tolist(),
            )

        candidate_record = dict(candidate)
        candidate_record["hand_position"] = hand_pos.tolist()
        candidate_record["hand_rotation"] = (r @ arm.rotation.T).tolist()
        if placement is not None and scene.get("task") == "insert_tubes":
            offset = float(scene.get("adapter_offset_m", 0.0))
            opened = float(candidate["opened"])
            closed = float(candidate["closed"])
            release_width = min(
                opened,
                max(
                    closed + 0.004,
                    float(candidate["width"]) + 2.0 * float(radius) - 2.0 * offset + 0.012,
                ),
            )
            candidate_record["release_width"] = float(release_width)
            placement["release_width_m"] = float(release_width)
        np.savez_compressed(out / "joints.npz", **trajectories)
        report = dict(
            status="planned",
            candidate=candidate_record,
            world_from_robot_base_translation_m=base_translation.tolist(),
            world_from_robot_base=world_from_base.tolist(),
            lift_height_m=lift_height,
            contact_height_m=height,
            finger_inner_radius_m=radius,
            hand_to_tcp_m=hand_to_tcp_m,
            initial_q=q_init.tolist(),
            world_collision=getattr(planner, "world_report", None),
            scene=scene,
            trajectory_file="joints.npz",
            sample_dt_s=sample_dt,
            planner="RoboDojo CuroboPlanner (GPU batched IK + TrajOpt) with soft-finger geometry candidates",
            collision_checked=True,
            physical_grasp_validated=False,
            friction_coefficient=friction,
            probe_lift_m=0.012,
            lift_goal_rotation=r.tolist(),
            attempts=attempts,
        )
        if placement is not None:
            report["placement"] = placement
        (out / "plan.json").write_text(json.dumps(report, indent=2))
        print(
            "SOFT_GRASP_PLAN",
            json.dumps({k: v for k, v in report.items() if k not in ("scene", "attempts")}),
            flush=True,
        )
        return report

    raise RuntimeError(f"cuRobo planning failed for all {min(len(candidates), max_attempts)} candidates: {attempts}")


def replan_placement_curobo(
    scene_dir,
    plan,
    current_q,
    world_from_goal_object,
    hand_from_object,
    *,
    planner=None,
    sample_dt=0.02,
):
    """Replan transfer/place/release/retreat/return_home for the actually gripped hand_from_object."""
    from soft_sim.robots.mechanism import mechanism

    scene = plan["scene"]
    arm = mechanism(scene.get("robot_name"))
    world_from_base = np.asarray(
        plan.get("world_from_robot_base", scene.get("world_from_robot_base", np.eye(4))),
        dtype=float,
    )
    base_pose = world_from_base[:3, 3].tolist() + matrix_to_quat_wxyz(world_from_base[:3, :3]).tolist()
    if planner is None:
        table_height = float(scene["table_z"]) - float(world_from_base[2, 3]) - 0.025
        planner = get_scene_planner(
            scene_dir,
            arm.name,
            list(arm.arm_joints),
            table_height,
            sample_dt=sample_dt,
        )

    n_arm = len(arm.arm_joints)
    q_curr = np.asarray(current_q, dtype=float)[:n_arm]
    q0_arm = np.asarray(plan["initial_q"], dtype=float)[:n_arm]
    world_from_goal_hand = np.asarray(world_from_goal_object, dtype=float) @ np.linalg.inv(
        np.asarray(hand_from_object, dtype=float)
    )
    goal_hand_pos = world_from_goal_hand[:3, 3]
    goal_hand_rot = world_from_goal_hand[:3, :3]
    goal_hand_quat = matrix_to_quat_wxyz(goal_hand_rot).tolist()
    above_pos = goal_hand_pos + np.array([0.0, 0.0, 0.06])
    tool_axis_world = goal_hand_rot @ (arm.rotation @ np.array([0.0, 0.0, 1.0]))
    retreat_pos = goal_hand_pos - tool_axis_world * 0.09

    transfer = planner.plan_path(
        q_curr.tolist(), above_pos.tolist() + goal_hand_quat, real_robot_pose=base_pose
    )
    if transfer.get("status") != "Success":
        raise RuntimeError(f"Placement replan transfer failed: {transfer.get('status')}")
    q_above = np.asarray(transfer["position"][-1], dtype=float)
    place = _cartesian_ik_segment(
        planner, q_above, above_pos, goal_hand_pos, goal_hand_quat, base_pose, steps=6
    )
    if place.get("status") != "Success":
        place = planner.plan_path(
            q_above.tolist(), goal_hand_pos.tolist() + goal_hand_quat, real_robot_pose=base_pose
        )
    if place.get("status") != "Success":
        raise RuntimeError(f"Placement replan place failed: {place.get('status')}")
    q_place = np.asarray(place["position"][-1], dtype=float)
    retreat = _cartesian_ik_segment(
        planner, q_place, goal_hand_pos, retreat_pos, goal_hand_quat, base_pose, steps=6
    )
    if retreat.get("status") != "Success":
        retreat = planner.plan_path(
            q_place.tolist(), retreat_pos.tolist() + goal_hand_quat, real_robot_pose=base_pose
        )
    if retreat.get("status") != "Success":
        raise RuntimeError(f"Placement replan retreat failed: {retreat.get('status')}")
    q_retreat = np.asarray(retreat["position"][-1], dtype=float)
    home = planner.plan_joint(q_retreat.tolist(), q0_arm.tolist())
    if home.get("status") != "Success":
        raise RuntimeError(f"Placement replan return_home failed: {home.get('status')}")

    updated_paths = dict(
        transfer=np.asarray(transfer["position"], dtype=float),
        place=np.asarray(place["position"], dtype=float),
        release=np.repeat(np.asarray(place["position"], dtype=float)[-1:], 2, axis=0),
        retreat=np.asarray(retreat["position"], dtype=float),
        return_home=np.asarray(home["position"], dtype=float),
    )
    revision = dict(
        world_from_goal_hand=world_from_goal_hand.tolist(),
        hand_from_object=np.asarray(hand_from_object, dtype=float).tolist(),
        world_collision=getattr(planner, "world_report", None),
    )
    return updated_paths, revision


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Plan a soft-finger grasp on an exported scene with cuRobo.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-height", type=float)
    parser.add_argument("--friction", type=float, default=0.7)
    parser.add_argument("--indent", type=float, default=0.003)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--place-on", type=Path)
    parser.add_argument("--place-pose", type=Path)
    parser.add_argument("--release-gap", type=float, default=0.002)
    parser.add_argument("--sample-dt", type=float, default=0.02)
    args = parser.parse_args(argv)
    plan_scene_curobo(
        args.scene,
        args.output,
        contact_height=args.contact_height,
        friction=args.friction,
        indent=args.indent,
        max_attempts=args.max_attempts,
        lift_height=args.lift_height,
        place_on=args.place_on,
        release_gap=args.release_gap,
        sample_dt=args.sample_dt,
        place_pose=args.place_pose,
    )
