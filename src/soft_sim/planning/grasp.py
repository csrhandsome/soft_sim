"""Geometry-aware soft-finger grasp candidates + mplib collision-aware planning.

Plans in the new soft_grasp_tcp frame. Does not treat the factory palm as the
contact center. A plan is only a geometric hypothesis until the FEM lift test.
"""

from pathlib import Path
import argparse
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import mplib
from mplib import Pose
from mplib.collision_detection import fcl
from soft_sim.geometry import matrix_to_quat_wxyz
from soft_sim.execution import support_placement_translation
from soft_sim.planning.candidates import (
    centered_closure,
    generate_candidates,
    section_radius,
    surface_contacts,
)


def bvh(vertices, faces):
    model = fcl.BVHModel()
    model.begin_model(len(faces), len(vertices))
    model.add_sub_model(
        list(np.asarray(vertices, dtype=float)), list(np.asarray(faces, dtype=np.int32))
    )
    model.end_model()
    return model


def plan_scene(
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
    """Plan one grasp (plus optional placement) on an exported scene directory."""
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
    # Use the actual triangle mesh (not convex_hull) so hollow containers/cups
    # preserve their interior cavities during placement collision checks.
    object_mesh = contact_mesh
    arm = info["arm"]
    installation = info["installation"]
    scale = info["scale"]
    height = info["height"]
    radius = info["radius"]
    tcp_origin_m = info["tcp_origin_m"]
    tool_forward_m = info["tool_forward_m"]
    hand_to_tcp_m = info["hand_to_tcp_m"]
    candidates = info["candidates"]
    table_z = scene["table_z"]
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
    from soft_sim.robots.fingers import make_planning_urdf, scene_joint_vector

    tool_in_hand = arm.rotation
    urdf = make_planning_urdf(out, height, scale, installation, arm)
    joint_names = list(arm.joint_names)
    n_arm = len(arm.arm_joints)
    planner = mplib.Planner(
        urdf=urdf,
        srdf=out / f"soft_{arm.name}.srdf",
        move_group="soft_grasp_tcp",
        user_joint_names=joint_names,
        joint_vel_limits=np.full(n_arm, 0.45),
        joint_acc_limits=np.full(n_arm, 0.7),
    )
    # Keep robot, table, target, and attached payload in one BASE frame.
    # Convert every world target explicitly. This avoids attached-body pose
    # inconsistencies when a non-identity robot base pose is used in mplib.
    planner.set_base_pose(Pose([0, 0, 0], [1, 0, 0, 0]))
    world_from_base = np.asarray(
        scene.get("world_from_robot_base", np.eye(4)), dtype=float
    )
    if "world_from_robot_base" not in scene:
        world_from_base[:3, 3] = [0.0, 0.0, table_z]
    base_translation = world_from_base[:3, 3]
    base_rotation = world_from_base[:3, :3]

    def base_pose(position, quaternion):
        rotation = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_matrix()
        return Pose(
            (np.asarray(position) - base_translation) @ base_rotation,
            matrix_to_quat_wxyz(base_rotation.T @ rotation),
        )

    target_vertices = (object_mesh.vertices - base_translation) @ base_rotation
    table_center = scene.get("table_center_m", [0.35, 0, table_z - 0.025])
    planner.planning_world.add_object(
        "table",
        fcl.CollisionObject(
            fcl.Box(scene.get("table_size_m", [1.05, 0.8, 0.05])),
            base_pose(table_center, [1, 0, 0, 0]),
        ),
    )
    planner.planning_world.add_object(
        "target", fcl.CollisionObject(bvh(target_vertices, object_mesh.faces))
    )
    obstacles_file = scene_dir / "obstacle_geometry.npz"
    if obstacles_file.exists():
        with np.load(obstacles_file) as obstacles:
            for index in range(int(obstacles["count"])):
                vertices_base = (
                    obstacles[f"vertices_{index}"] - base_translation
                ) @ base_rotation
                mesh = trimesh.Trimesh(
                    vertices_base, obstacles[f"faces_{index}"], process=False
                )
                if index not in scene.get("triangle_obstacles", []):
                    mesh = mesh.convex_hull
                planner.planning_world.add_object(
                    f"obstacle_{index}",
                    fcl.CollisionObject(bvh(mesh.vertices, mesh.faces)),
                )
    acm = planner.planning_world.get_allowed_collision_matrix()
    acm.set_entry(arm.base_link, "table", True)
    for side, parent in (("left", arm.finger_links[0]), ("right", arm.finger_links[1])):
        for component in (
            "soft_" + side,
            "soft_" + side + "_adapter",
            "soft_" + side + "_camera",
        ):
            acm.set_entry(
                component,
                [
                    parent,
                    "soft_" + side,
                    "soft_" + side + "_adapter",
                    "soft_" + side + "_camera",
                ],
                True,
            )

    def cartesian_segment(start_q, start_position, end_position, quaternion):
        path = [np.asarray(start_q).copy()]
        count = max(
            2,
            int(
                np.ceil(
                    np.linalg.norm(np.asarray(end_position) - start_position) / 0.002
                )
            ),
        )
        for t in np.linspace(0, 1, count + 1)[1:]:
            position = (1 - t) * np.asarray(start_position) + t * np.asarray(
                end_position
            )
            status, solution = planner.IK(
                base_pose(position, quaternion),
                path[-1],
                mask=[False] * n_arm + [True, True],
                n_init_qpos=12,
                return_closest=True,
                threshold=0.0005,
            )
            if status != "Success":
                return {
                    "status": "Cartesian IK: "
                    + status
                    + " fraction="
                    + str(round(float(t), 3))
                }
            for fraction in np.linspace(
                0, 1, max(3, int(np.max(np.abs(solution - path[-1])) / 0.015) + 1)
            )[1:]:
                sample = (1 - fraction) * path[-1] + fraction * solution
                if planner.check_for_env_collision(
                    sample
                ) or planner.check_for_self_collision(sample):
                    return {"status": "Cartesian interpolation collision"}
            path.append(solution.copy())
        times, positions, velocities, accelerations, duration = planner.TOPP(
            np.asarray(path)[:, :n_arm], sample_dt
        )
        return dict(status="Success", position=positions, time=times, duration=duration)

    attempts = []
    for index, candidate in enumerate(candidates[:max_attempts]):
        q = scene_joint_vector(scene, arm)
        q[-2:] = candidate["opened"] / 2
        tcp = np.array(candidate["tcp"])
        r = np.array(candidate["rotation"])
        quat = matrix_to_quat_wxyz(r)
        pre = tcp - r[:, 2] * 0.09
        initial_collision = planner.check_for_env_collision(
            q
        ) + planner.check_for_self_collision(q)
        if initial_collision:
            print(
                "INITIAL_COLLISION",
                [(x.link_name1, x.link_name2) for x in initial_collision],
                flush=True,
            )
            raise RuntimeError("Initial custom tool configuration is in collision")
        approach = planner.plan_pose(
            base_pose(pre, quat),
            q,
            mask=[False] * n_arm + [True, True],
            time_step=sample_dt,
            planning_time=2.0,
            verbose=False,
        )
        if approach["status"] != "Success":
            attempts.append(
                dict(index=index, phase="pregrasp", status=approach["status"])
            )
            continue
        qpre = np.r_[approach["position"][-1], q[-2:]]
        descend = cartesian_segment(qpre, pre, tcp, quat)
        if descend["status"] != "Success":
            attempts.append(
                dict(index=index, phase="final_approach", status=descend["status"])
            )
            continue
        qclosed = np.r_[descend["position"][-1], np.repeat(candidate["closed"] / 2, 2)]
        # Only intentional object/soft-finger contacts are allowed for closure.
        for side in ("soft_left", "soft_right"):
            acm.set_entry(side, "target", True)
        closed_collisions = []
        qopen = np.r_[descend["position"][-1], np.repeat(candidate["opened"] / 2, 2)]
        for fraction in np.linspace(0, 1, 12):
            sample = qopen * (1 - fraction) + qclosed * fraction
            closed_collisions = planner.check_for_env_collision(
                sample
            ) + planner.check_for_self_collision(sample)
            if closed_collisions:
                break
        if closed_collisions:
            np.save(out / "closed_collision_q.npy", qclosed)
            print(
                "CLOSED_COLLISION",
                index,
                [(x.link_name1, x.link_name2) for x in closed_collisions],
                flush=True,
            )
            for side in ("soft_left", "soft_right"):
                acm.set_entry(side, "target", False)
            attempts.append(
                dict(index=index, phase="closed_geometry", status="collision")
            )
            continue
        planner.remove_object("target")
        carried = (object_mesh.vertices - tcp) @ r
        planner.update_attached_object(
            bvh(carried, object_mesh.faces), Pose(), name="payload"
        )
        payload_attached = True
        acm.set_entry("payload", ["soft_left", "soft_right"], True)
        probe = cartesian_segment(qclosed, tcp, tcp + [0, 0, 0.012], quat)
        if probe["status"] == "Success":
            qprobe = np.r_[probe["position"][-1], qclosed[-2:]]
            lift = planner.plan_pose(
                base_pose(tcp + [0, 0, lift_height], quat),
                qprobe,
                mask=[False] * n_arm + [True, True],
                time_step=sample_dt,
                planning_time=3.0,
            )
            lift_rotation = r
            if lift["status"] != "Success":
                down = np.array([0.0, 0.0, -1.0])
                lift_rotation = np.column_stack(
                    [r[:, 0], np.cross(down, r[:, 0]), down]
                )
                lift = planner.plan_pose(
                    base_pose(
                        tcp + [0, 0, lift_height], matrix_to_quat_wxyz(lift_rotation)
                    ),
                    qprobe,
                    mask=[False] * n_arm + [True, True],
                    time_step=sample_dt,
                    planning_time=4.0,
                )
        else:
            lift = probe
        if lift["status"] != "Success":
            print("LIFT_DIAG", index, "probe", probe["status"], flush=True)
            if probe["status"] == "Success":
                planner.IK(
                    base_pose(
                        tcp + [0, 0, lift_height], matrix_to_quat_wxyz(lift_rotation)
                    ),
                    qprobe,
                    mask=[False] * n_arm + [True, True],
                    n_init_qpos=3,
                    return_closest=True,
                    verbose=True,
                )
            planner.detach_object("payload", also_remove=True)
            planner.planning_world.add_object(
                "target", fcl.CollisionObject(bvh(target_vertices, object_mesh.faces))
            )
            for side in ("soft_left", "soft_right"):
                acm.set_entry(side, "target", False)
            attempts.append(dict(index=index, phase="lift", status=lift["status"]))
            continue
        trajectories = {
            "pregrasp": approach["position"],
            "approach": descend["position"],
            "probe": probe["position"],
            "lift": lift["position"],
        }
        placement = None
        if placement_transform is not None:
            goal = placement_transform[:3, :3] @ tcp + placement_transform[:3, 3]
            goal_rotation = placement_transform[:3, :3] @ r
            goal_quat = matrix_to_quat_wxyz(goal_rotation)
            above = goal + [0, 0, 0.09]
            qlift = np.r_[lift["position"][-1], qclosed[-2:]]
            transfer = dict(status="Transfer changes orientation")
            if np.allclose(lift_rotation, goal_rotation, atol=1e-6):
                transfer = cartesian_segment(
                    qlift, tcp + [0, 0, lift_height], above, goal_quat
                )
            # A Cartesian IK chain can wind through a long joint-space route.
            # Compare an ordinary collision-checked transfer at the same joint
            # limits; shorter simulated motion also reduces capture cost.
            alternative = planner.plan_pose(
                base_pose(above, goal_quat),
                qlift,
                mask=[False] * n_arm + [True, True],
                time_step=sample_dt,
                planning_time=5.0,
            )
            if (
                transfer["status"] != "Success"
                or alternative["status"] == "Success"
                and len(alternative["position"]) < len(transfer["position"])
            ):
                transfer = alternative
            place = dict(status="Transfer: " + transfer["status"])
            retreat, home = place, place
            if transfer["status"] == "Success":
                place = cartesian_segment(
                    np.r_[transfer["position"][-1], qclosed[-2:]],
                    above,
                    goal,
                    goal_quat,
                )
                if place["status"] != "Success":
                    place = planner.plan_pose(
                        base_pose(goal, goal_quat),
                        np.r_[transfer["position"][-1], qclosed[-2:]],
                        mask=[False] * n_arm + [True, True],
                        time_step=sample_dt,
                        planning_time=4.0,
                    )
            if place["status"] == "Success":
                planner.detach_object("payload", also_remove=True)
                payload_attached = False
                placed_world = (
                    object_mesh.vertices @ placement_transform[:3, :3].T
                    + placement_transform[:3, 3]
                )
                placed_vertices = (placed_world - base_translation) @ base_rotation
                planner.planning_world.add_object(
                    "placed_target",
                    fcl.CollisionObject(bvh(placed_vertices, object_mesh.faces)),
                )
                acm.set_entry("placed_target", ["soft_left", "soft_right"], True)
                qplace = np.r_[place["position"][-1], qclosed[-2:]]
                qrelease = qplace.copy()
                qrelease[-2:] = candidate["opened"] / 2
                release_valid = all(
                    not (
                        planner.check_for_env_collision((1 - u) * qplace + u * qrelease)
                        + planner.check_for_self_collision(
                            (1 - u) * qplace + u * qrelease
                        )
                    )
                    for u in np.linspace(0, 1, 16)
                )
                # Once open, the placed block is an ordinary obstacle. A
                # retreat must not rely on permitted finger/payload contact.
                acm.set_entry("placed_target", ["soft_left", "soft_right"], False)
                retreat = (
                    cartesian_segment(
                        qrelease, goal, goal - goal_rotation[:, 2] * 0.09, goal_quat
                    )
                    if release_valid
                    else dict(status="Release collision")
                )
                if retreat["status"] == "Success":
                    acm.set_entry("placed_target", ["soft_left", "soft_right"], False)
                    home = planner.plan_qpos(
                        [q],
                        np.r_[retreat["position"][-1], qrelease[-2:]],
                        time_step=sample_dt,
                        planning_time=5.0,
                    )
            failed = next(
                (
                    (name, path["status"])
                    for name, path in (
                        ("transfer", transfer),
                        ("place", place),
                        ("retreat", retreat),
                        ("return_home", home),
                    )
                    if path["status"] != "Success"
                ),
                None,
            )
            if failed:
                attempts.append(dict(index=index, phase=failed[0], status=failed[1]))
                if payload_attached:
                    planner.detach_object("payload", also_remove=True)
                planner.remove_object("placed_target")
                planner.planning_world.add_object(
                    "target",
                    fcl.CollisionObject(bvh(target_vertices, object_mesh.faces)),
                )
                for side in ("soft_left", "soft_right"):
                    acm.set_entry(side, "target", False)
                continue
            trajectories.update(
                transfer=transfer["position"],
                place=place["position"],
                release=np.repeat(place["position"][-1:], 2, axis=0),
                retreat=retreat["position"],
                return_home=home["position"],
            )
            placement = dict(
                world_displacement=placement_transform.tolist(),
                tcp_goal_m=goal.tolist(),
                tcp_goal_rotation=goal_rotation.tolist(),
                release_gap_m=release_gap,
                support_geometry=str(Path(place_on).resolve()) if place_on else None,
                goal_pose_source=str(Path(place_pose).resolve())
                if place_pose
                else None,
                final_object_position=(
                    placement_transform[:3, :3]
                    @ np.asarray(scene["initial_object_position"])
                    + placement_transform[:3, 3]
                ).tolist(),
            )
        np.savez_compressed(out / "joints.npz", **trajectories)
        report = dict(
            status="planned",
            candidate=candidate,
            world_from_robot_base_translation_m=base_translation.tolist(),
            world_from_robot_base=world_from_base.tolist(),
            lift_height_m=lift_height,
            contact_height_m=height,
            finger_inner_radius_m=radius,
            hand_to_tcp_m=hand_to_tcp_m,
            initial_q=q.tolist(),
            scene=scene,
            trajectory_file="joints.npz",
            sample_dt_s=sample_dt,
            planner="mplib RRT pregrasp/lift + collision-checked Cartesian approach/probe; full shared STL + adapter/camera proxies + carried object",
            collision_checked=True,
            physical_grasp_validated=False,
            friction_coefficient=friction,
            probe_lift_m=0.012,
            lift_goal_rotation=lift_rotation.tolist(),
            attempts=attempts,
        )
        if placement is not None:
            report["placement"] = placement
            report["planner"] += (
                "; collision-checked transfer/place/release/retreat/return-home"
            )
        (out / "plan.json").write_text(json.dumps(report, indent=2))
        print(
            "SOFT_GRASP_PLAN",
            json.dumps(
                {k: v for k, v in report.items() if k not in ("scene", "attempts")}
            ),
            flush=True,
        )
        return
    (out / "attempts.json").write_text(json.dumps(attempts, indent=2))
    raise RuntimeError(
        "No collision-free pregrasp/approach/closed/lift combination: " + str(attempts)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene", type=Path, required=True, help="Exported scene directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Plan output directory"
    )
    parser.add_argument(
        "--contact-height",
        type=float,
        default=None,
        help="Contact height in meters on the scaled finger; default 55 mm times scale",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.7,
        help="Conservative friction coefficient for contact-normal screening",
    )
    parser.add_argument("--indent", type=float, default=0.003)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument(
        "--lift-height",
        type=float,
        default=0.10,
        help="TCP lift in meters; use margin above a task's object-lift threshold",
    )
    parser.add_argument(
        "--place-on",
        type=Path,
        help="World-space support geometry NPZ; extend pickup with placement and return home",
    )
    parser.add_argument(
        "--place-pose",
        type=Path,
        help="JSON world_from_object SE(3) goal, including object rotation",
    )
    parser.add_argument("--release-gap", type=float, default=0.002)
    parser.add_argument(
        "--sample-dt", type=float, default=0.02, help="Planner sample period, seconds"
    )
    args = parser.parse_args(argv)
    plan_scene(
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


if __name__ == "__main__":
    raise SystemExit(main())
