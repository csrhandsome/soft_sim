"""Audit recorded active X5 FEM and real fixtures against the task geometry.

Export states with the dataset environment, then check with the planning
environment. Task object poses come from each frame. Other scene obstacles
and the parked arm use their exported geometry; their motion is not certified.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation



def pose_matrix(position, quaternion):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_matrix()
    pose[:3, 3] = position
    return pose


def export_states(run, destination):
    from soft_sim.recording.reader import open_frame
    from soft_sim.recording.reader import capture_directory
    recording = capture_directory(run)
    episode = json.loads((recording / "episode.json").read_text())
    if episode["status"] not in ("committed", "complete"):
        raise ValueError("A complete recording is required")
    destination.mkdir(parents=True, exist_ok=False)
    for row in episode["frames"]:
        with open_frame(recording, row, include_images=False) as frame:
            names = ("finger_nodes_left", "finger_nodes_right", "finger_mounts", "fixture_to_world",
                     "joint_positions", "task_object_positions", "task_object_orientations_wxyz", "operation_index")
            values = {name: frame[name] for name in names}
            if "task_object_valid" in frame:
                values["task_object_valid"] = frame["task_object_valid"]
            np.savez(destination / f"{row['index']:06d}.npz", **values)
    (destination / "episode.json").write_text(json.dumps(episode, indent=2))


def fixture_slot(paths, link_name, component):
    """Fixture slot by link/component suffix; the actor prefix is arbitrary."""
    suffix = f"/{link_name}/SoftFixtures/{component}"
    matches = [index for index, path in enumerate(paths) if path.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one fixture slot for {link_name}/{component}, got {matches}")
    return matches[0]


def same_vertices(first, second):
    if first.shape != second.shape:
        return False
    ordered = [vertices[np.lexsort(vertices.T)] for vertices in (first, second)]
    return bool(np.allclose(*ordered, atol=1e-7, rtol=0))


def task_object_present(frame, index):
    """LeRobot can decode a one-object validity mask as a scalar."""
    if "task_object_valid" not in frame:
        return True
    return bool(np.asarray(frame["task_object_valid"]).reshape(-1)[index])


def resolve_operation_arm(episode, plans, operation_index, scene=None):
    """Resolve the active arm index for an operation, with legacy single-arm fallback."""
    if plans and 0 <= operation_index < len(plans):
        op_arm = plans[operation_index].get("active_arm")
        if op_arm is not None:
            return int(op_arm)
    if scene is not None and scene.get("active_arm") is not None:
        return int(scene["active_arm"])
    return int(episode.get("calibration", {}).get("active_arm", 0))


def finger_reference_for_arm(reference, arm_index, side):
    """Return (rest_nodes, pin_mask, tets) for the given arm and side, with legacy fallback."""
    prefix = f"robot{int(arm_index)}_" if arm_index is not None else ""
    if f"{prefix}finger_reference_{side}" not in reference.files:
        prefix = ""
    tets_key = f"{prefix}finger_tets_{side}"
    return (
        reference[f"{prefix}finger_reference_{side}"],
        reference[f"{prefix}finger_pin_{side}"],
        reference[tets_key] if tets_key in reference.files else None,
    )


def check(run, states):
    import trimesh
    from mplib.collision_detection import fcl
    from soft_sim.validation.installation import collision_object, wrist_obstacles
    from soft_sim.robots.fingers import fixture_meshes, finger_in_link, link_transforms, scene_joint_vector
    from soft_sim.robots.mechanism import mechanism
    from soft_sim.mesh import surface_faces

    episode = json.loads((states / "episode.json").read_text())
    from soft_sim.recording.reader import capture_directory, first_existing
    capture = capture_directory(run)
    scene_file = first_existing(run / "plans/operation_000/scene.json", run / "scene.json")
    scene_dir = scene_file.parent
    scene = json.loads(scene_file.read_text())
    recorded_scene = dict(episode["calibration"]["task_scene"])
    recorded_scene.pop("layout_snapshot", None)
    if recorded_scene != scene:
        raise ValueError("Collision states must match the scene export")
    installation, scale = scene["finger_installation"], scene["finger_scale"]
    arm = mechanism(scene["robot_name"])
    joint_names = episode["calibration"]["joint_names"]
    fallback_fixtures = episode["calibration"]["fixture_paths"]
    all_fixture_paths = episode["calibration"].get("all_fixture_paths")
    plans = json.loads(first_existing(run / "plans/executed.json", run / "executed_plans.json").read_text())
    objects, initial_hulls = {}, []
    for label, entry in scene["task_objects"].items():
        with np.load(scene_dir / entry["geometry_file"]) as geometry:
            world = trimesh.Trimesh(geometry["vertices"], geometry["faces"], process=False)
        if label in episode["calibration"]["task_object_labels"]:
            initial_hulls.extend([world.vertices, world.convex_hull.vertices])
        world.apply_transform(np.linalg.inv(pose_matrix(entry["position"], entry["quaternion_wxyz"])))
        objects[label] = world
    static = {}
    with np.load(scene_dir / "obstacle_geometry.npz") as obstacles:
        for index in range(int(obstacles["count"])):
            vertices = obstacles[f"vertices_{index}"]
            if any(same_vertices(vertices, hull) for hull in initial_hulls):
                continue
            static[f"obstacle_{index}"] = collision_object(trimesh.Trimesh(
                vertices, obstacles[f"faces_{index}"], process=False))
    table = trimesh.creation.box(extents=scene["table_size_m"])
    table.apply_translation(scene["table_center_m"])
    static["table"] = collision_object(table)
    expected_contacts, intersections = Counter(), []
    maximum_post_step_translation_m = 0.
    maximum_post_step_rotation_error = 0.
    with np.load(capture / "finger_reference.npz") as reference:
        def build_context(op_scene, op_static, operation_index):
            op_arm = resolve_operation_arm(episode, plans, operation_index, op_scene)
            op_mechanism = mechanism(op_scene.get("robot_name", scene["robot_name"]))
            op_fixtures = (
                all_fixture_paths[op_arm]
                if all_fixture_paths and op_arm < len(all_fixture_paths)
                else fallback_fixtures
            )
            return dict(
                arm_index=op_arm,
                mechanism=op_mechanism,
                installation=op_scene.get("finger_installation", installation),
                scale=op_scene.get("finger_scale", scale),
                wrists=wrist_obstacles(op_mechanism),
                joint_indices=[index for index, name in enumerate(joint_names) if name.startswith(f"robot{op_arm}/")],
                fixtures=op_fixtures,
                boundaries=[surface_faces(finger_reference_for_arm(reference, op_arm, side)[2]) for side in ("left", "right")],
                static=op_static,
            )

        context_by_operation = {0: build_context(scene, static, 0)}
        for row in episode["frames"]:
            with np.load(states / f"{row['index']:06d}.npz") as frame:
                operation_index = int(np.asarray(frame["operation_index"]).reshape(-1)[0])
                if operation_index not in context_by_operation:
                    directory = run / "plans" / f"operation_{operation_index:03d}"
                    if not (directory / "scene.json").is_file():
                        raise ValueError("Missing scene snapshot for recorded operation")
                    current_scene = json.loads((directory / "scene.json").read_text())
                    dynamic_vertices = []
                    for label in episode["calibration"]["task_object_labels"]:
                        if label not in current_scene["task_objects"]:
                            continue
                        with np.load(directory / current_scene["task_objects"][label]["geometry_file"]) as geometry:
                            mesh = trimesh.Trimesh(geometry["vertices"], geometry["faces"], process=False)
                        dynamic_vertices.extend([mesh.vertices, mesh.convex_hull.vertices])
                    current_static = {"table": static["table"]}
                    with np.load(directory / "obstacle_geometry.npz") as geometry:
                        for index in range(int(geometry["count"])):
                            vertices = geometry[f"vertices_{index}"]
                            if any(same_vertices(vertices, dynamic) for dynamic in dynamic_vertices):
                                continue
                            current_static[f"obstacle_{index}"] = collision_object(trimesh.Trimesh(
                                vertices, geometry[f"faces_{index}"], process=False))
                    context_by_operation[operation_index] = build_context(current_scene, current_static, operation_index)
                ctx = context_by_operation[operation_index]
                op_arm_mech = ctx["mechanism"]
                op_installation = ctx["installation"]
                op_scale = ctx["scale"]
                op_joint_indices = ctx["joint_indices"]
                op_fixtures = ctx["fixtures"]
                op_boundaries = ctx["boundaries"]
                mounts = frame["finger_mounts"]
                q = frame["joint_positions"][op_joint_indices]
                if op_arm_mech.joint_names != tuple(joint_names[index].split("/", 1)[-1] for index in op_joint_indices):
                    names = [joint_names[index].split("/", 1)[-1] for index in op_joint_indices]
                    q = q[[names.index(name) for name in op_arm_mech.joint_names]]
                links = link_transforms(q, mechanism=op_arm_mech)
                first = op_arm_mech.finger_links[0]
                first_fixture = fixture_slot(op_fixtures, first, "adapter")
                wrist_pose = frame["fixture_to_world"][first_fixture] @ np.linalg.inv(
                    finger_in_link(0, op_installation, op_arm_mech)) @ np.linalg.inv(
                    np.linalg.inv(links[op_arm_mech.hand_link]) @ links[first])
                obstacles = dict(ctx["static"])
                for name, mesh in ctx["wrists"].items():
                    mesh = mesh.copy()
                    mesh.apply_transform(wrist_pose)
                    obstacles[name] = collision_object(mesh)
                for index, label in enumerate(episode["calibration"]["task_object_labels"]):
                    if not task_object_present(frame, index):
                        continue
                    mesh = objects[label].copy()
                    mesh.apply_transform(pose_matrix(frame["task_object_positions"][index],
                                                    frame["task_object_orientations_wxyz"][index]))
                    obstacles[label] = collision_object(mesh)
                moving = {}
                for side, name in enumerate(("left", "right")):
                    moving[(side, "finger")] = collision_object(trimesh.Trimesh(
                        frame[f"finger_nodes_{name}"], op_boundaries[side], process=False))
                    for component, mesh in fixture_meshes(op_scale, side, op_installation, op_arm_mech).items():
                        parent = op_arm_mech.finger_links[side]
                        slot = fixture_slot(op_fixtures, parent, component)
                        actual = frame["fixture_to_world"][slot]
                        maximum_post_step_translation_m = max(maximum_post_step_translation_m,
                            float(np.linalg.norm(actual[:3, 3]-mounts[side, :3, 3])))
                        maximum_post_step_rotation_error = max(maximum_post_step_rotation_error,
                            float(np.max(np.abs(actual[:3, :3]-mounts[side, :3, :3]))))
                        mesh.apply_transform(actual @ np.linalg.inv(finger_in_link(side, op_installation, op_arm_mech)))
                        moving[(side, component)] = collision_object(mesh)
                target = plans[operation_index]["target_label"]
                for key, body in moving.items():
                    side, component = key
                    for name, obstacle in obstacles.items():
                        if fcl.collide(body, obstacle).is_collision():
                            pair = f"{side}_{component}/{name}"
                            if component == "finger" and name == target:
                                expected_contacts[pair] += 1
                            else:
                                intersections.append(dict(index=row["index"], time_s=row["time_s"],
                                    phase=row["phase"], pair=pair))
                    for other_key, other in moving.items():
                        if key >= other_key or (side == other_key[0] and {component, other_key[1]} == {"finger", "adapter"}):
                            continue
                        if fcl.collide(body, other).is_collision():
                            pair = f"{side}_{component}/{other_key[0]}_{other_key[1]}"
                            if component == other_key[1] == "finger":
                                expected_contacts[pair] += 1
                            else:
                                intersections.append(dict(index=row["index"], time_s=row["time_s"],
                                    phase=row["phase"], pair=pair))
    report = dict(valid=bool(episode["frames"]) and not intersections, frame_count=len(episode["frames"]),
        maximum_post_step_fixture_to_fem_target_translation_m=maximum_post_step_translation_m,
        maximum_post_step_fixture_to_fem_target_rotation_matrix_error=maximum_post_step_rotation_error,
        expected_contacts=dict(expected_contacts), unexpected_intersections=intersections,
        scope="Every recorded active-arm FEM boundary and actual fixture pose versus table, own wrist, moving task objects, "
              "and exported obstacle surfaces. Other scene obstacles and parked arm are frozen at each operation export. "
              "Own finger/wedge attachment and soft finger/active payload or opposing finger contact are expected. "
              "Fixture/FEM target offsets compare post-step rigid poses to pre-step prescribed targets; "
              "they are coupling epoch offsets, not same-epoch mount errors. "
              "Surface intersections only; no continuous-time or containment certificate.")
    (capture / "task_collision_validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--export-states", action="store_true")
    args = parser.parse_args()
    if args.export_states:
        export_states(args.run, args.states)
    else:
        report = check(args.run, args.states)
        print(json.dumps({key: value for key, value in report.items() if key != "unexpected_intersections"}, indent=2))
        print("Unexpected surface intersections:", len(report["unexpected_intersections"]))
        raise SystemExit(0 if report["valid"] else 1)
