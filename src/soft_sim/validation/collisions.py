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
    if episode["status"] != "complete":
        raise ValueError("A complete recording is required")
    destination.mkdir(parents=True, exist_ok=False)
    for row in episode["frames"]:
        with open_frame(recording, row, include_images=False) as frame:
            names = ("finger_nodes_left", "finger_nodes_right", "finger_mounts", "fixture_to_world",
                     "joint_positions", "task_object_positions", "task_object_orientations_wxyz", "operation_index")
            np.savez(destination / f"{row['index']:06d}.npz", **{name: frame[name] for name in names})
    (destination / "episode.json").write_text(json.dumps(episode, indent=2))


def same_vertices(first, second):
    if first.shape != second.shape:
        return False
    ordered = [vertices[np.lexsort(vertices.T)] for vertices in (first, second)]
    return bool(np.allclose(*ordered, atol=1e-7, rtol=0))


def check(run, states):
    import trimesh
    from mplib.collision_detection import fcl
    from soft_sim.validation.installation import collision_object, wrist_obstacles
    from soft_sim.robots.x5.mount_geometry import fixture_meshes
    from soft_sim.robots.x5.soft_tool import finger_in_link, link_transforms
    from soft_sim.mesh import surface_faces

    episode = json.loads((states / "episode.json").read_text())
    scene = json.loads((run / "scene.json").read_text())
    recorded_scene = dict(episode["calibration"]["task_scene"])
    recorded_scene.pop("layout_snapshot", None)
    if recorded_scene != scene:
        raise ValueError("Collision states must match the scene export")
    installation, scale = scene["finger_installation"], scene["finger_scale"]
    active = episode["calibration"]["active_arm"]
    joint_names = episode["calibration"]["joint_names"]
    joint_indices = [index for index, name in enumerate(joint_names) if name.startswith(f"robot{active}/")]
    fixtures = episode["calibration"]["fixture_paths"]
    with np.load(run / "recording/finger_reference.npz") as reference:
        boundaries = [surface_faces(reference[f"finger_tets_{side}"]) for side in ("left", "right")]
    objects, initial_hulls = {}, []
    for label, entry in scene["task_objects"].items():
        with np.load(run / entry["geometry_file"]) as geometry:
            world = trimesh.Trimesh(geometry["vertices"], geometry["faces"], process=False)
        initial_hulls.append(world.convex_hull.vertices)
        world.apply_transform(np.linalg.inv(pose_matrix(entry["position"], entry["quaternion_wxyz"])))
        objects[label] = world
    static = {}
    with np.load(run / "obstacle_geometry.npz") as obstacles:
        for index in range(int(obstacles["count"])):
            vertices = obstacles[f"vertices_{index}"]
            if any(same_vertices(vertices, hull) for hull in initial_hulls):
                continue
            static[f"obstacle_{index}"] = collision_object(trimesh.Trimesh(
                vertices, obstacles[f"faces_{index}"], process=False))
    table = trimesh.creation.box(extents=scene["table_size_m"])
    table.apply_translation(scene["table_center_m"])
    static["table"] = collision_object(table)
    wrists = wrist_obstacles()
    expected_contacts, intersections = Counter(), []
    maximum_post_step_translation_m = 0.
    maximum_post_step_rotation_error = 0.
    plans = json.loads((run / "executed_plans.json").read_text())
    for row in episode["frames"]:
        with np.load(states / f"{row['index']:06d}.npz") as frame:
            mounts = frame["finger_mounts"]
            links = link_transforms(frame["joint_positions"][joint_indices])
            first_fixture = fixtures.index(f"/World/envs/env_0/robot{active}/link8/SoftFixtures/adapter")
            wrist_pose = frame["fixture_to_world"][first_fixture] @ np.linalg.inv(finger_in_link(0, installation)) @ np.linalg.inv(
                np.linalg.inv(links["link6"]) @ links["link8"])
            obstacles = dict(static)
            for name, mesh in wrists.items():
                mesh = mesh.copy()
                mesh.apply_transform(wrist_pose)
                obstacles[name] = collision_object(mesh)
            for index, label in enumerate(episode["calibration"]["task_object_labels"]):
                mesh = objects[label].copy()
                mesh.apply_transform(pose_matrix(frame["task_object_positions"][index],
                                                frame["task_object_orientations_wxyz"][index]))
                obstacles[label] = collision_object(mesh)
            moving = {}
            for side, name in enumerate(("left", "right")):
                moving[(side, "finger")] = collision_object(trimesh.Trimesh(
                    frame[f"finger_nodes_{name}"], boundaries[side], process=False))
                for component, mesh in fixture_meshes(scale, side, installation).items():
                    parent = ("link8", "link7")[side]
                    slot = fixtures.index(f"/World/envs/env_0/robot{active}/{parent}/SoftFixtures/{component}")
                    actual = frame["fixture_to_world"][slot]
                    maximum_post_step_translation_m = max(maximum_post_step_translation_m,
                        float(np.linalg.norm(actual[:3, 3]-mounts[side, :3, 3])))
                    maximum_post_step_rotation_error = max(maximum_post_step_rotation_error,
                        float(np.max(np.abs(actual[:3, :3]-mounts[side, :3, :3]))))
                    mesh.apply_transform(actual @ np.linalg.inv(finger_in_link(side, installation)))
                    moving[(side, component)] = collision_object(mesh)
            target = plans[int(frame["operation_index"])]["target_label"]
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
              "and exported obstacle surfaces. Other scene obstacles and parked arm are frozen at export. "
              "Own finger/wedge attachment and soft finger/active payload or opposing finger contact are expected. "
              "Fixture/FEM target offsets compare post-step rigid poses to pre-step prescribed targets; "
              "they are coupling epoch offsets, not same-epoch mount errors. "
              "Surface intersections only; no continuous-time or containment certificate.")
    (run / "recording/task_collision_validation.json").write_text(json.dumps(report, indent=2))
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
