import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from soft_sim.robots.fingers import finger_in_link, fixture_meshes, measured_installation, mounts_from_q
from soft_sim.robots.mechanism import mechanism
from soft_sim.validation.collisions import check as check_collisions


def _build_fixture_paths(arm_index, arm):
    paths = []
    for link in arm.finger_links:
        for component in ("adapter", "mount", "camera", "wedge"):
            paths.append(f"/World/robot{arm_index}/{link}/SoftFixtures/{component}")
    return paths


def _build_fixture_to_world(q, base, inst, arm):
    mounts = mounts_from_q(q, base, inst, arm, joint_names=arm.joint_names)
    poses = []
    for side in (0, 1):
        for _ in ("adapter", "mount", "camera", "wedge"):
            poses.append(mounts[side])
    return np.stack(poses)


def _create_synthetic_run(root, *, multi_arm=True):
    arm = mechanism("x5")
    inst = measured_installation(1.0, mechanism=arm)
    base_0 = np.eye(4)
    base_0[:3, 3] = [-0.30, -0.45, 0.765]
    base_1 = np.eye(4)
    base_1[:3, 3] = [0.30, -0.45, 0.765]
    bases = [base_0, base_1]

    q0 = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.044, 0.044])
    q1 = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.044, 0.044])
    q_by_arm = [q0, q1]

    run_dir = root / "run"
    capture_dir = run_dir / "capture"
    states_dir = root / "states"
    (run_dir / "plans/operation_000").mkdir(parents=True)
    (run_dir / "plans/operation_001").mkdir(parents=True)
    capture_dir.mkdir(parents=True)
    states_dir.mkdir(parents=True)

    # Simple 4-node tetrahedron finger in the distal region above the camera/mount.
    local_nodes = np.array([
        [0.0, 0.0, 0.050],
        [0.005, 0.0, 0.050],
        [0.0, 0.005, 0.050],
        [0.0, 0.0, 0.060],
    ])
    pin_mask = np.array([True, False, False, False])
    tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
    ref_kwargs = dict(
        finger_reference_left=local_nodes,
        finger_reference_right=local_nodes,
        finger_pin_left=pin_mask,
        finger_pin_right=pin_mask,
        finger_tets_left=tets,
        finger_tets_right=tets,
    )
    if multi_arm:
        for idx in (0, 1):
            ref_kwargs[f"robot{idx}_finger_reference_left"] = local_nodes
            ref_kwargs[f"robot{idx}_finger_reference_right"] = local_nodes
            ref_kwargs[f"robot{idx}_finger_pin_left"] = pin_mask
            ref_kwargs[f"robot{idx}_finger_pin_right"] = pin_mask
            ref_kwargs[f"robot{idx}_finger_tets_left"] = tets
            ref_kwargs[f"robot{idx}_finger_tets_right"] = tets
    np.savez(capture_dir / "finger_reference.npz", **ref_kwargs)

    # Task objects placed on table away from fingers.
    box = trimesh.creation.box(extents=[0.03, 0.03, 0.06])
    obj0_pos = [0.15, -0.22, 0.80]
    obj1_pos = [-0.26, -0.11, 0.80]

    def write_scene(op_dir, op_arm, target_label):
        task_objects = {}
        for label, pos in (("tube0", obj0_pos), ("tube1", obj1_pos)):
            m = box.copy()
            m.apply_translation(pos)
            fname = f"task_object_{label}.npz"
            np.savez(op_dir / fname, vertices=m.vertices, faces=m.faces)
            task_objects[label] = {
                "prim_path": f"/World/{label}",
                "position": pos,
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "geometry_file": fname,
                "functional_frames": {},
            }
        # Static obstacle at the parked arm's base (far from active arm).
        obs = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
        obs.apply_translation(bases[1 - op_arm][:3, 3] + np.array([0.0, 0.0, 0.05]))
        np.savez(op_dir / "obstacle_geometry.npz", count=1, vertices_0=obs.vertices, faces_0=obs.faces)
        scene = {
            "task": "insert_tubes",
            "robot_name": "x5",
            "active_arm": op_arm,
            "finger_scale": 1.0,
            "finger_installation": inst,
            "table_center_m": [0.0, -0.05, 0.74],
            "table_size_m": [1.4, 1.1, 0.05],
            "target_label": target_label,
            "world_from_robot_base": bases[op_arm].tolist(),
            "task_objects": task_objects,
        }
        (op_dir / "scene.json").write_text(json.dumps(scene))
        return scene

    scene0 = write_scene(run_dir / "plans/operation_000", 1, "tube0")
    op1_arm = 0 if multi_arm else 1
    write_scene(run_dir / "plans/operation_001", op1_arm, "tube1")

    executed_plans = [
        {"operation_index": 0, "target_label": "tube0", **({"active_arm": 1} if multi_arm else {})},
        {"operation_index": 1, "target_label": "tube1", **({"active_arm": op1_arm} if multi_arm else {})},
    ]
    (run_dir / "plans/executed.json").write_text(json.dumps(executed_plans))

    joint_names = [f"robot{i}/{name}" for i in (0, 1) for name in arm.joint_names]
    fixture_paths_by_arm = [_build_fixture_paths(0, arm), _build_fixture_paths(1, arm)]
    calibration = {
        "active_arm": 1,
        "joint_names": joint_names,
        "fixture_paths": fixture_paths_by_arm[1],
        "task_object_labels": ["tube0", "tube1"],
        "task_scene": scene0,
    }
    if multi_arm:
        calibration["all_fixture_paths"] = fixture_paths_by_arm

    joint_positions = np.concatenate([q0, q1])
    task_obj_pos = np.array([obj0_pos, obj1_pos])
    task_obj_quat = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    frames_meta = []
    for frame_idx, (op_idx, op_arm) in enumerate(((0, 1), (1, op1_arm))):
        mounts = mounts_from_q(q_by_arm[op_arm], bases[op_arm], inst, arm, joint_names=arm.joint_names)
        fixture_to_world = _build_fixture_to_world(q_by_arm[op_arm], bases[op_arm], inst, arm)
        nodes_l = local_nodes @ mounts[0, :3, :3].T + mounts[0, :3, 3]
        nodes_r = local_nodes @ mounts[1, :3, :3].T + mounts[1, :3, 3]
        np.savez(
            states_dir / f"{frame_idx:06d}.npz",
            operation_index=np.array(op_idx),
            finger_mounts=mounts,
            fixture_to_world=fixture_to_world,
            joint_positions=joint_positions,
            finger_nodes_left=nodes_l,
            finger_nodes_right=nodes_r,
            task_object_positions=task_obj_pos,
            task_object_orientations_wxyz=task_obj_quat,
            task_object_valid=np.array([1, 1], dtype=np.uint8),
        )
        frames_meta.append({"index": frame_idx, "time_s": frame_idx * 0.1, "phase": "pregrasp"})

    episode = {"status": "complete", "calibration": calibration, "frames": frames_meta}
    (capture_dir / "episode.json").write_text(json.dumps(episode))
    (states_dir / "episode.json").write_text(json.dumps(episode))
    return run_dir, states_dir


class PerOperationArmCollisionTests(unittest.TestCase):
    def test_cross_operation_arm_switch_collision_check_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, states_dir = _create_synthetic_run(Path(tmp), multi_arm=True)
            report = check_collisions(run_dir, states_dir)
            self.assertTrue(report["valid"], msg=report["unexpected_intersections"])
            self.assertEqual(report["frame_count"], 2)
            self.assertLess(report["maximum_post_step_fixture_to_fem_target_translation_m"], 1e-9)

    def test_legacy_single_arm_fallback_collision_check_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, states_dir = _create_synthetic_run(Path(tmp), multi_arm=False)
            report = check_collisions(run_dir, states_dir)
            self.assertTrue(report["valid"], msg=report["unexpected_intersections"])
            self.assertEqual(report["frame_count"], 2)
