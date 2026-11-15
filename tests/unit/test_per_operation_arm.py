import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from soft_sim.config import ExperimentConfig
from soft_sim.execution import phase_commands
from soft_sim.robots.fingers import finger_in_link, fixture_meshes, measured_installation, mounts_from_q
from soft_sim.robots.mechanism import mechanism
from soft_sim.sim.scene import SceneExporter
from soft_sim.tasks.collect import select_arm_for_position
from soft_sim.validation.collisions import finger_reference_for_arm, resolve_operation_arm
from soft_sim.validation.finger_motion import check as check_finger_motion


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeRobot:
    def __init__(self, arm, joint_pos, body_poses):
        self.joint_names = arm.joint_names
        self.body_names = (arm.base_link, arm.hand_link, *arm.finger_links)
        self.data = SimpleNamespace(
            joint_pos=[_FakeTensor(joint_pos)],
            body_link_pose_w=[_FakeTensor(body_poses)],
        )


class _FakeObject:
    def __init__(self, prim_path, position, quaternion_wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.prim_path = prim_path
        self._pose = (np.asarray(position, dtype=float), np.asarray(quaternion_wxyz, dtype=float))

    def get_world_pose(self):
        return self._pose


class _FakeLayoutManager:
    def __init__(self, objects):
        self._objects = objects

    def get_instance_name(self, label, env_idx=0):
        return label

    def get_scene_object(self, inst_name, env_idx=0):
        return self._objects[inst_name]

    def get_instance_metadata(self, inst_name, env_idx=0):
        return {}


class PerOperationArmUnitTests(unittest.TestCase):
    def test_auto_arm_sequence_and_explicit_override(self):
        # Actual insert_tubes seed0 tube0/tube1/tube2 world X coordinates.
        positions = [
            np.array([0.1520, -0.2203, 0.7783]),
            np.array([-0.2588, -0.1103, 0.7783]),
            np.array([0.3714, -0.1073, 0.7809]),
        ]
        self.assertEqual([select_arm_for_position(p) for p in positions], [1, 0, 1])
        self.assertEqual(
            [select_arm_for_position(p, explicit_active_arm=0) for p in positions],
            [0, 0, 0],
        )
        self.assertEqual(
            [select_arm_for_position(p, explicit_active_arm=1) for p in positions],
            [1, 1, 1],
        )

    def test_scene_exporter_set_active_arm_updates_base_target_and_parked_obstacle(self):
        arm = mechanism("x5")
        inst = measured_installation(1.0, mechanism=arm)
        mechanisms = [arm, arm]
        installations = [inst, inst]
        cfg = ExperimentConfig.load()
        base_0 = np.eye(4)
        base_0[:3, 3] = [-0.30, -0.45, 0.765]
        base_1 = np.eye(4)
        base_1[:3, 3] = [0.30, -0.45, 0.765]

        q0 = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.044, 0.044])
        q1 = np.array([-0.1, 0.4, 0.5, 0.0, 0.0, 0.0, 0.030, 0.030])

        def make_body_poses(base_pose):
            poses = []
            for idx, _ in enumerate((arm.base_link, arm.hand_link, *arm.finger_links)):
                pos = base_pose[:3, 3] + np.array([0.0, 0.0, 0.01 * idx])
                poses.append(np.r_[pos, [1.0, 0.0, 0.0, 0.0]])
            return np.asarray(poses)

        robots = [
            _FakeRobot(arm, q0, make_body_poses(base_0)),
            _FakeRobot(arm, q1, make_body_poses(base_1)),
        ]
        mounts_0 = mounts_from_q(q0, base_0, inst, arm, joint_names=arm.joint_names)
        mounts_1 = mounts_from_q(q1, base_1, inst, arm, joint_names=arm.joint_names)

        def make_tool(mounts, offset_z):
            nodes = np.array([
                [0.0, 0.0, offset_z],
                [0.02, 0.0, offset_z],
                [0.0, 0.02, offset_z],
                [0.0, 0.0, offset_z + 0.02],
            ])
            return SimpleNamespace(
                mounts=mounts,
                positions=lambda: [nodes + mounts[0, :3, 3], nodes + mounts[1, :3, 3]],
                mesh_settings={"dummy": True},
                mesh_topology={"dummy": True},
            )

        tools = [make_tool(mounts_0, 0.0), make_tool(mounts_1, 0.5)]
        objects = {
            "tube0": _FakeObject("/World/tube0", [0.1520, -0.2203, 0.7783]),
            "tube1": _FakeObject("/World/tube1", [-0.2588, -0.1103, 0.7783]),
        }
        layout = {
            "task": "insert_tubes",
            "objects": {
                "tube0": {"shape": "cylinder", "dimensions": [0.02, 0.08]},
                "tube1": {"shape": "cylinder", "dimensions": [0.02, 0.08]},
            },
        }
        env = SimpleNamespace(
            stage=object(),
            tools=tools,
            removed_finger_colliders={},
            robot_manager=SimpleNamespace(robot_key=robots),
            scene_manager=SimpleNamespace(
                layout_manager=_FakeLayoutManager(objects),
                get_objects=lambda env_ids: objects,
            ),
        )

        def fake_world_geometry(stage, path, world_from_root=None):
            # Encode robot index / prim identity in a simple tetrahedron transformed by world_from_root.
            local = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
            if world_from_root is not None:
                local = local @ world_from_root[:3, :3].T + world_from_root[:3, 3]
            faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
            return local, faces

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exporter = SceneExporter(
                env=env,
                cfg=cfg,
                layout=layout,
                layout_path=root / "layout.json",
                layout_hash="abc",
                installation=inst,
                base=base_1,
                arm_index=1,
                mechanisms=mechanisms,
                installations=installations,
                fem_report=[],
                fem_valid=True,
                finger_scale=1.0,
                adapter_offset_m=0.0,
                texture_max_resolution=256,
                task_name="insert_tubes",
            )
            with patch("soft_sim.sim.scene.world_geometry", side_effect=fake_world_geometry), \
                 patch("soft_sim.sim.scene.SceneExporter._numpy_pose", side_effect=lambda self_or_pose, pose=None: pose if pose is not None else self_or_pose):
                # _numpy_pose is a method: (self, pose) -> pose
                pass

        # Re-run cleanly with proper method patch on _numpy_pose
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exporter = SceneExporter(
                env=env,
                cfg=cfg,
                layout=layout,
                layout_path=root / "layout.json",
                layout_hash="abc",
                installation=inst,
                base=base_1,
                arm_index=1,
                mechanisms=mechanisms,
                installations=installations,
                fem_report=[],
                fem_valid=True,
                finger_scale=1.0,
                adapter_offset_m=0.0,
                texture_max_resolution=256,
                task_name="insert_tubes",
            )
            with patch("soft_sim.sim.scene.world_geometry", side_effect=fake_world_geometry), \
                 patch("soft_sim.sim.robodojo.robot_link_path", side_effect=lambda stage, idx, name: f"/World/robot{idx}/{name}"), \
                 patch.object(SceneExporter, "_numpy_pose", side_effect=lambda pose: pose):
                scene_0, _, _ = exporter.export(root / "op0", "tube0")
                self.assertEqual(scene_0["active_arm"], 1)
                self.assertEqual(scene_0["target_label"], "tube0")
                np.testing.assert_allclose(scene_0["world_from_robot_base"], base_1)
                np.testing.assert_allclose(scene_0["initial_q"], q1)
                with np.load(root / "op0/obstacle_geometry.npz") as obs0:
                    # Parked arm 0 base x = -0.30 should appear in obstacle_geometry, not arm 1 (+0.30).
                    xs0 = [float(obs0[f"vertices_{i}"][:, 0].mean()) for i in range(int(obs0["count"]))]
                    self.assertTrue(any(x < -0.1 for x in xs0))

                exporter.set_active_arm(0, base_0)
                scene_1, _, _ = exporter.export(root / "op1", "tube1")
                self.assertEqual(scene_1["active_arm"], 0)
                self.assertEqual(scene_1["target_label"], "tube1")
                np.testing.assert_allclose(scene_1["world_from_robot_base"], base_0)
                np.testing.assert_allclose(scene_1["initial_q"], q0)
                with np.load(root / "op1/obstacle_geometry.npz") as obs1:
                    # Parked arm 1 base x = +0.30 should now appear in obstacle_geometry.
                    xs1 = [float(obs1[f"vertices_{i}"][:, 0].mean()) for i in range(int(obs1["count"]))]
                    self.assertTrue(any(x > 0.1 for x in xs1))

    def test_per_arm_last_width_independence(self):
        arm = mechanism("x5")
        last_widths = [arm.max_open_width_m, arm.max_open_width_m]
        traj = np.zeros((2, 8))
        # Arm 1 closes to 0.025m.
        _, _, widths_1 = phase_commands(traj, 5, 0.01, last_widths[1], 0.025)
        last_widths[1] = 0.025
        self.assertAlmostEqual(widths_1[-1], 0.025)
        # Arm 0 starts its first phase from its own untouched open width (~0.088m), not arm 1's 0.025m.
        _, _, widths_0 = phase_commands(traj, 5, 0.01, last_widths[0], 0.050)
        self.assertGreater(widths_0[0], 0.080)
        self.assertLess(widths_0[0], arm.max_open_width_m)

    def test_resolve_operation_arm_and_per_arm_finger_reference(self):
        episode_multi = {
            "calibration": {"active_arm": 1},
            "result": {"executed_plans": [{"operation_index": 0, "active_arm": 1}, {"operation_index": 1, "active_arm": 0}]},
        }
        self.assertEqual(resolve_operation_arm(episode_multi, episode_multi["result"]["executed_plans"], 0), 1)
        self.assertEqual(resolve_operation_arm(episode_multi, episode_multi["result"]["executed_plans"], 1), 0)

        episode_legacy = {"calibration": {"active_arm": 1}}
        self.assertEqual(resolve_operation_arm(episode_legacy, [{"operation_index": 0}], 0), 1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "finger_reference.npz"
            ref0 = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
            ref1 = ref0 * 2.0
            pin0 = np.array([True, False, False, False])
            pin1 = np.array([True, True, False, False])
            tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
            np.savez(
                path,
                finger_reference_left=ref1,
                finger_reference_right=ref1,
                finger_pin_left=pin1,
                finger_pin_right=pin1,
                finger_tets_left=tets,
                finger_tets_right=tets,
                robot0_finger_reference_left=ref0,
                robot0_finger_reference_right=ref0,
                robot0_finger_pin_left=pin0,
                robot0_finger_pin_right=pin0,
                robot0_finger_tets_left=tets,
                robot0_finger_tets_right=tets,
                robot1_finger_reference_left=ref1,
                robot1_finger_reference_right=ref1,
                robot1_finger_pin_left=pin1,
                robot1_finger_pin_right=pin1,
                robot1_finger_tets_left=tets,
                robot1_finger_tets_right=tets,
            )
            with np.load(path) as ref:
                r0, p0, _ = finger_reference_for_arm(ref, 0, "left")
                r1, p1, _ = finger_reference_for_arm(ref, 1, "left")
                np.testing.assert_allclose(r0, ref0)
                np.testing.assert_allclose(r1, ref1)
                np.testing.assert_array_equal(p0, pin0)
                np.testing.assert_array_equal(p1, pin1)

    def test_finger_motion_uses_per_arm_reference_across_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp)
            ref0 = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
            ref1 = np.vstack([ref0, [[0.01, 0.01, 0.01]]])  # different node count for robot1!
            pin0 = np.array([True, False, False, False])
            pin1 = np.array([True, False, False, False, False])
            tets = np.array([[0, 1, 2, 3]], dtype=np.int32)
            np.savez(
                rec / "finger_reference.npz",
                finger_reference_left=ref1,
                finger_reference_right=ref1,
                finger_pin_left=pin1,
                finger_pin_right=pin1,
                finger_tets_left=tets,
                finger_tets_right=tets,
                robot0_finger_reference_left=ref0,
                robot0_finger_reference_right=ref0,
                robot0_finger_pin_left=pin0,
                robot0_finger_pin_right=pin0,
                robot0_finger_tets_left=tets,
                robot0_finger_tets_right=tets,
                robot1_finger_reference_left=ref1,
                robot1_finger_reference_right=ref1,
                robot1_finger_pin_left=pin1,
                robot1_finger_pin_right=pin1,
                robot1_finger_tets_left=tets,
                robot1_finger_tets_right=tets,
            )
            mounts = np.stack([np.eye(4), np.eye(4)])
            cameras = np.stack([np.eye(4), np.eye(4)])
            np.savez(
                rec / "000000.npz",
                operation_index=np.array(0),
                finger_mounts=mounts,
                finger_camera_to_world=cameras,
                finger_nodes_left=ref1,
                finger_nodes_right=ref1,
            )
            np.savez(
                rec / "000001.npz",
                operation_index=np.array(1),
                finger_mounts=mounts,
                finger_camera_to_world=cameras,
                finger_nodes_left=ref0,
                finger_nodes_right=ref0,
            )
            episode = {
                "status": "complete",
                "fps": 10,
                "calibration": {"active_arm": 1},
                "result": {
                    "executed_plans": [
                        {"operation_index": 0, "active_arm": 1},
                        {"operation_index": 1, "active_arm": 0},
                    ]
                },
                "frames": [
                    {"index": 0, "time_s": 0.0, "phase": "settle", "file": "000000.npz"},
                    {"index": 1, "time_s": 0.1, "phase": "approach", "file": "000001.npz"},
                ],
            }
            (rec / "episode.json").write_text(json.dumps(episode))
            report = check_finger_motion(rec)
            self.assertTrue(report["valid"])
