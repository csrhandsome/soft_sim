"""Unit tests for scene-aware cuRobo world collision loading and caching."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from soft_sim.planning.world_collision import (
    clear_scene_planner_cache,
    get_scene_planner,
    load_scene_world_spec,
    transform_world_vertices_to_base,
)


def _write_test_scene(
    scene_dir: Path,
    world_from_robot_base: np.ndarray,
    vertices_world: np.ndarray,
    faces: np.ndarray,
    table_z: float = 0.765,
    active_arm: int = 1,
) -> None:
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene = {
        "task": "insert_tubes",
        "robot_name": "x5",
        "active_arm": active_arm,
        "target_label": "tube0",
        "table_z": table_z,
        "world_from_robot_base": world_from_robot_base.tolist(),
    }
    (scene_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    np.savez(
        scene_dir / "obstacle_geometry.npz",
        count=np.array(1, dtype=np.int32),
        vertices_0=vertices_world.astype(np.float32),
        faces_0=faces.astype(np.int32),
    )


def _hollow_frame_mesh(center_world: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Build a non-convex square ring (outer 0.2m, inner hole 0.08m) at z=center_z..+0.05."""
    cx, cy, cz = center_world
    outer = 0.10
    inner = 0.04
    pts_2d = [
        (-outer, -outer),
        (outer, -outer),
        (outer, outer),
        (-outer, outer),
        (-inner, -inner),
        (inner, -inner),
        (inner, inner),
        (-inner, inner),
    ]
    verts = []
    for dz in (0.0, 0.05):
        for dx, dy in pts_2d:
            verts.append([cx + dx, cy + dy, cz + dz])
    # Top annular ring triangles (indices 8..15) + inner wall triangles
    faces = [
        [8, 9, 13],
        [8, 13, 12],
        [9, 10, 14],
        [9, 14, 13],
        [10, 11, 15],
        [10, 15, 14],
        [11, 8, 12],
        [11, 12, 15],
        # Inner hole vertical wall
        [4, 5, 13],
        [4, 13, 12],
        [5, 6, 14],
        [5, 14, 13],
    ]
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


class TestCuroboWorld(unittest.TestCase):
    def setUp(self) -> None:
        clear_scene_planner_cache()

    def tearDown(self) -> None:
        clear_scene_planner_cache()

    def test_world_to_base_transform_left_and_right_arms(self) -> None:
        # Right arm base rotated +90 deg around Z at (0.3, -0.45, 0.765)
        right_base = np.array(
            [
                [0.0, -1.0, 0.0, 0.30],
                [1.0, 0.0, 0.0, -0.45],
                [0.0, 0.0, 1.0, 0.765],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Left arm base rotated -90 deg around Z at (-0.3, -0.45, 0.765)
        left_base = np.array(
            [
                [0.0, 1.0, 0.0, -0.30],
                [-1.0, 0.0, 0.0, -0.45],
                [0.0, 0.0, 1.0, 0.765],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        p_world = np.array([[0.30, -0.25, 0.865]], dtype=np.float64)
        # In right_base: p_world - t = (0, 0.2, 0.1); R^T @ (0, 0.2, 0.1) = (0.2, 0.0, 0.1)
        np.testing.assert_allclose(
            transform_world_vertices_to_base(p_world, right_base),
            np.array([[0.20, 0.0, 0.10]], dtype=np.float32),
            atol=1e-6,
        )
        # In left_base: p_world - t = (0.6, 0.2, 0.1); R^T @ (0.6, 0.2, 0.1) = (-0.2, 0.6, 0.1)
        np.testing.assert_allclose(
            transform_world_vertices_to_base(p_world, left_base),
            np.array([[-0.20, 0.60, 0.10]], dtype=np.float32),
            atol=1e-6,
        )

    def test_preserves_non_convex_holes_and_serializable_report(self) -> None:
        verts, faces = _hollow_frame_mesh((0.0, 0.02, 0.765))
        base = np.eye(4, dtype=np.float64)
        base[2, 3] = 0.765
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "op0"
            _write_test_scene(scene_dir, base, verts, faces)
            spec = load_scene_world_spec(scene_dir)

        obs = spec["meshes"]["obstacle_0"]
        self.assertEqual(len(obs["vertices"]), len(verts))
        self.assertEqual(len(obs["faces"]), len(faces))
        # Inner-hole vertices (indices 4..7, 12..15) must remain inside the outer hull
        v_arr = np.asarray(obs["vertices"], dtype=np.float32)
        self.assertTrue(np.any(np.abs(v_arr[:, 0]) < 0.05))
        self.assertAlmostEqual(spec["table_height"], -0.025, places=6)

        report = spec["world_report"]
        self.assertTrue(report["non_convex_triangles_preserved"])
        self.assertEqual(report["obstacle_count"], 1)
        self.assertEqual(report["collision_cache"], {"cuboid": 4, "mesh": 1})
        dumped = json.dumps(report)
        self.assertIn("table_cuboid", dumped)
        self.assertIn("static_task_obstacles_non_convex_meshes", dumped)

    def test_cache_separation_and_both_ik_and_mp_receive_scene_world(self) -> None:
        recorded_calls: list[dict[str, object]] = []

        class FakeCfg:
            def __init__(self, kind: str, scene_model: dict, collision_cache: dict) -> None:
                self.kind = kind
                self.scene_model = scene_model
                self.collision_cache = collision_cache
                self.trajopt_solver_config = types.SimpleNamespace(interpolation_dt=0.02)
                self.graph_planner_config = None

        fake_mp_mod = types.ModuleType("curobo.motion_planner")
        fake_ik_mod = types.ModuleType("curobo.inverse_kinematics")
        fake_curobo_mod = types.ModuleType("curobo")

        def fake_mp_create(**kwargs):
            recorded_calls.append({"kind": "mp", **kwargs})
            return FakeCfg("mp", kwargs["scene_model"], kwargs["collision_cache"])

        def fake_ik_create(**kwargs):
            recorded_calls.append({"kind": "ik", **kwargs})
            return FakeCfg("ik", kwargs["scene_model"], kwargs["collision_cache"])

        fake_mp_mod.MotionPlannerCfg = types.SimpleNamespace(create=fake_mp_create)
        fake_ik_mod.InverseKinematicsCfg = types.SimpleNamespace(create=fake_ik_create)
        fake_curobo_mod.motion_planner = fake_mp_mod
        fake_curobo_mod.inverse_kinematics = fake_ik_mod

        class FakeBasePlanner:
            def __init__(
                self,
                robot_origin_pose,
                active_joints_name,
                all_joints,
                dt=None,
                yml_path=None,
                table_height=0.74,
            ) -> None:
                self.robot_origin_pose = robot_origin_pose
                self.active_joints_name = active_joints_name
                self.all_joints = all_joints
                self.dt = dt
                self.yml_path = yml_path
                self.device_cfg = "cpu"
                self.use_cuda_graph = False
                self.use_graph_planner = False
                self.robot_cfg, self.scene_model = self._build_robot_and_scene_cfg({}, table_height)
                self.mp_cfg = self._build_motion_planner_cfg(1)
                self.mp_batch_cfg = self._build_motion_planner_cfg(8)
                self.ik_cfg = self._build_ik_cfg()
                checker = types.SimpleNamespace(
                    collision_types={"cuboid": True, "mesh": True, "voxel": False}
                )
                self.motion_planner = types.SimpleNamespace(
                    trajopt_solver=types.SimpleNamespace(scene_collision_checker=checker)
                )
                self.motion_planner_batch = types.SimpleNamespace(
                    trajopt_solver=types.SimpleNamespace(scene_collision_checker=checker)
                )
                self.ik_solver = types.SimpleNamespace(scene_collision_checker=checker)

            def _build_robot_and_scene_cfg(self, yml_data, table_height):
                return (
                    {"robot_cfg": {}},
                    {
                        "cuboid": {
                            "table": {
                                "dims": [3.0, 3.0, 0.05],
                                "pose": [0.0, 0.0, float(table_height) - 0.025, 1.0, 0.0, 0.0, 0.0],
                            }
                        }
                    },
                )

        old_mods = {
            k: sys.modules.get(k)
            for k in ("curobo", "curobo.motion_planner", "curobo.inverse_kinematics")
        }
        sys.modules["curobo"] = fake_curobo_mod
        sys.modules["curobo.motion_planner"] = fake_mp_mod
        sys.modules["curobo.inverse_kinematics"] = fake_ik_mod
        try:
            verts_a, faces_a = _hollow_frame_mesh((0.0, 0.0, 0.765))
            verts_b, faces_b = _hollow_frame_mesh((0.1, 0.0, 0.765))
            base_right = np.eye(4, dtype=np.float64)
            base_right[0, 3] = 0.30
            base_right[2, 3] = 0.765
            base_left = np.eye(4, dtype=np.float64)
            base_left[0, 3] = -0.30
            base_left[2, 3] = 0.765

            with tempfile.TemporaryDirectory() as tmp:
                dir_right = Path(tmp) / "right"
                dir_left = Path(tmp) / "left"
                dir_moved = Path(tmp) / "moved"
                _write_test_scene(dir_right, base_right, verts_a, faces_a, active_arm=1)
                _write_test_scene(dir_left, base_left, verts_a, faces_a, active_arm=0)
                _write_test_scene(dir_moved, base_right, verts_b, faces_b, active_arm=1)

                joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
                p_right_1 = get_scene_planner(
                    dir_right, "x5", joints, -0.025, planner_cls=FakeBasePlanner, yml_path="x5.yml"
                )
                p_right_2 = get_scene_planner(
                    dir_right, "x5", joints, -0.025, planner_cls=FakeBasePlanner, yml_path="x5.yml"
                )
                p_left = get_scene_planner(
                    dir_left, "x5", joints, -0.025, planner_cls=FakeBasePlanner, yml_path="x5.yml"
                )
                p_moved = get_scene_planner(
                    dir_moved, "x5", joints, -0.025, planner_cls=FakeBasePlanner, yml_path="x5.yml"
                )

                self.assertIs(p_right_1, p_right_2)
                self.assertIsNot(p_right_1, p_left)
                self.assertIsNot(p_right_1, p_moved)

                # Verify both MP and IK received cuboid + mesh + collision_cache
                kinds = [c["kind"] for c in recorded_calls[:3]]
                self.assertEqual(kinds, ["mp", "mp", "ik"])
                for call in recorded_calls[:3]:
                    self.assertIn("cuboid", call["scene_model"])
                    self.assertIn("mesh", call["scene_model"])
                    self.assertEqual(call["collision_cache"], {"cuboid": 4, "mesh": 1})
                self.assertTrue(
                    p_right_1.world_report["collision_backends"]["ik_solver"]["collision_types"][
                        "mesh"
                    ]
                )
        finally:
            for k, v in old_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


if __name__ == "__main__":
    unittest.main()
