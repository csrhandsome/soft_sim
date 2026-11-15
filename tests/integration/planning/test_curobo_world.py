"""GPU smoke test for scene-aware cuRobo v2 world collision."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

from soft_sim.planning.world_collision import clear_scene_planner_cache, get_scene_planner

DEFAULT_SCENE_DIR = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "runs"
    / "verify_insert_tubes_s0_per_op_arm_v1_20261115"
    / "plans"
    / "operation_000"
)


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rot).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


class TestCuroboWorldSmoke(unittest.TestCase):
    def setUp(self) -> None:
        clear_scene_planner_cache()

    def tearDown(self) -> None:
        clear_scene_planner_cache()

    def test_scene_world_builds_and_checks_real_mesh_collision(self) -> None:
        import torch

        self.assertTrue(torch.cuda.is_available(), "CUDA is required for cuRobo GPU smoke")

        scene_dir = Path(os.environ.get("SOFT_SIM_SCENE_DIR", str(DEFAULT_SCENE_DIR)))
        self.assertTrue(
            (scene_dir / "scene.json").is_file()
            and (scene_dir / "obstacle_geometry.npz").is_file(),
            f"Exported scene_dir not found: {scene_dir}",
        )

        scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
        base_tf = np.asarray(scene["world_from_robot_base"], dtype=np.float64)
        base_pose = base_tf[:3, 3].tolist() + _rotmat_to_quat_wxyz(base_tf[:3, :3])
        arm_joints = [f"joint{i}" for i in range(1, 7)]
        q0 = [float(x) for x in scene.get("initial_q", [0.0] * 6)[:6]]

        planner = get_scene_planner(
            scene_dir=scene_dir,
            robot_name=str(scene.get("robot_name", "x5")),
            arm_joints=arm_joints,
            table_height=None,
            sample_dt=0.02,
        )

        print("WORLD_REPORT:", json.dumps(planner.world_report, indent=2))

        backends = planner.world_report["collision_backends"]
        for comp_name in ("motion_planner", "motion_planner_batch", "ik_solver"):
            comp = backends[comp_name]
            self.assertTrue(comp["present"], f"{comp_name} missing scene_collision_checker")
            self.assertTrue(comp["collision_types"]["cuboid"], f"{comp_name} cuboid=False")
            self.assertTrue(comp["collision_types"]["mesh"], f"{comp_name} mesh=False")

        # Downward-facing wrist orientation matching curobo.py hand_rotation
        down_quat = _rotmat_to_quat_wxyz(
            np.array(
                [
                    [0.0, -0.70710678, -0.70710678],
                    [0.0, -0.70710678, 0.70710678],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        )

        # Clear free-space pose above the right arm workspace (above tube0 pregrasp)
        clear_pose = [0.116, -0.166, 0.98] + down_quat
        ik_clear = planner.solve_ik_to_joint(q0, clear_pose, real_robot_pose=base_pose)
        self.assertEqual(ik_clear.get("status"), "Success", f"Clear IK failed: {ik_clear}")
        plan_clear = planner.plan_path(q0, clear_pose, real_robot_pose=base_pose)
        self.assertEqual(plan_clear.get("status"), "Success", f"Clear plan failed: {plan_clear}")
        pos = np.asarray(plan_clear["position"])
        self.assertEqual(pos.ndim, 2)
        self.assertEqual(pos.shape[1], 6)
        self.assertGreater(pos.shape[0], 1)

        # Pose directly inside the exported slot mesh ([0.0, 0.0206, 0.795]) must be rejected by mesh collision
        slot_pos = scene["task_objects"]["slot"]["position"]
        blocked_pose = [float(slot_pos[0]), float(slot_pos[1]), float(slot_pos[2]) + 0.01] + down_quat
        ik_blocked = planner.solve_ik_to_joint(q0, blocked_pose, real_robot_pose=base_pose)
        plan_blocked = planner.plan_path(q0, blocked_pose, real_robot_pose=base_pose)
        print("CLEAR_RESULT:", ik_clear.get("status"), plan_clear.get("status"), pos.shape)
        print("BLOCKED_RESULT:", ik_blocked.get("status"), plan_blocked.get("status"))
        self.assertEqual(
            ik_blocked.get("status"),
            "Fail",
            f"Expected IK inside slot mesh to Fail, got {ik_blocked}",
        )
        self.assertEqual(
            plan_blocked.get("status"),
            "Fail",
            f"Expected plan_path inside slot mesh to Fail, got {plan_blocked}",
        )


if __name__ == "__main__":
    unittest.main()
