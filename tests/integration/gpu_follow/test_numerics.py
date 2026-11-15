"""Check quaternion conventions, target rounding and fixture diagnostics."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from soft_sim.sim.gpu_follow import GpuToolFollower, measured_mounts, node_targets, fixture_errors


class GpuFollowNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("gpu-follow requires a visible NVIDIA CUDA device")
        wp.init()

    def test_targets_match_scipy_for_measured_nonunit_quaternions(self):
        rng = np.random.default_rng(572)
        device = "cuda:0"
        offsets_np = np.repeat(np.eye(4)[None], 4, axis=0)
        offsets_np[:, :3, :3] = Rotation.random(4, random_state=rng).as_matrix()
        offsets_np[:, :3, 3] = rng.normal(size=(4, 3)) * .03
        offsets = torch.tensor(offsets_np, dtype=torch.float64, device=device)
        mounts = torch.empty_like(offsets)
        local_np = rng.normal(size=(4, 8491, 3)) * .015
        local = torch.tensor(local_np, device=device)
        values = rng.normal(size=(4, 8491, 4)).astype(np.float32)
        values[..., 3] = rng.integers(0, 2, size=values.shape[:2])
        targets = torch.tensor(values, device=device)
        links = torch.tensor([3, 1], dtype=torch.int32, device=device)
        wp_offsets, wp_mounts = [wp.from_torch(t, dtype=wp.mat44d) for t in (offsets, mounts)]
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        maximum = 0.
        # Include stationary repetition, large translations and near-pi rotations.
        for frame in range(40):
            expected = []
            for arm in range(2):
                poses_np = rng.normal(size=(6, 7)).astype(np.float32)
                poses_np[:, :3] *= 3 if frame % 2 else .1
                poses_np[:, 3:] *= .98  # SciPy normalizes too
                if frame == 0:
                    poses_np[:, 3:] = [0., 1., 0., 0.]
                poses = torch.tensor(poses_np, device=device)
                wp.launch(measured_mounts, 2, inputs=[wp.from_torch(poses), wp.from_torch(links),
                          wp_offsets, wp_mounts, arm*2], stream=stream)
                for side, row in enumerate((3, 1)):
                    pose = poses_np[row]
                    matrix = np.eye(4)
                    matrix[:3, 3] = pose[:3]
                    matrix[:3, :3] = Rotation.from_quat(np.r_[pose[4:7], pose[3]]).as_matrix()
                    expected.append(matrix @ offsets_np[arm*2+side])
            wp.launch(node_targets, local.shape[:2], inputs=[wp.from_torch(local, dtype=wp.vec3d),
                      wp_mounts, wp.from_torch(targets, dtype=wp.vec4f)], stream=stream)
            actual = targets.cpu().numpy()
            np.testing.assert_allclose(mounts.cpu().numpy(), expected, atol=3e-15, rtol=0)
            for body, pose in enumerate(expected):
                reference = (local_np[body] @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
                maximum = max(maximum, float(np.abs(actual[body, :, :3]-reference).max()))
                np.testing.assert_allclose(actual[body, :, :3], reference, atol=1e-7, rtol=0)
            np.testing.assert_array_equal(actual[..., 3], values[..., 3])
        print(f"GPU_FOLLOW_TARGET_MAX_ERROR_M {maximum:.12g}", flush=True)

    def test_fixture_errors_match_independent_numpy_check(self):
        rng = np.random.default_rng(19)
        poses_np = rng.normal(size=(4, 7)).astype(np.float32)
        offsets_np = np.repeat(np.eye(4)[None], 4, axis=0)
        offsets_np[:, :3, :3] = Rotation.random(4, random_state=rng).as_matrix()
        offsets_np[:, :3, 3] = rng.normal(size=(4, 3)) * .02
        actual = []
        for pose, offset in zip(poses_np, offsets_np):
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
            matrix[:3, 3] = pose[:3]
            actual.append(matrix @ offset)
        mounts_np = np.stack(actual)
        mounts_np[:, :3, 3] += [1e-5, -2e-5, 3e-5]
        mounts_np[:, :3, :3] = mounts_np[:, :3, :3] @ Rotation.from_rotvec([1e-4, 0, 0]).as_matrix()
        poses = torch.tensor(poses_np, device="cuda:0")
        offsets = torch.tensor(offsets_np, device="cuda:0")
        mounts = torch.tensor(mounts_np, device="cuda:0")
        errors = torch.empty((4, 2), device="cuda:0", dtype=torch.float64)
        wp.launch(fixture_errors, 4, inputs=[wp.from_torch(poses),
                  wp.from_torch(offsets, dtype=wp.mat44d), wp.from_torch(mounts, dtype=wp.mat44d),
                  wp.from_torch(errors)], stream=wp.stream_from_torch(torch.cuda.current_stream()))
        reference = np.stack((np.linalg.norm(np.asarray(actual)[:, :3, 3]-mounts_np[:, :3, 3], axis=1),
                              np.abs(np.asarray(actual)[:, :3, :3]-mounts_np[:, :3, :3]).max(axis=(1, 2))), axis=1)
        np.testing.assert_allclose(errors.cpu().numpy(), reference, atol=2e-15, rtol=0)

    def test_hot_path_has_no_host_readback_and_handles_reordered_bodies(self):
        from soft_sim.sim.soft_tool import MountedSoftTool
        device = "cuda:0"
        local = np.arange(24, dtype=float).reshape(8, 3) * .001
        positions = [torch.tensor(local[None] + i, device=device, dtype=torch.float32)
                     for i in range(4)]
        # Backend order deliberately differs from requested arm/side order.
        order = [2, 0, 3, 1]
        saved = []
        batch = SimpleNamespace(count=4, get_sim_nodal_positions=lambda: torch.cat([positions[i] for i in order]),
                                set_sim_kinematic_targets=lambda data, indices: saved.append((data.clone(), indices.clone())))
        parents = [f"/World/Arm{i}/Slide{side}" for i in range(2) for side in range(2)]
        rigid_poses = torch.tensor([[i, 0, 0, 0, 0, 0, 1] for i in range(4)],
                                   dtype=torch.float32, device=device)
        rigid = SimpleNamespace(prim_paths=parents, get_transforms=lambda: rigid_poses)
        sim = SimpleNamespace(create_soft_body_view=lambda pattern: batch,
                              create_rigid_body_view=lambda pattern: rigid)
        tools, articulations = [], []
        for arm in range(2):
            tool = MountedSoftTool.__new__(MountedSoftTool)
            tool.sleep_threshold = 0.
            tool.targets = [torch.ones((1, 8, 4), device=device) for _ in range(2)]
            for target in tool.targets:
                target[:, :3, 3] = 0.
            tool.local_nodes = [local.copy(), local.copy()]
            tool.paths = [f"/World/SoftTool{arm}/Finger_{side}" for side in range(2)]
            tool.views = [SimpleNamespace(get_sim_nodal_positions=lambda i=i: positions[i])
                          for i in (arm*2, arm*2+1)]
            tool.tensor_sim = sim
            tool.fixtures = SimpleNamespace(parents=parents[arm*2:arm*2+2])
            tool.camera_ops = []
            tools.append(tool)
            poses = torch.tensor([[arm*2+side, 0, 0, 1, 0, 0, 0] for side in range(2)],
                                  dtype=torch.float32, device=device)
            articulations.append(SimpleNamespace(body_names=["left", "right"],
                                                 data=SimpleNamespace(body_link_pose_w=poses[None])))
        follower = GpuToolFollower(tools, articulations, [("left", "right")]*2,
                                   [np.repeat(np.eye(4)[None], 2, axis=0)]*2)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Unexpected per-step CPU readback")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("Unexpected per-step scalar sync")):
            follower.follow()
            follower.follow()  # identical mounts are valid with sleep disabled
        data = saved[-1][0].cpu().numpy()
        for row, body in enumerate(order):
            expected = (local + [body, 0, 0]).astype(np.float32)
            np.testing.assert_array_equal(data[row, :, :3], expected)
            np.testing.assert_array_equal(data[row, :, 3], [0, 0, 0, 1, 1, 1, 1, 1])
        for arm, tool in enumerate(tools):
            self.assertIsNone(tool._mounts)
            np.testing.assert_array_equal(tool.mounts[:, 0, 3], [arm*2, arm*2+1])
            np.testing.assert_array_equal(tool.fixtures._pre_step_error_gpu.cpu().numpy(), 0.)
