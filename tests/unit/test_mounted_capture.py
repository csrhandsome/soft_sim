"""Headless IsaacLab must flush Fabric even when render() does nothing."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from soft_sim.config import ExperimentConfig
from soft_sim.geometry import mounted_camera_pose, rotation_y
from soft_sim.sim.soft_tool import MountedSoftTool, capture_mounted_tools


class HeadlessCapture(unittest.TestCase):
    def test_gpu_mounts_download_once_on_demand_and_refresh_after_a_step(self):
        tool = MountedSoftTool.__new__(MountedSoftTool)
        first = np.repeat(np.eye(4)[None], 2, axis=0)
        download = unittest.mock.Mock(return_value=SimpleNamespace(numpy=lambda: first))
        tool._mounts_gpu = SimpleNamespace(cpu=download)
        tool._mounts = None
        download.assert_not_called()
        np.testing.assert_array_equal(tool.mounts, first)
        np.testing.assert_array_equal(tool.mounts, first)
        download.assert_called_once_with()
        second = first.copy()
        second[:, 0, 3] = .01
        next_download = unittest.mock.Mock(return_value=SimpleNamespace(numpy=lambda: second))
        tool._mounts_gpu = SimpleNamespace(cpu=next_download)
        tool._mounts = None
        np.testing.assert_array_equal(tool.mounts, second)
        next_download.assert_called_once_with()
        tool.mounts = first
        self.assertIsNone(tool._mounts_gpu)
        np.testing.assert_array_equal(tool.mounts, first)

    def test_replication_observes_current_fabric_without_stepping_physics(self):
        calls = []
        world = SimpleNamespace(forward=lambda: calls.append("forward"),
                                render=lambda: calls.append("headless_render"))
        color = np.arange(12, dtype=np.uint8).reshape(2,2,3)*10
        depth = np.full((2,2), .03, dtype=np.float32)

        class Annotator:
            def __init__(self, pixels):
                self.pixels = pixels

            def get_data(self):
                # A headless render alone leaves the GPU scene at reset.
                if "forward" not in calls:
                    return np.zeros_like(self.pixels)
                return self.pixels

        def replicate(**kwargs):
            self.assertEqual(calls, ["surface", "forward", "headless_render"])
            self.assertEqual(kwargs["delta_time"], 0.)
            calls.append("replicate")

        omni, replicator, core = (ModuleType(name) for name in
                                  ("omni", "omni.replicator", "omni.replicator.core"))
        omni.replicator, replicator.core = replicator, core
        core.orchestrator = SimpleNamespace(step=replicate)
        tool = MountedSoftTool.__new__(MountedSoftTool)
        tool.config = SimpleNamespace(camera=SimpleNamespace(height=2,width=2,far_m=2.))
        tool.render_settings = dict(rt_subframes=2)
        tool.positions = lambda: [np.zeros((4,3))]*2
        tool.sync_render_surface = lambda positions=None: calls.append("surface")
        tool.camera_ops = []
        tool.annotators = [(Annotator(color),Annotator(depth))]*2
        with patch.dict(sys.modules, {"omni": omni, "omni.replicator": replicator,
                                      "omni.replicator.core": core}):
            rgb, distances = tool.capture(world)
        np.testing.assert_array_equal(rgb, np.stack([color]*2))
        np.testing.assert_array_equal(distances, np.stack([depth]*2))
        self.assertEqual(calls[-1],"replicate")

    def test_snapshot_is_shared_but_post_render_motion_is_still_rejected(self):
        nodes = [np.zeros((4, 3))] * 2
        tool = MountedSoftTool.__new__(MountedSoftTool)
        tool.positions = unittest.mock.Mock(return_value=[np.full((4, 3), 1e-6)] * 2)
        tool.sync_render_surface = unittest.mock.Mock()
        tool.sync_cameras = unittest.mock.Mock()
        tool.render_settings = dict(rt_subframes=4)
        core = ModuleType("omni.replicator.core")
        core.orchestrator = SimpleNamespace(step=unittest.mock.Mock())
        world = SimpleNamespace(forward=unittest.mock.Mock(), render=unittest.mock.Mock())
        with patch.dict(sys.modules, {"omni.replicator.core": core}):
            # Import the package as well, as capture() uses a dotted import.
            omni, replicator = ModuleType("omni"), ModuleType("omni.replicator")
            omni.replicator, replicator.core = replicator, core
            with patch.dict(sys.modules, {"omni": omni, "omni.replicator": replicator}):
                with self.assertRaisesRegex(RuntimeError, "Rendering advanced FEM"):
                    tool.capture(world, positions=nodes)
        tool.sync_render_surface.assert_called_once_with(nodes)
        tool.sync_cameras.assert_called_once_with()
        tool.positions.assert_called_once_with()
        core.orchestrator.step.assert_called_once_with(
            rt_subframes=4, delta_time=0., pause_timeline=False, wait_for_render=True)

    def test_collection_reads_each_tool_once_and_embeds_each_surface_once(self):
        for active_index in (0, 1):
            nodes = [[np.full((4, 3), i)] * 2 for i in range(2)]
            tools = [SimpleNamespace(positions=unittest.mock.Mock(return_value=value),
                     sync_render_surface=unittest.mock.Mock(),
                     capture=unittest.mock.Mock(return_value=("rgb", "depth"))) for value in nodes]
            world = object()
            rgb, depth, snapshots = capture_mounted_tools(tools, active_index, world)
            self.assertEqual((rgb, depth), ("rgb", "depth"))
            for index, tool in enumerate(tools):
                tool.positions.assert_called_once_with()
                self.assertIs(snapshots[index], nodes[index])
                if index == active_index:
                    tool.capture.assert_called_once_with(world, positions=nodes[index])
                    tool.sync_render_surface.assert_not_called()  # capture owns this embedding
                else:
                    tool.sync_render_surface.assert_called_once_with(nodes[index])
                    tool.capture.assert_not_called()

    def test_deferred_camera_flush_uses_latest_mounts(self):
        tool = MountedSoftTool.__new__(MountedSoftTool)
        tool.config = ExperimentConfig()
        tool.camera_ops = [SimpleNamespace(Set=unittest.mock.Mock()) for _ in range(2)]
        tool.views, tool.targets, tool.indices = [], [], []
        tool.fixtures = SimpleNamespace(follow=unittest.mock.Mock())
        fake_torch = ModuleType("torch")
        gf = SimpleNamespace(Matrix4d=lambda *values: np.asarray(values).reshape(4, 4))
        pxr = ModuleType("pxr")
        pxr.Gf = gf
        with patch.dict(sys.modules, {"torch": fake_torch, "pxr": pxr}):
            for step in range(50):
                mounts = np.repeat(np.eye(4)[None], 2, axis=0)
                mounts[:, :3, :3] = rotation_y(step * .01)
                mounts[:, 0, 3] = step * .001
                tool.follow(mounts, update_cameras=False)
            for op in tool.camera_ops:
                op.Set.assert_not_called()
            tool.sync_cameras()
        self.assertEqual(tool.fixtures.follow.call_count, 50)
        for op, mount, optical in zip(tool.camera_ops, mounts, tool.camera_poses):
            op.Set.assert_called_once()
            expected = mounted_camera_pose(tool.config.camera, mount[:3, 3], mount[:3, :3])
            np.testing.assert_array_equal(optical, expected)
            expected[:3, :3] = expected[:3, :3] @ np.diag([1., -1., -1.])
            np.testing.assert_array_equal(op.Set.call_args.args[0], expected.T)

    def test_deferred_cameras_still_drive_fem_targets_at_every_moving_step(self):
        class Targets:
            dtype, device = np.float32, "cpu"

            def __init__(self):
                self.values = np.ones((1, 4, 4), dtype=self.dtype)
                self.values[:, :2, 3] = 0.

            def __setitem__(self, index, value):
                self.values[index] = value

        tool = MountedSoftTool.__new__(MountedSoftTool)
        tool.camera_ops = [SimpleNamespace(Set=unittest.mock.Mock()) for _ in range(2)]
        tool.targets = [Targets(), Targets()]
        tool.indices = [np.array([0], dtype=np.int32)] * 2
        tool.local_nodes = [np.arange(12, dtype=float).reshape(4, 3) * .001] * 2
        tool._last_target_points = [None, None]
        tool.views = [SimpleNamespace(set_sim_kinematic_targets=unittest.mock.Mock()) for _ in range(2)]
        torch = ModuleType("torch")
        torch.as_tensor = lambda values, **kwargs: np.asarray(values, dtype=kwargs["dtype"])
        with patch.dict(sys.modules, {"torch": torch}):
            for step in range(50):
                mounts = np.repeat(np.eye(4)[None], 2, axis=0)
                mounts[:, :3, :3] = rotation_y(step * .001)
                mounts[:, 0, 3] = step * .0001
                tool.follow(mounts, update_cameras=False)
                for side, (view, targets, mount) in enumerate(zip(tool.views, tool.targets, mounts)):
                    self.assertEqual(view.set_sim_kinematic_targets.call_count, step + 1)
                    expected = (tool.local_nodes[side] @ mount[:3, :3].T + mount[:3, 3]).astype(np.float32)
                    np.testing.assert_array_equal(targets.values[0, :, :3], expected)
                    np.testing.assert_array_equal(targets.values[0, :, 3], [0., 0., 1., 1.])
            tool.follow(mounts, update_cameras=False)
            self.assertEqual(tool.views[0].set_sim_kinematic_targets.call_count, 50)
        for op in tool.camera_ops:
            op.Set.assert_not_called()
