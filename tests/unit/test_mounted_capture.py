"""Headless IsaacLab must flush Fabric even when render() does nothing."""
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from soft_sim.sim.soft_tool import MountedSoftTool


class HeadlessCapture(unittest.TestCase):
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
        tool.sync_render_surface = lambda: calls.append("surface")
        tool.annotators = [(Annotator(color),Annotator(depth))]*2
        with patch.dict(sys.modules, {"omni": omni, "omni.replicator": replicator,
                                      "omni.replicator.core": core}):
            rgb, distances = tool.capture(world)
        np.testing.assert_array_equal(rgb, np.stack([color]*2))
        np.testing.assert_array_equal(distances, np.stack([depth]*2))
        self.assertEqual(calls[-1],"replicate")
