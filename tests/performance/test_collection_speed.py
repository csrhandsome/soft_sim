"""Compare collection code with its previous per-step/per-frame work schedule.

Run via scripts/run.py test --group performance. Timings include array work and
copies on CPU; fake Kit/Torch interfaces deliberately exclude GPU/renderer cost.
Correctness and operation counts are checked before measuring wall time.
"""
from contextlib import contextmanager
import json
import platform
from statistics import median
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from soft_sim.config import ExperimentConfig
from soft_sim.execution import phase_commands, trajectory_command
from soft_sim.geometry import rotation_y
from soft_sim.sim.soft_tool import MountedSoftTool, capture_mounted_tools
from soft_sim.storage import Storage


@contextmanager
def cpu_interfaces():
    omni, replicator, core, pxr = [ModuleType(name) for name in
        ("omni", "omni.replicator", "omni.replicator.core", "pxr")]
    omni.replicator, replicator.core = replicator, core
    core.orchestrator = SimpleNamespace(step=lambda **kwargs: None)
    pxr.Vt = SimpleNamespace(Vec3fArray=SimpleNamespace(FromNumpy=lambda value: value))
    pxr.Gf = SimpleNamespace(Matrix4d=lambda *values: np.asarray(values).reshape(4, 4))
    with patch.dict(sys.modules, {"omni": omni, "omni.replicator": replicator,
            "omni.replicator.core": core, "pxr": pxr, "torch": ModuleType("torch")}):
        yield


class Surface:
    def __init__(self):
        self.writes = 0
        self.points = None

    def GetPointsAttr(self):
        return self

    def Set(self, points):
        self.writes += 1
        self.points = points


class LegacySurfaceTool(MountedSoftTool):
    def sync_render_surface(self, positions=None):
        # Previous embedding always downloaded its own nodes. Keeping capture
        # and image checks shared isolates the scheduling change being tested.
        super().sync_render_surface()


def make_tool(legacy=False, seed=11):
    rng = np.random.default_rng(seed)
    tool = (LegacySurfaceTool if legacy else MountedSoftTool).__new__(
        LegacySurfaceTool if legacy else MountedSoftTool)
    tool.config = ExperimentConfig()
    tool.config.camera.width = tool.config.camera.height = 256
    tool.render_settings = dict(rt_subframes=4)
    tool.camera_ops = []
    points = [rng.normal(size=(8491, 3)).astype(np.float32) for _ in range(2)]
    tool.reads = 0

    def positions():
        tool.reads += 1
        return [value.copy() for value in points]

    tool.positions = positions
    tool.render_meshes = [Surface(), Surface()]
    indices = rng.integers(0, len(points[0]), size=(20000, 4))
    weights = rng.uniform(size=indices.shape)
    weights /= weights.sum(axis=1, keepdims=True)
    tool.surface_bindings = [(indices, weights)] * 2
    color = rng.integers(0, 256, (256, 256, 4), dtype=np.uint8)
    depth = np.full((256, 256), .03, dtype=np.float32)
    tool.annotators = [(SimpleNamespace(get_data=lambda: color),
                        SimpleNamespace(get_data=lambda: depth))] * 2
    return tool


def previous_frame_schedule(tools, active_index, world):
    for tool in tools:
        tool.sync_render_surface()
    rgb, depth = tools[active_index].capture(world)
    nodes = [tool.positions() for tool in tools]
    return rgb, depth, nodes


class CollectionSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = []

    @classmethod
    def tearDownClass(cls):
        report = dict(scope="CPU collection code only; excludes PhysX, RTX and CUDA transfers",
                      python=sys.version.split()[0], numpy=np.__version__,
                      platform=platform.platform(), comparisons=cls.reports)
        destination = Storage.create().cache_dir("benchmarks") / "collection_performance.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PERFORMANCE_REPORT {destination}", flush=True)

    def compare(self, name, previous, optimized, *, repeats=5, iterations=1, **details):
        previous(), optimized()  # warm both paths, outside the timed region
        samples = [[], []]
        # Alternate order to reduce cache/frequency/thermal bias.
        for repeat in range(repeats):
            for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                function = (previous, optimized)[index]
                started = time.perf_counter()
                for _ in range(iterations):
                    function()
                samples[index].append((time.perf_counter() - started) / iterations)
        before, after = map(median, samples)
        result = dict(name=name, previous_s=before, optimized_s=after, speedup=before/after,
                      repeats=repeats, iterations=iterations, samples_s=samples, **details)
        self.reports.append(result)
        print("PERFORMANCE " + json.dumps(result), flush=True)
        return before / after

    def test_phase_preparation_is_faster_and_commands_are_identical(self):
        path = np.random.default_rng(9).normal(size=(257, 6))
        steps, dt = 1500, .001

        def previous():
            return [trajectory_command(path, steps * dt, (i + 1) * dt, .088, .02)
                    for i in range(steps)]

        def optimized():
            return phase_commands(path, steps, dt, .088, .02)

        expected, actual = previous(), optimized()
        for index, values in enumerate(actual):
            np.testing.assert_array_equal(values, np.asarray([row[index] for row in expected]))
        self.assertGreater(self.compare("phase_commands", previous, optimized,
                           iterations=10, steps=steps, waypoints=len(path)), 2.)

    def test_frame_staging_is_faster_with_the_same_images_and_nodes(self):
        old_tools = [make_tool(True, seed=i) for i in range(2)]
        new_tools = [make_tool(seed=i) for i in range(2)]
        world = SimpleNamespace(forward=lambda: None, render=lambda: None)
        with cpu_interfaces():
            previous = lambda: previous_frame_schedule(old_tools, 1, world)
            optimized = lambda: capture_mounted_tools(new_tools, 1, world)
            before, after = previous(), optimized()
            for old, new in zip(before[:2], after[:2]):
                np.testing.assert_array_equal(old, new)
            for old_arm, new_arm in zip(before[2], after[2]):
                for old, new in zip(old_arm, new_arm):
                    np.testing.assert_array_equal(old, new)
            self.assertEqual([tool.reads for tool in old_tools], [2, 5])
            self.assertEqual([tool.reads for tool in new_tools], [1, 2])
            for old_tool, new_tool in zip(old_tools, new_tools):
                for old, new in zip(old_tool.render_meshes, new_tool.render_meshes):
                    np.testing.assert_array_equal(old.points, new.points)
            self.assertEqual(sum(s.writes for t in old_tools for s in t.render_meshes), 6)
            self.assertEqual(sum(s.writes for t in new_tools for s in t.render_meshes), 4)
            self.assertGreater(self.compare("frame_staging", previous, optimized,
                iterations=8, nodes_per_finger=8491, surface_vertices_per_finger=20000,
                previous_node_reads=7, optimized_node_reads=3,
                previous_surface_writes=6, optimized_surface_writes=4), 1.05)

    def test_camera_authoring_at_capture_rate_is_faster_and_has_the_same_final_pose(self):
        mounts = []
        for step in range(500):
            pose = np.repeat(np.eye(4)[None], 2, axis=0)
            pose[:, :3, :3] = rotation_y(step * .001)
            pose[:, 0, 3] = step * .0001
            mounts.append(pose)

        def make_camera_tool():
            tool = MountedSoftTool.__new__(MountedSoftTool)
            tool.config = ExperimentConfig()
            tool.views, tool.targets, tool.indices = [], [], []
            tool.camera_ops = [Surface(), Surface()]
            return tool

        old, new = make_camera_tool(), make_camera_tool()

        def previous():
            for pose in mounts:
                old.follow(pose)

        def optimized():
            for index, pose in enumerate(mounts):
                new.follow(pose, update_cameras=False)
                if (index + 1) % 50 == 0:
                    new.sync_cameras()

        with cpu_interfaces():
            previous(), optimized()
            for a, b in zip(old.camera_ops, new.camera_ops):
                self.assertEqual(a.writes, 500)
                self.assertEqual(b.writes, 10)
                np.testing.assert_array_equal(a.points, b.points)
            self.assertGreater(self.compare("camera_authoring", previous, optimized,
                physics_steps=len(mounts), fps=20, timestep_s=.001,
                previous_camera_writes=1000, optimized_camera_writes=20), 1.1)


if __name__ == "__main__":
    unittest.main()
