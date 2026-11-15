"""Tests for bounded RoboDojo/Isaac Sim host-compatibility helpers."""

import unittest
import numpy as np

from unittest import mock

from soft_sim.sim.robodojo import (
    ensure_articulation_dof_properties_compat,
    simulator_config,
)


class _FakeCudaTensor:
    """Simulates a CUDA tensor whose direct NumPy conversion raises TypeError."""

    def __init__(self, array):
        self._array = np.asarray(array)
        self.device = "cuda:0"
        self.cpu_called = False

    def __getitem__(self, item):
        return _FakeCudaTensor(self._array[item])

    def detach(self):
        return self

    def cpu(self):
        self.cpu_called = True
        return _FakeHostTensor(self._array)

    def __array__(self, dtype=None):
        raise TypeError(
            "can't convert cuda:0 device type tensor to numpy. "
            "Use Tensor.cpu() to copy the tensor to host memory first."
        )


class _FakeHostTensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def numpy(self):
        return self._array


class _FakeView:
    def __init__(self, use_cuda_tensors):
        wrap = _FakeCudaTensor if use_cuda_tensors else np.asarray
        self._types = wrap([[1, 2]])
        self._limits = wrap([[[-1.5, 1.5], [0.0, 0.04]]])
        self._drives = wrap([[1, 1]])
        self._efforts = wrap([[120.0, 45.0]])
        self._velocities = wrap([[3.5, 0.8]])
        self._stiffness = wrap([[600.0, 200.0]])
        self._damping = wrap([[30.0, 10.0]])

    def get_dof_types(self):
        return self._types

    def get_dof_limits(self):
        return self._limits

    def get_drive_types(self):
        return self._drives

    def get_max_efforts(self):
        return self._efforts

    def get_joint_max_velocities(self):
        return self._velocities

    def get_gains(self):
        return self._stiffness, self._damping


def _make_articulation_cls():
    class _FakeSingleArticulation:
        def __init__(self, use_cuda_tensors):
            self.num_dof = 2
            self._articulation_view = _FakeView(use_cuda_tensors=use_cuda_tensors)

        @property
        def dof_properties(self):
            dtype = np.dtype(
                [
                    ("type", int),
                    ("hasLimits", bool),
                    ("lower", float),
                    ("upper", float),
                    ("driveMode", int),
                    ("maxVelocity", float),
                    ("maxEffort", float),
                    ("stiffness", float),
                    ("damping", float),
                ]
            )
            properties = np.zeros(self.num_dof, dtype=dtype)
            properties["type"] = self._articulation_view.get_dof_types()[0]
            properties["lower"] = self._articulation_view.get_dof_limits()[0][:, 0]
            properties["upper"] = self._articulation_view.get_dof_limits()[0][:, 1]
            properties["hasLimits"] = properties["lower"] < properties["upper"]
            properties["driveMode"] = self._articulation_view.get_drive_types()[0]
            properties["maxEffort"] = self._articulation_view.get_max_efforts()[0]
            properties["maxVelocity"] = self._articulation_view.get_joint_max_velocities()[0]
            stiffnesses, dampings = self._articulation_view.get_gains()
            properties["stiffness"] = stiffnesses[0]
            properties["damping"] = dampings[0]
            return properties

    return _FakeSingleArticulation


class ArticulationCompatTest(unittest.TestCase):
    def test_cuda_tensor_getters_copy_to_host_and_preserve_values(self):
        cls = _make_articulation_cls()
        cuda_art = cls(use_cuda_tensors=True)
        with self.assertRaises(TypeError):
            _ = cuda_art.dof_properties

        self.assertTrue(ensure_articulation_dof_properties_compat(cls))
        # Idempotent on repeated calls.
        self.assertFalse(ensure_articulation_dof_properties_compat(cls))

        props = cuda_art.dof_properties
        np.testing.assert_array_equal(props["type"], [1, 2])
        np.testing.assert_array_equal(props["hasLimits"], [True, True])
        np.testing.assert_allclose(props["lower"], [-1.5, 0.0])
        np.testing.assert_allclose(props["upper"], [1.5, 0.04])
        np.testing.assert_array_equal(props["driveMode"], [1, 1])
        np.testing.assert_allclose(props["maxEffort"], [120.0, 45.0])
        np.testing.assert_allclose(props["maxVelocity"], [3.5, 0.8])
        np.testing.assert_allclose(props["stiffness"], [600.0, 200.0])
        np.testing.assert_allclose(props["damping"], [30.0, 10.0])
        # Device tensor on the view itself remains untouched.
        self.assertEqual(cuda_art._articulation_view._efforts.device, "cuda:0")

    def test_cpu_numpy_getters_remain_compatible(self):
        cls = _make_articulation_cls()
        ensure_articulation_dof_properties_compat(cls)
        cpu_art = cls(use_cuda_tensors=False)
        props = cpu_art.dof_properties
        np.testing.assert_allclose(props["lower"], [-1.5, 0.0])
        np.testing.assert_allclose(props["upper"], [1.5, 0.04])
        np.testing.assert_allclose(props["maxEffort"], [120.0, 45.0])
        np.testing.assert_allclose(props["stiffness"], [600.0, 200.0])
        np.testing.assert_allclose(props["damping"], [30.0, 10.0])


class SimulatorConfigBufferTest(unittest.TestCase):
    def test_physx_deformable_buffer_capacities_exceed_observed_overflow(self):
        base_cfg = {"scene": {}, "physx": {}, "render": {}}
        with mock.patch("soft_sim.sim.robodojo.load_yaml", return_value=base_cfg):
            cfg = simulator_config(
                seed=0,
                dt=0.001,
                solver_iterations=80,
                texture_max_resolution=1024,
            )
        physx = cfg["physx"]
        self.assertEqual(physx["min_position_iteration_count"], 80)
        self.assertEqual(cfg["dt"], 0.001)
        self.assertGreaterEqual(physx["gpu_collision_stack_size"], 134217728)
        self.assertGreaterEqual(physx["gpu_max_soft_body_contacts"], 1048576)


if __name__ == "__main__":
    unittest.main()
