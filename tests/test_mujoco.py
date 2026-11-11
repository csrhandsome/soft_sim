"""Actual volume-deformation/contact integration tests, no fake physics backend."""

from dataclasses import replace
from pathlib import Path
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mujoco'))
from shared.config import ExperimentConfig
from shared.experiment import EpisodeSpec
from soft_gripper_env import MuJoCoGripperEnv


class MuJoCoIntegration(unittest.TestCase):
    def test_contacts_deform_soft_fingers_and_ignore_camera_focal_length(self):
        config = ExperimentConfig()
        config.simulation.render = False
        config.gripper.mount_tilt_range_deg = (5., 5.)
        config.gripper.final_jaw_width_range_m = (.015, .015)
        base = EpisodeSpec.sample(config, 1)
        base = replace(base, object_position_m=(0., 0., .18), object_radius_m=.018)
        forces = []
        with MuJoCoGripperEnv(config) as env:
            for shape, focal in (('sphere', 2.), ('sphere', 4.), ('box', 2.), ('cylinder', 2.)):
                episode = replace(base, object_shape=shape, focal_lengths_mm=(focal, focal))
                observation = env.reset(episode=episode)
                np.testing.assert_array_equal(observation.force.normal_force_n, [0., 0.])
                for _ in range(1100):
                    env.step()
                observation = env.observe()
                self.assertTrue(observation.force.valid)
                self.assertTrue(observation.bilateral_contact)
                self.assertTrue(np.all(observation.force.normal_force_n > .1))
                self.assertLess(observation.force.contact_force_world_n[0, 0], 0)
                self.assertGreater(observation.force.contact_force_world_n[1, 0], 0)
                self.assertGreater(abs(observation.displacement_m).max(), .0001)
                self.assertLess(abs(observation.displacement_m).max(), .03)
                np.testing.assert_array_equal(env.data.warning.number, np.zeros_like(env.data.warning.number))
                forces.append(observation.force.normal_force_n)
            np.testing.assert_allclose(forces[0], forces[1], atol=1e-9)

    @unittest.skipUnless(os.environ.get('SOFT_SIM_TEST_RENDER') == '1', 'Set SOFT_SIM_TEST_RENDER=1 to test OpenGL rendering')
    def test_dual_rgb_depth_and_calibration(self):
        config = ExperimentConfig()
        config.camera.width, config.camera.height = 96, 80
        with MuJoCoGripperEnv(config) as env:
            observation = env.reset(3)
            self.assertEqual(observation.rgb.shape, (2, 80, 96, 3))
            self.assertEqual(observation.depth_m.shape, (2, 80, 96))
            self.assertTrue(np.isfinite(observation.depth_m).any())
            self.assertGreater(np.std(observation.rgb), 5.)
            for side in (0, 1):
                focal = env.episode.focal_lengths_mm[side]
                self.assertAlmostEqual(observation.camera_intrinsics[side, 0, 0], 96 * focal / config.camera.sensor_width_mm)


if __name__ == '__main__':
    unittest.main()
