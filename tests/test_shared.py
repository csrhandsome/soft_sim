"""Numerical contracts that must be identical across the two backends."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from shared.config import ExperimentConfig
from shared.experiment import EpisodeSpec, gripper_state
from shared.force import QuasistaticForceEstimator
from shared.geometry import camera_intrinsics, finger_pose, mounted_camera_pose, rotation_y, transform_points
from shared.learning import RidgeForceRegressor
from shared.mesh import TetMesh, load_stl, tet_volumes


class SharedContracts(unittest.TestCase):
    def test_camera_extrinsics_remain_fixed_in_each_finger(self):
        config = ExperimentConfig()
        episode = EpisodeSpec.sample(config, 9)
        for profile in ('robotiq_2f85', 'panda', 'angular'):
            config.gripper.profile = profile
            relative = []
            for time in (0., .5, 1.):
                state = gripper_state(config.gripper, episode, time)
                for side in (0, 1):
                    position, rotation = finger_pose(config.gripper, side, state.angles_deg[side],
                                                     state.base_separation_m, state.height_offset_m)
                    camera = mounted_camera_pose(config.camera, position, rotation)
                    relative.append(np.r_[rotation.T @ (camera[:3, 3] - position),
                                          (rotation.T @ camera[:3, :3]).ravel()])
            np.testing.assert_allclose(relative, np.tile(relative[0], (6, 1)), atol=1e-12)

    def test_focal_changes_intrinsics_without_changing_mechanics(self):
        config = ExperimentConfig()
        episode = EpisodeSpec.sample(config, 0)
        altered = replace(episode, focal_lengths_mm=(4., 4.))
        a, b = gripper_state(config.gripper, episode, .6), gripper_state(config.gripper, altered, .6)
        np.testing.assert_equal(a.angles_deg, b.angles_deg)
        self.assertEqual(a.base_separation_m, b.base_separation_m)
        k1, k2 = camera_intrinsics(config.camera, 2.), camera_intrinsics(config.camera, 4.)
        self.assertEqual(k2[0, 0], 2 * k1[0, 0])
        self.assertEqual(k1[0, 2], k2[0, 2])

    def test_parallel_mount_tilt_is_independent_of_jaw_motion(self):
        config = ExperimentConfig()
        episode = EpisodeSpec.sample(config, 4)
        start = gripper_state(config.gripper, episode, 0)
        end = gripper_state(config.gripper, episode, 1)
        np.testing.assert_equal(start.angles_deg, end.angles_deg)
        self.assertLess(end.jaw_width_m, start.jaw_width_m)
        self.assertGreater(end.height_offset_m, start.height_offset_m)
        config.gripper.profile = 'panda'
        self.assertEqual(gripper_state(config.gripper, episode, 1).height_offset_m, 0.)

    def test_reference_mesh_integrity_and_units(self):
        config = ExperimentConfig()
        mesh = TetMesh.load(config.mesh)
        vertices, faces = load_stl(Path(__file__).resolve().parents[1] / config.mesh.stl)
        np.testing.assert_allclose(np.ptp(vertices, axis=0), [.04, .04, .08818], atol=1e-8)
        self.assertEqual(len(faces), 1700)
        self.assertTrue(np.all(tet_volumes(mesh.vertices, mesh.tets) > 0))
        self.assertLess(abs(mesh.metadata['volume_relative_error']), .05)

    def test_force_estimator_rigid_motion_and_known_stretch(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
        config = ExperimentConfig()
        config.material.young_modulus_pa = 1000.
        config.material.poisson_ratio = .25
        estimator = QuasistaticForceEstimator(vertices, np.array([[0, 1, 2, 3]]),
                                              np.array([True, False, True, True]), config.material, [0, 0, 0])
        rigid = transform_points(vertices, np.array([.1, -.2, .3]), rotation_y(.7))
        np.testing.assert_allclose(estimator.estimate(rigid), 0., atol=1e-10)
        stretched = vertices.copy()
        stretched[1, 0] = 1.1
        self.assertAlmostEqual(estimator.estimate(stretched)[0], 20., places=8)

    def test_model_serialization_and_prediction(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(40, 4))
        y = x @ np.array([[1., 0], [0, 1], [2, 3], [-1, 2]]) + 20
        model = RidgeForceRegressor.fit(x, y, alpha=.001)
        np.testing.assert_allclose(model.predict_features(x), y, atol=.001)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.npz'
            model.save(path)
            np.testing.assert_equal(model.predict_features(x), RidgeForceRegressor.load(path).predict_features(x))

    def test_invalid_configuration_is_rejected(self):
        config = ExperimentConfig()
        config.material.poisson_ratio = .5
        with self.assertRaises(ValueError):
            config.validate()
        config.material.poisson_ratio = .45
        config.camera.focal_length_range_mm = [float('nan'), 3.]
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == '__main__':
    unittest.main()
