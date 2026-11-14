import unittest

import numpy as np

from soft_sim.validation.fem_projection import projected_depth


class DeformedVolumeProjection(unittest.TestCase):
    def test_optical_depth_is_preserved_under_world_pose_change(self):
        vertices = np.array([[-.01, -.01, .02], [.01, -.01, .02], [.01, .01, .02], [-.01, .01, .02]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        camera = np.eye(4)
        camera[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        camera[:3, 3] = [.3, -.45, .9]
        world = vertices@camera[:3, :3].T+camera[:3, 3]
        intrinsics = np.array([[100., 0., 128.], [0., 100., 128.], [0., 0., 1.]])
        pixels = np.array([[128, 128], [138, 138], [118, 128]])
        np.testing.assert_allclose(projected_depth(world, faces, camera, intrinsics, pixels), .02)
        shifted = camera.copy()
        shifted[2, 3] += .005
        np.testing.assert_allclose(projected_depth(world, faces, shifted, intrinsics, pixels), .015)
        missed = projected_depth(world, faces, camera, intrinsics, np.array([[999, 999]]))
        self.assertTrue(np.isnan(missed).all())
