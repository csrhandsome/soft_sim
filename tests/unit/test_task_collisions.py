import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from soft_sim.validation.collisions import pose_matrix, same_vertices


class TaskCollisionGeometry(unittest.TestCase):
    def test_recorded_quaternion_moves_world_geometry_without_changing_shape(self):
        quaternion = Rotation.from_euler("z", 90., degrees=True).as_quat()
        pose = pose_matrix([.2, -.1, .8], np.r_[quaternion[3], quaternion[:3]])
        vertices = np.array([[.01, 0., 0.], [0., .02, 0.], [0., 0., .03]])
        world = vertices @ pose[:3, :3].T + pose[:3, 3]
        np.testing.assert_allclose(world[0], [.2, -.09, .8])
        np.testing.assert_allclose((world-pose[:3, 3]) @ pose[:3, :3], vertices, atol=1e-16)

    def test_dynamic_task_obstacles_are_matched_independently_of_vertex_order(self):
        vertices = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .02, 0.], [0., 0., .03]])
        self.assertTrue(same_vertices(vertices, vertices[[2, 0, 3, 1]]))
        self.assertFalse(same_vertices(vertices, vertices + [.001, 0., 0.]))
        self.assertFalse(same_vertices(vertices, vertices[:3]))
