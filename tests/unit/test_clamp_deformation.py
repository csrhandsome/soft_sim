import unittest

import numpy as np

from soft_sim.validation.clamp import element_stretches, rigid_fit
from soft_sim.geometry import rotation_y


class ClampDeformationTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.], [0., 0., .03]])
        self.tets = np.array([[0, 1, 2, 3]])

    def test_rigid_motion_has_unit_stretches_and_zero_fit_residual(self):
        current = self.points @ rotation_y(.7).T + [.2, -.1, .4]
        stretch, determinant = element_stretches(self.points, current, self.tets)
        fit, angle = rigid_fit(self.points, current)
        np.testing.assert_allclose(stretch, 1., atol=1e-12)
        np.testing.assert_allclose(determinant, 1., atol=1e-12)
        np.testing.assert_allclose(fit, current, atol=1e-12)
        self.assertAlmostEqual(angle, np.degrees(.7))

    def test_stretch_survives_superposed_rigid_motion(self):
        current = (self.points * [1.2, .9, 1.]) @ rotation_y(.7).T + [.2, -.1, .4]
        stretch, determinant = element_stretches(self.points, current, self.tets)
        np.testing.assert_allclose(stretch, [[1.2, 1., .9]], atol=1e-12)
        np.testing.assert_allclose(determinant, 1.08, atol=1e-12)
        fit, _ = rigid_fit(self.points, current)
        self.assertGreater(np.linalg.norm(fit - current), .0001)

    def test_inversion_is_not_hidden_by_singular_values(self):
        current = self.points * [-1., 1., 1.]
        stretch, determinant = element_stretches(self.points, current, self.tets)
        np.testing.assert_allclose(stretch, 1.)
        self.assertLess(determinant[0], 0.)
        fit, _ = rigid_fit(self.points, current)
        self.assertGreater(np.linalg.norm(fit - current), .001)
