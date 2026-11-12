"""Known box distances detect deep interpenetration and pose-convention errors."""
import unittest

import numpy as np

from benchmarks.check_box_contact import box_signed_distances


class BoxContactMeasurements(unittest.TestCase):
    def test_known_inside_face_and_corner_distances(self):
        points = [[0, 0, 0], [.021, 0, 0], [.023, .034, 0]]
        distance = box_signed_distances(points, [0, 0, 0], [1, 0, 0, 0], [.04, .06, .08])
        np.testing.assert_allclose(distance, [-.02, .001, .005], atol=1e-12)

    def test_translated_rotated_box_uses_world_from_object_quaternion(self):
        center = np.array([.42, -.12, .155])
        point = center + np.array([.021 / np.sqrt(2), .021 / np.sqrt(2), 0])
        quat = [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]
        distance = box_signed_distances([point], center, quat, [.04, .06, .08])
        np.testing.assert_allclose(distance, [.001], atol=1e-12)

    def test_invalid_or_missing_geometry_is_rejected(self):
        for points in ([], [[float("nan"), 0, 0]]):
            with self.assertRaises(ValueError):
                box_signed_distances(points, [0, 0, 0], [1, 0, 0, 0], [.04, .06, .08])


if __name__ == "__main__":
    unittest.main()
