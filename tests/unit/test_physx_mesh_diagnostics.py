import unittest

import numpy as np

from soft_sim.validation.physx import rest_cell_diagnostics


class LegacyMassDiagnostics(unittest.TestCase):
    def test_small_positive_cell_is_not_classified_as_inverted(self):
        vertices = np.array([[0., 0., 0.], [.0001, 0., 0.], [0., .0001, 0.], [0., 0., .0001]])
        report = rest_cell_diagnostics(vertices, np.array([[0, 1, 2, 3]]))
        self.assertGreater(report["minimum_determinant_m3"], 0)
        self.assertEqual(report["small_positive_cells"], 1)
        self.assertEqual(report["nonpositive_or_nonfinite_cells"], 0)

    def test_true_inversion_is_kept_separate_from_small_volume_diagnostics(self):
        vertices = np.array([[0., 0., 0.], [.0001, 0., 0.], [0., .0001, 0.], [0., 0., .0001]])
        report = rest_cell_diagnostics(vertices, np.array([[1, 0, 2, 3]]))
        self.assertEqual(report["small_positive_cells"], 0)
        self.assertEqual(report["nonpositive_or_nonfinite_cells"], 1)
