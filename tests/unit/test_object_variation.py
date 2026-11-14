"""Rotation and object scheduling must be reproducible across backends."""
import unittest
import numpy as np
from soft_sim.config import ExperimentConfig
from soft_sim.force_estimation.specs import EpisodeSpec
from soft_sim.geometry import rotation_rpy_degrees, matrix_to_quat_wxyz

class ObjectVariation(unittest.TestCase):
    def test_cycle_covers_every_shape_and_changes_orientation(self):
        c=ExperimentConfig();c.object.selection='cycle'
        c.object.rotation_rpy_range_deg=((-10.,10.),(-15.,15.),(-90.,90.))
        episodes=[EpisodeSpec.sample(c,i) for i in range(3)]
        self.assertEqual([e.object_shape for e in episodes],list(c.object.shapes))
        self.assertEqual(episodes[0].to_dict(),EpisodeSpec.sample(c,0).to_dict())
        self.assertFalse(np.allclose(episodes[0].object_rpy_deg,episodes[1].object_rpy_deg))
        for e in episodes:
            r=rotation_rpy_degrees(e.object_rpy_deg)
            np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r),1.)
            self.assertAlmostEqual(np.linalg.norm(matrix_to_quat_wxyz(r)),1.)
    def test_invalid_collision_offsets_and_rotation_ranges_rejected(self):
        for rest,contact in ((.001,.001),(-.001,.001),(0.,0.)):
            c=ExperimentConfig();c.simulation.rest_offset_m=rest;c.simulation.contact_offset_m=contact
            with self.assertRaises(ValueError):c.validate()
        c=ExperimentConfig();c.object.rotation_rpy_range_deg=((0.,1.),)
        with self.assertRaises(ValueError):c.validate()

if __name__=='__main__':unittest.main()
