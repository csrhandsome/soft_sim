"""Refinement must preserve the supplied solid, including its cage openings."""
from types import SimpleNamespace
import unittest

import numpy as np

from soft_sim.config import resolve_asset
from soft_sim.mesh import contains_points, load_stl, refine_surface
from soft_sim.sim.render_surface import bind_points


def signed_volume(vertices, faces):
    triangles = vertices[faces]
    return np.einsum('ij,ij->i', triangles[:, 0],
                     np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6


class SurfaceRefinement(unittest.TestCase):
    def test_stl_refinement_preserves_closed_solid_and_openings(self):
        vertices, faces = load_stl(resolve_asset('assets/meshes/soft_finger.stl'))
        refined, triangles = refine_surface(vertices, faces, .004)
        edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(counts == 2), 'Refinement introduced a crack or nonmanifold edge')
        self.assertLessEqual(np.linalg.norm(refined[unique[:, 0]] - refined[unique[:, 1]], axis=1).max(), .004 * (1 + 1e-12))
        self.assertAlmostEqual(signed_volume(refined, triangles) / signed_volume(vertices, faces), 1., places=12)
        np.testing.assert_allclose(refined.min(axis=0), vertices.min(axis=0))
        np.testing.assert_allclose(refined.max(axis=0), vertices.max(axis=0))
        points = np.random.default_rng(17).uniform(vertices.min(axis=0), vertices.max(axis=0), (48, 3))
        np.testing.assert_array_equal(contains_points(refined, triangles, points), contains_points(vertices, faces, points))

    def test_partial_edge_splits_preserve_winding_and_area(self):
        # A scalene triangle exercises the one-, two- and three-edge cases
        # through successive conforming splits.
        vertices = np.array([[0., 0., 0.], [4., 0., 0.], [1., 2., 0.]])
        for limit in (3.8, 2.5, 1., .3):
            refined, triangles = refine_surface(vertices, [[0, 1, 2]], limit)
            t = refined[triangles]
            normals = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            self.assertTrue(np.all(normals[:, 2] > 0))
            self.assertAlmostEqual(normals[:, 2].sum(), 8.)

    def test_zero_preserves_baseline_and_invalid_limits_fail(self):
        vertices = np.eye(3)
        faces = np.array([[0, 1, 2]])
        refined, triangles = refine_surface(vertices, faces, 0.)
        np.testing.assert_array_equal(refined, vertices)
        np.testing.assert_array_equal(triangles, faces)
        for limit in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                refine_surface(vertices, faces, limit)

    def test_display_embedding_follows_fem_without_rigid_pose_error(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        mesh = SimpleNamespace(vertices=vertices, tets=np.array([[0, 1, 2, 3]]))
        points = np.array([[.1, .2, .3], [.3, .2, .1], [0., 0., 0.]])
        indices, weights = bind_points(mesh, points)
        # A combined stretch, shear, rotation and translation must agree
        # with applying the same affine map directly to the surface points.
        affine = np.array([[0., -1., .1], [1.1, 0., .2], [0., 0., .9]])
        translated = vertices @ affine.T + [2., 3., 4.]
        displayed = np.einsum('ni,nij->nj', weights, translated[indices])
        np.testing.assert_allclose(displayed, points @ affine.T + [2., 3., 4.], atol=1e-14)

    def test_embedding_finds_containing_tet_beyond_nearest_centroids(self):
        long_tet = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
        # Small cells just across the X=0 boundary have closer centroids but
        # cannot contain this point. The long cell must be selected instead.
        cells = [long_tet]
        for i in range(120):
            origin = np.array([-.02 - i * .001, .1, .1])
            cells.append(origin + np.array([[0., 0., 0.], [-.001, 0., 0.], [0., .001, 0.], [0., 0., .001]]))
        mesh = SimpleNamespace(vertices=np.concatenate(cells), tets=np.arange(4 * len(cells)).reshape(-1, 4))
        indices, weights = bind_points(mesh, np.array([[.001, .1, .1]]))
        np.testing.assert_array_equal(indices, [[0, 1, 2, 3]])
        self.assertGreaterEqual(weights.min(), 0.)


if __name__ == '__main__':
    unittest.main()
