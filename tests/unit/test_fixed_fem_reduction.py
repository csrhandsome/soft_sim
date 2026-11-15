"""Check elastic energy and fixed-surface kinematics, beyond mesh counts."""
import unittest

import numpy as np

from soft_sim.mesh import TetMesh, tet_volumes
from soft_sim.config import MeshConfig
from soft_sim.fem_reduction import bind_fixed_surface, prepare_fem_meshes, reduce_fixed_cells


def lattice():
    points = np.array([(x, y, z) for z in range(4) for y in range(4) for x in range(4)], dtype=float)*.002
    cells = []
    pattern = np.array([(0,1,3,7), (0,3,2,7), (0,2,6,7), (0,6,4,7), (0,4,5,7), (0,5,1,7)])
    for z in range(3):
        for y in range(3):
            for x in range(3):
                corners = np.array([(z+dz)*16+(y+dy)*4+x+dx
                                    for dz,dy,dx in [(0,0,0),(0,0,1),(0,1,0),(0,1,1),
                                                     (1,0,0),(1,0,1),(1,1,0),(1,1,1)]])
                cells.extend(corners[pattern])
    cells = np.asarray(cells, dtype=np.int32)
    inverted = tet_volumes(points, cells) < 0
    cells[inverted] = cells[inverted][:,[1,0,2,3]]
    return TetMesh(points, cells, {}).validate()


def corotated_energy(mesh, positions):
    rest, current = mesh.vertices[mesh.tets], positions[mesh.tets]
    dm = np.stack([rest[:,i]-rest[:,0] for i in (1,2,3)], axis=-1)
    ds = np.stack([current[:,i]-current[:,0] for i in (1,2,3)], axis=-1)
    deformation = ds @ np.linalg.inv(dm)
    u, _, vt = np.linalg.svd(deformation)
    rotation = u @ vt
    # The precise positive Lamé values are immaterial to equivalence: every
    # retained cell and its rest gradient must match, removed cells are rigid.
    density = 2*np.sum((deformation-rotation)**2, axis=(1,2)) + 7*(np.linalg.det(deformation)-1)**2
    return float(np.dot(density, tet_volumes(mesh.vertices, mesh.tets)))


class FixedFemReduction(unittest.TestCase):
    def setUp(self):
        self.mesh = lattice()
        self.pin = self.mesh.vertices[:,2] <= .002 + 1e-10
        self.reduced, self.retained, self.anchor = reduce_fixed_cells(self.mesh, self.pin)

    def test_arbitrary_free_deformation_preserves_total_elastic_energy(self):
        rng = np.random.default_rng(821)
        displacement = rng.normal(0, 2e-5, self.mesh.vertices.shape)
        displacement[self.pin] = 0
        positions = self.mesh.vertices + displacement
        first = corotated_energy(self.mesh, positions)
        second = corotated_energy(self.reduced, positions)
        self.assertGreater(first, 0)
        self.assertAlmostEqual(first/second, 1., places=12)
        self.assertLess(len(self.reduced.tets), len(self.mesh.tets))
        np.testing.assert_array_equal(self.reduced.vertices, self.mesh.vertices)

    def test_energy_equivalence_survives_rigid_mount_motion(self):
        displacement = np.zeros_like(self.mesh.vertices)
        displacement[~self.pin,0] = .0001
        theta = .7
        rotation = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        positions = (self.mesh.vertices+displacement) @ rotation.T + [1.1,-.7,.9]
        a, b = corotated_energy(self.mesh, positions), corotated_energy(self.reduced, positions)
        self.assertAlmostEqual(a/b, 1., places=10)

    def test_fixed_support_extrapolation_tracks_a_rigid_surface_exactly(self):
        points = np.array([[-.02,.03,0],[.04,-.03,.001],[0,0,.002]])
        indices, weights = bind_fixed_surface(self.reduced, MeshConfig(pin_height_m=.002),
            points, np.zeros((3,4),dtype=int), np.zeros((3,4)), self.anchor)
        rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        moved = self.reduced.vertices @ rotation.T + [2.,-.3,1.]
        actual = np.einsum('ni,nij->nj',weights,moved[indices])
        np.testing.assert_allclose(actual, points @ rotation.T + [2.,-.3,1.], atol=1e-12, rtol=0)
        self.assertTrue(self.pin[self.reduced.tets[self.anchor]].all())

    def test_default_meshes_keep_full_collision_and_all_free_cells(self):
        original_tets = self.mesh.tets.copy()
        meshes = prepare_fem_meshes(self.mesh, MeshConfig(pin_height_m=.002))
        self.assertIs(meshes.collision, self.mesh)
        np.testing.assert_array_equal(meshes.collision.tets, original_tets)
        np.testing.assert_array_equal(meshes.simulation.vertices, self.mesh.vertices)
        free_cells = ~self.pin[original_tets].all(axis=1)
        retained_free = ~self.pin[meshes.simulation.tets].all(axis=1)
        np.testing.assert_array_equal(meshes.simulation.tets[retained_free],original_tets[free_cells])
        self.assertLess(len(meshes.simulation.tets),len(original_tets))
        self.assertTrue(self.pin[meshes.simulation.tets[meshes.fixed_support_tet]].all())

    def test_disabling_elimination_restores_full_mechanics(self):
        meshes = prepare_fem_meshes(self.mesh,MeshConfig(pin_height_m=.002,eliminate_fixed_cells=False))
        self.assertIs(meshes.simulation,self.mesh)
        self.assertIs(meshes.collision,self.mesh)
        self.assertIsNone(meshes.fixed_support_tet)

    def test_no_fully_fixed_cells_leave_the_mesh_intact(self):
        meshes = prepare_fem_meshes(self.mesh,MeshConfig(pin_height_m=.0001))
        self.assertIs(meshes.simulation,self.mesh)
        self.assertIsNone(meshes.fixed_support_tet)

    def test_pre_reduced_mesh_cannot_silently_replace_full_collision(self):
        with self.assertRaisesRegex(ValueError,'original full FEM'):
            prepare_fem_meshes(self.reduced,MeshConfig(pin_height_m=.002))

    def test_fixed_surface_binding_preserves_free_surface_weights(self):
        config = MeshConfig(pin_height_m=.002)
        support = self.reduced.tets[~self.pin[self.reduced.tets].all(axis=1)][0]
        surface = np.array([[.001,.001,0],self.reduced.vertices[support].mean(axis=0)])
        indices = np.tile(support,(2,1))
        weights = np.full((2,4),.25)
        bound_indices,bound_weights = bind_fixed_surface(self.reduced,config,surface,indices,weights,self.anchor)
        np.testing.assert_array_equal(bound_indices[1],support)
        np.testing.assert_array_equal(bound_weights[1],np.full(4,.25))
        np.testing.assert_allclose(np.einsum('ni,nij->nj',bound_weights,self.reduced.vertices[bound_indices]),surface)

    def test_a_free_cell_cannot_anchor_the_fixed_surface(self):
        anchor = int(np.flatnonzero(~self.pin[self.reduced.tets].all(axis=1))[0])
        with self.assertRaisesRegex(ValueError,'only fixed nodes'):
            bind_fixed_surface(self.reduced,MeshConfig(pin_height_m=.002),np.zeros((1,3)),
                               np.zeros((1,4),dtype=int),np.zeros((1,4)),anchor)

    def test_invalid_boundary_selection_is_rejected(self):
        for mask in (np.zeros(len(self.pin),dtype=bool), np.ones(len(self.pin),dtype=bool)):
            with self.assertRaises(ValueError):
                reduce_fixed_cells(self.mesh, mask)
