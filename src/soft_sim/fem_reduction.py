"""Remove fully constrained FEM work while preserving the full collision mesh."""
from dataclasses import dataclass

import numpy as np

from soft_sim.mesh import TetMesh, tet_volumes


def base_pin_mask(local_nodes, height_m, pitch_m):
    """Keep the intended support layer despite cooking quantization."""
    tolerance = .01 * pitch_m
    return local_nodes[:, 2] <= local_nodes[:, 2].min() + height_m + tolerance, tolerance


def reduce_fixed_cells(mesh, pin):
    mesh.validate()
    pin = np.asarray(pin, dtype=bool)
    if pin.shape != (len(mesh.vertices),) or not pin.any() or pin.all():
        raise ValueError('Select a fixed region and leave free vertices')
    fixed = pin[mesh.tets].all(axis=1)
    keep = ~fixed
    covered = np.zeros(len(mesh.vertices), dtype=bool)
    covered[mesh.tets[keep].ravel()] = True
    while not covered.all():
        counts = (~covered[mesh.tets]).sum(axis=1)
        candidates = np.flatnonzero(fixed & ~keep & (counts > 0) & (counts < 4))
        if not len(candidates):
            raise ValueError('Cannot cover the original vertices from retained cells')
        chosen = int(candidates[np.argmax(counts[candidates])])
        keep[chosen] = True
        covered[mesh.tets[chosen]] = True
    retained = np.flatnonzero(keep)
    fixed_candidates = np.flatnonzero(keep & fixed)
    if not len(fixed_candidates):
        raise ValueError('No fully fixed support tetrahedron remains')
    anchor_original = int(fixed_candidates[np.argmax(tet_volumes(mesh.vertices, mesh.tets)[fixed_candidates])])
    anchor = int(np.flatnonzero(retained == anchor_original)[0])
    reduced = TetMesh(mesh.vertices.copy(), mesh.tets[keep].copy(), dict(mesh.metadata))
    # Connectivity is checked explicitly: a vertex cover alone is insufficient
    # to guarantee a connected volume for an arbitrary input asset.
    reduced.validate()
    reduced.metadata.update(volume_m3=reduced.volume,
                            dirichlet_elimination=dict(original_fem_tets=len(mesh.tets),
                                original_fem_nodes=len(mesh.vertices),
                                removed_fully_fixed_tets=int((~keep).sum()), support_tet=anchor,
                                all_original_vertices_retained=True))
    if 'stl_volume_m3' in mesh.metadata:
        reduced.metadata['volume_relative_error'] = reduced.volume/mesh.metadata['stl_volume_m3']-1
    return reduced, retained, anchor


@dataclass(frozen=True)
class FemMeshes:
    simulation: TetMesh
    collision: TetMesh
    fixed_support_tet: int | None = None


def prepare_fem_meshes(mesh, config):
    """Derive mechanics from the original full mesh; never reduce collision."""
    if not config.eliminate_fixed_cells:
        return FemMeshes(mesh, mesh)
    if 'dirichlet_elimination' in mesh.metadata:
        raise ValueError('Fixed-cell elimination requires the original full FEM mesh; '
                         'do not pass a previously reduced --fem-mesh')
    pin, _ = base_pin_mask(mesh.vertices, config.pin_height_m, config.voxel_pitch_m)
    if not pin[mesh.tets].all(axis=1).any():
        return FemMeshes(mesh, mesh)
    reduced, _, anchor = reduce_fixed_cells(mesh, pin)
    return FemMeshes(reduced, mesh, anchor)


def bind_fixed_surface(mesh, config, surface, indices, weights, anchor):
    """Make fixed STL points follow one fully constrained reference cell."""
    if not 0 <= anchor < len(mesh.tets):
        raise ValueError('Fixed surface support tetrahedron is out of bounds')
    pin, _ = base_pin_mask(mesh.vertices, config.pin_height_m, config.voxel_pitch_m)
    support = mesh.tets[anchor]
    if not pin[support].all():
        raise ValueError('Fixed surface support must contain only fixed nodes')
    mask, _ = base_pin_mask(surface, config.pin_height_m, config.voxel_pitch_m)
    cell = mesh.vertices[support]
    matrix = np.stack([cell[i]-cell[0] for i in (1, 2, 3)], axis=-1)
    tail = np.linalg.solve(matrix, (surface[mask]-cell[0]).T).T
    indices[mask] = support
    weights[mask] = np.c_[1-tail.sum(axis=1), tail]
    if not np.allclose(np.einsum('ni,nij->nj',weights,mesh.vertices[indices]),surface,atol=1e-8):
        raise ValueError('Invalid fixed surface embedding')
    return indices, weights
