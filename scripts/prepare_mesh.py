"""Convert the supplied finger STL into a volume mesh shared by both engines."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from shared.config import ExperimentConfig, resolve_asset
from shared.mesh import TetMesh, load_stl, tet_volumes, voxel_tetrahedralize


def gmsh_tetrahedralize(path, scale, pitch):
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("Gmsh is optional. Run with `uv run --project mujoco --with gmsh python scripts/prepare_mesh.py --method gmsh`.") from exc
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 1)
        gmsh.merge(str(path))
        gmsh.model.mesh.classifySurfaces(np.deg2rad(40), True, True, np.pi)
        gmsh.model.mesh.createGeometry()
        surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMin', pitch / scale / 2)
        gmsh.option.setNumber('Mesh.MeshSizeMax', pitch / scale)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.model.mesh.generate(3)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        types, _, node_tags = gmsh.model.mesh.getElements(3)
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        tets = []
        for kind, nodes in zip(types, node_tags):
            if kind != 4:
                raise ValueError(f"Expected linear tetrahedra, got Gmsh element type {kind}")
            tets.extend([lookup[int(tag)] for tag in nodes])
        tets = np.asarray(tets, dtype=np.int32).reshape(-1, 4)
        points = np.asarray(coordinates).reshape(-1, 3) * scale
        used, inverse = np.unique(tets, return_inverse=True)
        return points[used], inverse.reshape(-1, 4)
    finally:
        gmsh.finalize()


def tetgen_tetrahedralize(vertices, faces, pitch):
    """Conforming PLC mesh, as used by DiffPD's tet_mesh.tetrahedralize.

    TetGen may split source triangles to improve quality, but keeps the PLC
    surface. pitch controls maximum tet volume, not a guaranteed edge length.
    """
    import tetgen
    generator = tetgen.TetGen(np.asarray(vertices), np.asarray(faces, dtype=np.int32))
    points, tets = generator.tetrahedralize(minratio=1.4, mindihedral=10.,
        maxvolume=pitch**3 / 6, fixedvolume=True, steinerleft=-1, quiet=True)
    used, inverse = np.unique(tets, return_inverse=True)
    return points[used], inverse.reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--method', choices=['voxel', 'gmsh', 'tetgen'], default='voxel')
    parser.add_argument('--pitch', type=float, help='Voxel edge / Gmsh maximum edge length, meters')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    if args.pitch is not None:
        config.mesh.voxel_pitch_m = args.pitch
    config.validate()
    path = resolve_asset(config.mesh.stl)
    vertices, faces = load_stl(path, config.mesh.scale_to_m)
    pitch = config.mesh.voxel_pitch_m
    if args.method == 'gmsh':
        points, tets = gmsh_tetrahedralize(path, config.mesh.scale_to_m, pitch)
    elif args.method == 'tetgen':
        points, tets = tetgen_tetrahedralize(vertices, faces, pitch)
    else:
        points, tets = voxel_tetrahedralize(vertices, faces, pitch)
    inverted = tet_volumes(points, tets) < 0
    tets[inverted] = tets[inverted][:, [1, 0, 2, 3]]
    surface_volume = abs(np.einsum('ij,ij->i', vertices[faces[:, 0]],
                                 np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6)
    metadata = dict(method=args.method, pitch_m=pitch, scale_to_m=config.mesh.scale_to_m,
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    stl_volume_m3=float(surface_volume), stl_bounds_m=[vertices.min(0).tolist(), vertices.max(0).tolist()],
                    approximation=args.method == 'voxel')
    if args.method == 'tetgen':
        metadata.update(maximum_tet_volume_m3=pitch**3 / 6,
                        mesher='tetgen', radius_edge_ratio=1.4, minimum_dihedral_deg=10.)
    mesh = TetMesh(points, tets.astype(np.int32), metadata).validate()
    metadata['volume_m3'] = mesh.volume
    metadata['volume_relative_error'] = mesh.volume / surface_volume - 1
    output = resolve_asset(args.output or config.mesh.tetra_mesh)
    mesh.save(output)
    print(json.dumps(dict(output=str(output), vertices=len(points), tetrahedra=len(tets), **metadata), indent=2))


if __name__ == '__main__':
    main()
