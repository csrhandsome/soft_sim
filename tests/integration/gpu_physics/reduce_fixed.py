"""Export the default fixed-cell reduction for independent audits.

The runtime derives this from the original full mesh automatically. An exported
reduced mesh is for experiment adapters and must keep the full collision mesh.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from soft_sim.config import MeshConfig
from soft_sim.mesh import TetMesh
from soft_sim.fem_reduction import base_pin_mask, reduce_fixed_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, required=True)
    parser.add_argument('--pin-height', type=float, default=.004)
    args = parser.parse_args()
    mesh = TetMesh.load(MeshConfig(tetra_mesh=str(args.mesh), pin_height_m=args.pin_height))
    pin, tolerance = base_pin_mask(mesh.vertices, args.pin_height, mesh.metadata['pitch_m'])
    reduced, retained, anchor = reduce_fixed_cells(mesh, pin)
    reduced.metadata['dirichlet_elimination'].update(
        original_mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        pin_height_m=args.pin_height, pin_tolerance_m=tolerance,
        collision_mesh=str(args.mesh.resolve()))
    reduced.save(args.output)
    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.mapping, anchor_tet=np.array(anchor, dtype=np.int32),
                        retained_original_tets=retained)
    print(json.dumps(dict(output=str(args.output), mapping=str(args.mapping),
                         original_tets=len(mesh.tets), reduced_tets=len(reduced.tets),
                         vertices=len(mesh.vertices), free_vertices=int((~pin).sum())), indent=2))


if __name__ == '__main__':
    main()
