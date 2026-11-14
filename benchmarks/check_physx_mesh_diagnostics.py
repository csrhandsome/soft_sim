"""Explain legacy PhysX mass diagnostics without suppressing engine errors.

PhysX 5.6's updateNodeInverseVolumes reports det <= 1e-9 as "degenerate
or inverted", including small positive tetrahedra in meter-scale meshes.
This audit keeps that threshold separate from actual nonpositive volumes.
Cooking setup errors remain separately reported and are not marked resolved.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.mesh import tet_volumes

SDK_SOURCE = "https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-physx-5.6.1/physx/source/physxextensions/src/ExtDeformableVolumeExt.cpp"


def rest_cell_diagnostics(vertices, tets):
    determinants = 6*tet_volumes(np.asarray(vertices, dtype=np.float64), tets)
    return dict(elements=len(tets), minimum_determinant_m3=float(determinants.min()),
        nonpositive_or_nonfinite_cells=int(np.count_nonzero(~np.isfinite(determinants) | (determinants <= 0))),
        small_positive_cells=int(np.count_nonzero((determinants > 0) & (determinants <= 1e-9))))


def audit(folder, log_path=None):
    folder = Path(folder)
    sides = []
    with np.load(folder / "initial_fem.npz", allow_pickle=False) as archive:
        for key in archive.files:
            if "_reference_" not in key:
                continue
            prefix, side = key.split("_reference_")
            rest, tets = archive[key], archive[f"{prefix}_tets_{side}"]
            pose = archive[f"{prefix}_mounts"][int(side)]
            nodes = archive[f"{prefix}_nodes_{side}"]
            local = (nodes-pose[:3, 3])@pose[:3, :3]
            entry = rest_cell_diagnostics(rest, tets)
            entry.update(robot=prefix, side=int(side), minimum_initial_jacobian=float(
                (tet_volumes(local, tets)/tet_volumes(rest, tets)).min()))
            sides.append(entry)
    lines = Path(log_path or folder.with_suffix(".log")).read_text(errors="replace").splitlines()
    marker = next((i for i, line in enumerate(lines) if "TASK_STAGE after_reset" in line), None)
    messages = [i for i, line in enumerate(lines) if "updateNodeInverseVolumes(): tetrahedron is degenerate or inverted" in line]
    expected = sum(side["small_positive_cells"] for side in sides)
    rest_valid = all(side["nonpositive_or_nonfinite_cells"] == 0 for side in sides)
    explained = bool(marker is not None and rest_valid and len(messages) == expected
                     and all(i < marker for i in messages))
    report = dict(sdk_source=SDK_SOURCE, determinant_threshold_m3=1e-9,
        rest_mesh_positive=rest_valid, sides=sides,
        expected_small_positive_mass_diagnostics=expected,
        observed_mass_diagnostics=len(messages),
        small_positive_threshold_explains_mass_diagnostics=explained,
        mass_diagnostics_after_reset=sum(i > marker for i in messages) if marker is not None else None,
        cooking_setup_error_count=sum("failed to setup cooking params" in line for line in lines),
        scope="Explains only matching initialization mass messages. Does not clear cooking errors, validate later physics, or calibrate the material.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    report = audit(args.folder, args.log)
    destination = args.output or args.folder / "physx_mesh_diagnostics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
