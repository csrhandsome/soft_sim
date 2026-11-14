"""Compare stock and replacement X5 fingers in their actual mounting frame.

Render orthographic silhouettes of one finger, including the replacement
adapter. Static dimensions do not establish grasp success or dynamic clearance.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.x5_soft_tool import MOUNT_X, link_transforms
from benchmarks.x5_mount_geometry import measured_installation, assembly_meshes


def mounted_meshes(scale, profile="measured"):
    links = link_transforms(np.zeros(8))
    slide = np.linalg.inv(links['link6']) @ links['link8']
    stock = trimesh.load(ROOT / 'RoboDojo/Assets/Robots/x5/meshes/link8.STL', force='mesh')
    stock.apply_transform(slide)
    installation = measured_installation(scale) if profile == "measured" else None
    replacement = trimesh.util.concatenate(list(assembly_meshes(scale, 0, installation).values()))
    replacement.apply_transform(slide)
    # X is forward from the slide; Y/Z remain in the shared wrist frame.
    for mesh in (stock, replacement):
        mesh.apply_translation([-MOUNT_X, 0., 0.])
        mesh.apply_scale(1000.)
    return stock, replacement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scales', type=float, nargs='+', default=[.65, .75, .8])
    parser.add_argument('--mount-profile', choices=('measured', 'parallel_legacy'), default='measured')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/generated/x5_measured_finger_size_comparison')
    args = parser.parse_args()
    if any(not np.isfinite(scale) or not .25 <= scale <= 1 for scale in args.scales):
        parser.error('Scales must be finite and in [0.25, 1]')
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, len(args.scales), figsize=(5 * len(args.scales), 8.5), squeeze=False)
    reports = []
    for column, scale in enumerate(args.scales):
        stock, soft = mounted_meshes(scale, args.mount_profile)
        reports.append(dict(scale=scale, stock_extent_xyz_mm=stock.extents.tolist(),
            replacement_with_adapter_extent_xyz_mm=soft.extents.tolist(),
            stock_forward_from_slide_mm=float(stock.bounds[1, 0]),
            replacement_forward_from_slide_mm=float(soft.bounds[1, 0])))
        for row, lateral in enumerate((1, 2)):
            ax = axes[row, column]
            for mesh, color in ((stock, '#b8c0ca'), (soft, '#f87522')):
                triangles = mesh.vertices[mesh.faces][:, :, [0, lateral]]
                ax.add_collection(PolyCollection(triangles, facecolors=color, edgecolors='none'))
            ax.axvline(0, color='#475569', linewidth=.8, linestyle='--')
            ax.set_xlim(-20, 76)
            ax.set_ylim((-51, 3) if lateral == 1 else (-35, 35))
            ax.set_aspect('equal')
            ax.set_xlabel('Forward from slide (mm)')
            ax.set_ylabel(('Closing axis Y' if lateral == 1 else 'Thickness axis Z') + ' (mm)')
            ax.set_title(f'Scale {scale:g}' if row == 0 else f'Stock 71.0 mm / soft {soft.bounds[1, 0]:.1f} mm')
            ax.grid(alpha=.15)
    fig.suptitle('Original X5 finger versus mounted soft finger and adapter', fontsize=15)
    fig.legend(handles=[Patch(color='#b8c0ca', label='Original X5'),
        Patch(color='#f87522', label='Soft finger + adapter')], loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, .06, 1, .94), h_pad=3.)
    fig.savefig(args.output / 'comparison.png', dpi=180)
    fig.savefig(args.output / 'comparison.svg')
    plt.close(fig)
    report = dict(scope=__doc__.strip(), mounting='link8, native slide at zero, shared wrist frame',
                  mount_profile=args.mount_profile, comparisons=reports)
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
