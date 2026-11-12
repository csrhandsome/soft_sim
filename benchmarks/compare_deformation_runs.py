"""Compare raw RGB at the last held state; label each run's actual conditions."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(episodes, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(episodes), 3, figsize=(14, 4.6 * len(episodes)),
                             squeeze=False, constrained_layout=True)
    rows = []
    for folder, row_axes in zip(map(Path, episodes), axes):
        result = json.loads((folder / 'result.json').read_text())
        episode = json.loads((folder / 'recording/episode.json').read_text())
        if episode['status'] != 'complete':
            raise ValueError('Incomplete episode: ' + str(folder))
        row = next(r for r in reversed(episode['frames']) if r['phase'] in ('hold', 'verify_hold'))
        with np.load(folder / 'recording' / row['file'], allow_pickle=False) as f:
            width = float(f['width_parameter_m']) * 1000
            for ax, key, label in zip(row_axes, ('external_rgb', 'left_rgb', 'right_rgb'),
                                      ('External RGB', 'Left RGB', 'Right RGB')):
                ax.imshow(f[key])
                ax.set_title(label)
                ax.set_axis_off()
        settings = result['deformable_mesh']
        source = settings.get('physics_mesh_source', 'physx_voxel_cooking')
        pin = result['config']['mesh']['pin_height_m'] * 1000
        young = result['config']['material']['young_modulus_pa'] / 1e6
        iterations = result['config']['simulation']['solver_iterations']
        self_contact = settings.get('self_collision', False)
        validation_file = folder / 'recording/validation.json'
        valid = json.loads(validation_file.read_text())['valid'] if validation_file.exists() else None
        status = 'checks passed' if valid else ('diagnostic: checks failed' if valid is False else 'unchecked')
        mesh_label = 'Conforming volume mesh' if source == 'explicit_tetrahedra' else 'Automatic volume mesh'
        description = (f'{folder.name.removeprefix("robodojo_episode_")}\n{mesh_label}\n'
                       f'Closure command {width:.0f} mm; clamp {pin:.0f} mm\n'
                       f'E={young:g} MPa; iterations={iterations}; self-contact={self_contact}\n'
                       f't={row["time_s"]:.2f}s; {status}')
        row_axes[0].set_title(description + '\nExternal RGB', fontsize=10)
        rows.append(dict(episode=str(folder), frame=row, closure_command_mm=width,
                         pin_height_mm=pin, young_modulus_mpa=young, mesh_source=source,
                         solver_iterations=iterations, self_collision=self_contact, validation_passed=valid))
    fig.suptitle('Raw recorded RGB / different closure commands are labelled / no deformation amplification')
    fig.savefig(output / 'comparison_hold.png', dpi=150)
    plt.close(fig)
    (output / 'comparison.json').write_text(json.dumps(rows, indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    compare(args.episodes, args.output)
