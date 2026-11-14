"""Separate clamp slip, body motion and local elastic strain in recorded FEM.

No displacement amplification or physics changes. Local stretch is invariant
to rigid rotation; the best rigid fit is a diagnostic, not a correction.
"""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np
try:
    from benchmarks.episode_reader import open_frame
except ModuleNotFoundError:
    from episode_reader import open_frame

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.mesh import load_stl, tet_volumes, contains_points
from shared.config import MeshConfig, resolve_asset
from backends.isaac_sim.render_surface import bind_points


def rigid_fit(reference, current):
    a, b = reference.mean(axis=0), current.mean(axis=0)
    u, _, vt = np.linalg.svd((reference - a).T @ (current - b))
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    predicted = (reference - a) @ rotation + b
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return predicted, float(angle)


def element_stretches(reference, current, tets):
    a = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    b = np.transpose(current[tets[:, 1:]] - current[tets[:, :1]], (0, 2, 1))
    gradient = b @ np.linalg.inv(a)
    return np.linalg.svd(gradient, compute_uv=False), np.linalg.det(gradient)


def audit(folder, output, maximum_pin_error_mm=.01):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recording = folder / 'recording'
    episode = json.loads((recording / 'episode.json').read_text())
    result = json.loads((folder / 'result.json').read_text())
    if episode['status'] != 'complete' or not episode['frames']:
        raise ValueError('A complete recording is required')
    if not np.isfinite(maximum_pin_error_mm) or maximum_pin_error_mm <= 0:
        raise ValueError('Pin tolerance must be positive and finite')
    held = [r for r in episode['frames'] if r['phase'] in ('hold', 'verify_hold')]
    unloaded = [r for r in episode['frames'] if r['phase'] in ('settle', 'approach')]
    if not held or not unloaded:
        raise ValueError('Both precontact and hold states are required')
    row, before = held[-1], unloaded[-1]
    config = MeshConfig(**result['config']['mesh'])
    vertices, faces = load_stl(resolve_asset(config.stl), config.scale_to_m)
    stl_volume = abs(np.einsum('ij,ij->i', vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6)
    references = np.load(recording / 'finger_reference.npz', allow_pickle=False)
    frames = ExitStack()
    state = frames.enter_context(open_frame(recording, row, include_images=False))
    baseline = frames.enter_context(open_frame(recording, before, include_images=False))
    sides, arrays = {}, {}
    edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    # STL ribs are at constant Z; longitudinal rails must not be reported as ribs.
    rib_mask = ((np.abs(vertices[edges[:, 0], 2] - vertices[edges[:, 1], 2]) < 1e-7)
                & (vertices[edges[:, 0], 2] > .01)
                & (vertices[edges[:, 0], 2] < .07)
                & (np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) > .01))
    rib_edges = edges[rib_mask]
    fraction = np.linspace(0, 1, 33)[None, :, None]
    samples = vertices[rib_edges[:, 0], None] * (1 - fraction) + vertices[rib_edges[:, 1], None] * fraction
    max_pin = [0., 0.]
    minimum_jacobian = [float('inf'), float('inf')]
    reference_volumes = {name: tet_volumes(references['finger_reference_' + name],
                                          references['finger_tets_' + name]) for name in ('left', 'right')}
    for frame_row in episode['frames']:
        with open_frame(recording, frame_row, include_images=False) as f:
            for side, name in enumerate(('left', 'right')):
                pose = f['finger_mounts'][side]
                x = (f['finger_nodes_' + name] - pose[:3, 3]) @ pose[:3, :3]
                ref, pin = references['finger_reference_' + name], references['finger_pin_' + name]
                if x.shape != ref.shape or not np.isfinite(x).all():
                    raise ValueError('Invalid recorded FEM nodes')
                max_pin[side] = max(max_pin[side], float(np.linalg.norm(x[pin] - ref[pin], axis=1).max() * 1000))
                relative_volume = tet_volumes(x, references['finger_tets_' + name]) / reference_volumes[name]
                minimum_jacobian[side] = min(minimum_jacobian[side], float(relative_volume.min()))
    for side, name in enumerate(('left', 'right')):
        rest = references['finger_reference_' + name]
        pin, tets = references['finger_pin_' + name], references['finger_tets_' + name]
        def local(f):
            pose = f['finger_mounts'][side]
            return (f['finger_nodes_' + name] - pose[:3, 3]) @ pose[:3, :3]
        current, precontact = local(state), local(baseline)
        displacement = np.linalg.norm(current - precontact, axis=1) * 1000
        prediction, angle = rigid_fit(precontact[~pin], current[~pin])
        residual = np.linalg.norm(current[~pin] - prediction, axis=1) * 1000
        stretches, jacobian = element_stretches(rest, current, tets)
        volumes = abs(tet_volumes(rest, tets))
        centroid_inside = contains_points(vertices, faces, rest[tets].mean(axis=1))
        strain = np.max(abs(stretches - 1), axis=1)
        regions = []
        height = (rest[:, 2] - rest[:, 2].min()) * 1000
        for lo, hi in ((0, 4.1), (4.1, 6.1), (6.1, 10), (10, 20), (20, 40), (40, 60), (60, 90)):
            mask = (height >= lo) & (height < hi)
            regions.append(dict(height_mm=[lo, hi], nodes=int(mask.sum()),
                pinned_nodes=int((mask & pin).sum()),
                maximum_contact_displacement_mm=float(displacement[mask].max()) if mask.any() else None))
        ribs = {}
        if len(samples):
            from types import SimpleNamespace
            indices, weights = bind_points(SimpleNamespace(vertices=rest, tets=tets), samples.reshape(-1, 3))
            curve = np.einsum('ni,nij->nj', weights, current[indices]).reshape(samples.shape)
            chord = curve[:, :1] * (1 - fraction) + curve[:, -1:] * fraction
            deviation = np.linalg.norm(curve - chord, axis=-1) * 1000
            worst = np.argmax(deviation.max(axis=1))
            ribs = dict(edges_checked=len(rib_edges), maximum_curve_to_chord_mm=float(deviation.max()),
                        maximum_negative_embedding_weight=float(np.maximum(-weights, 0).max()),
                        worst_reference_endpoints_mm=(vertices[rib_edges[worst]] * 1000).tolist())
            arrays[name + '_rib_reference_m'] = samples[worst]
            arrays[name + '_rib_current_m'] = curve[worst]
        sides[name] = dict(nodes=len(rest), tets=len(tets), pinned_nodes=int(pin.sum()),
            maximum_pinned_node_error_mm=max_pin[side],
            fem_volume_to_stl_ratio=float(volumes.sum() / stl_volume),
            reference_tet_centroids_outside_stl=int((~centroid_inside).sum()),
            maximum_contact_displacement_mm=float(displacement.max()),
            best_free_body_rigid_fit_rotation_deg=angle,
            nonrigid_residual_rms_mm=float(np.sqrt(np.mean(residual**2))),
            nonrigid_residual_max_mm=float(residual.max()),
            absolute_principal_stretch_minus_one_percentiles=dict(zip(('p50', 'p95', 'p99', 'max'),
                np.percentile(strain, [50, 95, 99, 100]).tolist())),
            volume_fraction_above_one_percent_stretch=float(volumes[strain > .01].sum() / volumes.sum()),
            minimum_deformation_jacobian=float(jacobian.min()),
            minimum_deformation_jacobian_all_frames=minimum_jacobian[side],
            regions=regions, transverse_ribs=ribs)
        arrays.update({name + '_rest_m': rest, name + '_current_m': current,
                       name + '_pin': pin, name + '_precontact_m': precontact})
    report = dict(episode=str(folder), held_frame=row, precontact_frame=before,
        maximum_allowed_pin_error_mm=maximum_pin_error_mm,
        clamp_valid=max(max_pin) <= maximum_pin_error_mm,
        positive_element_volumes=min(minimum_jacobian) > 0,
        stl_volume_mm3=float(stl_volume * 1e9), sides=sides,
        scope='Clamp error and element Jacobian over all frames; rigid fit, stretch and transverse-rib curvature at final hold. '
              'Volume agreement and positive Jacobians do not establish mesh convergence or real-world accuracy.')
    (output / 'clamp_deformation.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / 'clamp_deformation.npz', **arrays)
    references.close(); frames.close()
    return report


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    report = json.loads((output / 'clamp_deformation.json').read_text())
    with np.load(output / 'clamp_deformation.npz', allow_pickle=False) as a:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
        for ax, name in zip(axes[:2], ('left', 'right')):
            rest, current, pin = a[name + '_rest_m'] * 1000, a[name + '_current_m'] * 1000, a[name + '_pin']
            ax.scatter(rest[:, 0], rest[:, 2], s=1, c='.75', alpha=.25, label='Reference')
            ax.scatter(current[~pin, 0], current[~pin, 2], s=1, c='tab:blue', alpha=.3, label='Actual FEM')
            ax.scatter(current[pin, 0], current[pin, 2], s=2, c='tab:red', label='Pinned nodes')
            ax.set(title=name.capitalize() + ' / mount frame / scale 1:1', xlabel='X (mm)', ylabel='Z (mm)', aspect='equal')
            ax.legend(markerscale=3)
        ax = axes[2]
        for name, color in (('left', 'tab:blue'), ('right', 'tab:orange')):
            if name + '_rib_reference_m' in a:
                ref, cur = a[name + '_rib_reference_m'] * 1000, a[name + '_rib_current_m'] * 1000
                ax.plot(ref[:, 0], ref[:, 1], '--', c=color, alpha=.5)
                ax.plot(cur[:, 0], cur[:, 1], c=color, label=name + ' deformed')
        ax.set(title='Most curved transverse rib / scale 1:1', xlabel='X (mm)', ylabel='Y (mm)', aspect='equal')
        ax.legend()
        fig.suptitle('Recorded FEM deformation; no displacement amplification')
        fig.savefig(output / 'clamp_deformation.png', dpi=160)
        plt.close(fig)


def profile_video(folder, output):
    """Project actual FEM nodes in the mount frame, at original displacement scale."""
    import cv2
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    folder, output = Path(folder), Path(output)
    recording = folder / 'recording'
    episode = json.loads((recording / 'episode.json').read_text())
    reference = np.load(recording / 'finger_reference.npz', allow_pickle=False)
    figure = Figure(figsize=(12.8, 7.2), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    artists = []
    for ax, name, color in zip(axes, ('left', 'right'), ('tab:blue', 'tab:orange')):
        rest = reference['finger_reference_' + name] * 1000
        pin = reference['finger_pin_' + name]
        ax.scatter(rest[:, 0], rest[:, 2], s=.7, c='.65', label='Reference FEM nodes')
        artist = ax.scatter(rest[~pin, 0], rest[~pin, 2], s=.7, c=color, label='Current FEM nodes')
        ax.scatter(rest[pin, 0], rest[pin, 2], s=1, c='tab:red', label='Rigidly pinned region')
        ax.set(xlim=(-25, 25), ylim=(-4, 94), aspect='equal', xlabel='Mount X (mm)', ylabel='Mount Z (mm)')
        ax.legend(loc='upper right', fontsize=7, markerscale=2)
        artists.append(artist)
    figure.subplots_adjust(top=.84, bottom=.08, wspace=.08)
    title = figure.suptitle('', fontsize=12)
    path = output / 'deformation_profile.partial.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), episode['fps'], (1280, 720))
    if not writer.isOpened():
        raise RuntimeError('Cannot create profile video')
    try:
        for row in episode['frames']:
            with open_frame(recording, row) as f:
                for side, (ax, artist, name) in enumerate(zip(axes, artists, ('left', 'right'))):
                    pose = f['finger_mounts'][side]
                    local = (f['finger_nodes_' + name] - pose[:3, 3]) @ pose[:3, :3]
                    rest, pin = reference['finger_reference_' + name], reference['finger_pin_' + name]
                    artist.set_offsets(local[~pin][:, [0, 2]] * 1000)
                    maximum = np.linalg.norm(local - rest, axis=1).max() * 1000
                    ax.set_title(f'{name.capitalize()}: maximum node displacement {maximum:.2f} mm')
                command = float(f['width_parameter_m']) * 1000
            title.set_text(f'Actual FEM side projections / {row["time_s"]:.2f}s / {row["phase"]} / command {command:.1f} mm\n'
                           'Rigid mount motion removed; displacement scale x1; red = pinned nodes')
            canvas.draw()
            writer.write(cv2.cvtColor(np.asarray(canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        reference.close()
    path.rename(output / 'deformation_profile.mp4')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-pin-error-mm', type=float, default=.01)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--profile-video', action='store_true')
    args = parser.parse_args()
    report = audit(args.episode, args.output, args.maximum_pin_error_mm)
    if args.plot:
        plot(args.output)
    if args.profile_video:
        profile_video(args.episode, args.output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['clamp_valid'] and report['positive_element_volumes'] else 1)
