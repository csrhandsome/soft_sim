"""Measure bending hidden by straight STL edges using a recorded FEM state.

This diagnoses surface sampling, not material accuracy or contact convergence.
No simulation steps or image stabilization are performed.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from soft_sim.sim.render_surface import bind_points
from soft_sim.config import MeshConfig, resolve_asset
from soft_sim.mesh import load_stl, refine_surface


def check(episode_directory, output, surface_max_edge_m=.002):
    root, output = Path(episode_directory), Path(output)
    from soft_sim.recording.reader import capture_directory, first_existing, run_directory
    recording = capture_directory(root)
    run_root = run_directory(recording)
    episode = json.loads((recording / 'episode.json').read_text())
    result = json.loads(first_existing(run_root / 'results/task.json', run_root / 'result.json').read_text())
    if episode['status'] != 'complete':
        raise ValueError('A complete recorded episode is required')
    config = MeshConfig(**result['config']['mesh'])
    vertices, faces = load_stl(resolve_asset(config.stl), config.scale_to_m)
    refined, refined_faces = refine_surface(vertices, faces, surface_max_edge_m)
    edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    long_edges = edges[lengths > .01]
    if not len(long_edges):
        raise ValueError('No STL edges longer than 10 mm to inspect')
    rows = [row for row in episode['frames'] if row['phase'] in ('hold', 'verify_hold')]
    if not rows:
        raise ValueError('No held state in the recording')
    row = rows[-1]
    fraction = np.linspace(0., 1., 17)[None, :, None]
    samples = vertices[long_edges[:, 0], None] * (1 - fraction) + vertices[long_edges[:, 1], None] * fraction
    sides, worst = {}, None
    with np.load(recording / 'finger_reference.npz', allow_pickle=False) as reference, np.load(recording / row['file'], allow_pickle=False) as frame:
        for side, name in enumerate(('left', 'right')):
            mesh = SimpleNamespace(vertices=reference['finger_reference_' + name], tets=reference['finger_tets_' + name])
            indices, weights = bind_points(mesh, samples.reshape(-1, 3))
            pose = frame['finger_mounts'][side]
            nodes = (frame['finger_nodes_' + name] - pose[:3, 3]) @ pose[:3, :3]
            deformed = np.einsum('ni,nij->nj', weights, nodes[indices]).reshape(samples.shape)
            chord = deformed[:, :1] * (1 - fraction) + deformed[:, -1:] * fraction
            error = np.linalg.norm(deformed - chord, axis=-1) * 1000
            index = error.max(axis=1).argmax()
            maximum = float(error[index].max())
            sides[name] = dict(maximum_curve_to_chord_mm=maximum,
                               p95_edge_maximum_curve_to_chord_mm=float(np.percentile(error.max(axis=1), 95)),
                               maximum_negative_embedding_weight=float(np.maximum(-weights, 0).max()),
                               worst_edge_source_endpoints_mm=(vertices[long_edges[index]] * 1000).tolist())
            if worst is None or maximum > worst[0]:
                worst = (maximum, samples[index], deformed[index], chord[index])
    report = dict(reference_episode=str(root), frame=row, scope=__doc__.strip(),
                  material=result['config']['material'],
                  original_surface_vertices=len(vertices), original_surface_triangles=len(faces),
                  maximum_original_edge_mm=float(lengths.max() * 1000), long_edges_checked=len(long_edges),
                  refined_surface_max_edge_m=surface_max_edge_m, refined_vertices=len(refined),
                  refined_triangles=len(refined_faces), sides=sides)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'surface_audit.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / 'beam_curve.npz', reference_m=worst[1], fem_m=worst[2], coarse_chord_m=worst[3])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--surface-max-edge-m', type=float, default=.002)
    args = parser.parse_args()
    print(json.dumps(check(args.episode_directory, args.output, args.surface_max_edge_m), indent=2))
