"""Compare held-pose probes on common STL points instead of different FEM nodes.

Offline usage (no Isaac Sim):
    PYTHONPATH=src .venv/bin/python -m tests.integration.gpu_physics.compare_mesh REPORT
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from soft_sim.config import ROOT
from soft_sim.mesh import load_stl, tet_volumes
from soft_sim.sim.render_surface import bind_points


def compare(path, *, allow_timestep_change=False):
    path = Path(path)
    report = json.loads(path.read_text())
    baseline, candidate = report['runs']
    assert baseline['returncode'] == candidate['returncode'] == 0
    a, b = baseline['report'], candidate['report']
    for key in ('fem_iterations', 'scene_min_iterations',
                'single_update', 'profiled', 'follow_backend', 'checkpoint_steps'):
        if key == 'checkpoint_steps' and allow_timestep_change:
            continue
        assert a[key] == b[key], f'Changed more than the mesh: {key}'
    if allow_timestep_change:
        assert np.isclose(a['steps']*a['dt'], b['steps']*b['dt'])
        assert np.isclose(a['warmup_steps']*a['dt'], b['warmup_steps']*b['dt'])
    else:
        for key in ('steps', 'warmup_steps', 'dt'):
            assert a[key] == b[key], f'Changed more than the mesh: {key}'
    assert a['fem_iterations'] == 80 and not a['profiled']
    assert a['cache_pose_error_m'] == b['cache_pose_error_m'] == 0
    config = json.loads((path.parent.parent/'config.json').read_text())
    vertices, _ = load_stl(ROOT/config['mesh']['stl'],
                           config['mesh']['scale_to_m'] * config['task']['finger_scale'])
    pins_below = vertices[:, 2].min() + config['mesh']['pin_height_m'] * config['task']['finger_scale']
    free = vertices[:, 2] > pins_below
    limit = .05
    node_cap = .5 * config['mesh']['voxel_pitch_m'] * config['task']['finger_scale']
    fingers = []
    with np.load(baseline['output'].replace('.json', '.npz')) as old, \
         np.load(candidate['output'].replace('.json', '.npz')) as new:
        steps = old['checkpoint_steps']
        assert len(steps) >= 4
        np.testing.assert_allclose(steps*a['dt'], new['checkpoint_steps']*b['dt'], atol=1e-12, rtol=0)
        late = steps >= .6 * a['steps']
        for arm in range(len(a['mesh_topology'])):
            for side in range(len(a['mesh_topology'][arm])):
                responses, checks = [], []
                for archive, run in ((old, a), (new, b)):
                    rest = archive[f'rest_{arm}_{side}']
                    tets = archive[f'tets_{arm}_{side}']
                    indices, weights = bind_points(SimpleNamespace(vertices=rest, tets=tets), vertices[free])
                    # A rest reconstruction alone also permits extrapolation.
                    assert weights.min() >= -1e-5 and weights.max() <= 1 + 1e-5
                    mount = archive[f'mounts_{arm}'][side]
                    local = (archive[f'history_{arm}_{side}'] - mount[:3, 3]) @ mount[:3, :3]
                    assert np.isfinite(local).all()
                    displacement = local - rest
                    response = np.einsum('ni,tnij->tnj', weights, displacement[:, indices])
                    responses.append(response)
                    volumes = tet_volumes(rest, tets)
                    minimum_jacobian = min(float((tet_volumes(frame, tets)/volumes).min()) for frame in local)
                    pin_error = float(np.linalg.norm(displacement[:, archive[f'pin_{arm}_{side}']], axis=-1).max())
                    topology = run['mesh_topology'][arm][side]
                    assert topology['valid'] and topology['unanchored_nodes'] == 0
                    assert run['fingers'][arm*2+side]['authored_iterations'] == 80
                    checks.append(dict(minimum_checkpoint_jacobian=minimum_jacobian,
                                       maximum_checkpoint_pin_error_m=pin_error,
                                       effective_pin_height_m=topology['effective_pin_height_m'],
                                       tensor_reported_zero_free_rest_elements=topology['tensor_reported_zero_free_rest_elements']))
                first, second = responses
                amplitude_a = np.linalg.norm(first, axis=-1).max(axis=1)
                amplitude_b = np.linalg.norm(second, axis=-1).max(axis=1)
                rms_a = np.sqrt(np.mean(np.sum(first**2, axis=-1), axis=1))
                drift = np.linalg.norm(first-second, axis=-1)
                rms_drift = np.sqrt(np.mean(drift**2, axis=1))
                assert (amplitude_a > 0).all() and (rms_a > 0).all()
                amplitude_change = amplitude_b/amplitude_a - 1
                rms_relative = rms_drift/rms_a
                valid = all(c['minimum_checkpoint_jacobian'] > 0 and
                            c['maximum_checkpoint_pin_error_m'] < 1e-5 for c in checks)
                passes = bool(valid and np.max(np.abs(amplitude_change[late])) <= limit and
                              rms_relative[late].max() <= limit and drift[late].max() <= node_cap)
                fingers.append(dict(arm=arm, side=side, baseline=checks[0], candidate=checks[1],
                    late_baseline_mean_maximum_displacement_m=float(amplitude_a[late].mean()),
                    late_candidate_mean_maximum_displacement_m=float(amplitude_b[late].mean()),
                    late_maximum_absolute_relative_amplitude_change=float(np.abs(amplitude_change[late]).max()),
                    late_mean_relative_amplitude_change=float(amplitude_change[late].mean()),
                    late_maximum_surface_drift_m=float(drift[late].max()),
                    late_maximum_relative_rms_surface_drift=float(rms_relative[late].max()),
                    relative_amplitude_change_by_checkpoint=amplitude_change.tolist(),
                    relative_rms_surface_drift_by_checkpoint=rms_relative.tolist(),
                    passes_static_screen=passes))
        timing_windows = {}
        for label, archive, run in (('baseline', old, a), ('candidate', new, b)):
            times = archive['loop_ms']
            assert len(times) == run['steps'] and np.isfinite(times).all()
            timing_windows[label] = [float(np.median(part)) for part in np.array_split(times, 5)]
    before, after = a['timing_ms']['loop']['median'], b['timing_ms']['loop']['median']
    reduction = 100*(1-after/before)
    duration_reduction = 100*(1-(after/before)*(a['dt']/b['dt']))
    result = dict(reference=baseline['label'], candidate=candidate['label'],
        baseline_median_ms=before, candidate_median_ms=after, time_reduction_percent=reduction,
        physics_throughput_time_reduction_percent=duration_reduction,
        baseline_timestep_s=a['dt'], candidate_timestep_s=b['dt'],
        timing_five_window_medians_ms=timing_windows, checkpoint_steps=steps.tolist(),
        late_window_measured_step_min=.6*a['steps'], common_surface_points=int(free.sum()),
        relative_amplitude_limit=limit, relative_rms_surface_drift_limit=limit,
        maximum_surface_drift_limit_m=node_cap, fingers=fingers,
        passes_static_screen=all(f['passes_static_screen'] for f in fingers),
        passes_performance_screen=bool(duration_reduction >= 15), performance_screen_minimum_reduction_percent=15,
        qualifies_for_full_replay=False,
        scope='One short A/B pair; late checkpoint window is not proof of static equilibrium. '
              'Common STL vertices are not an area-weighted full-surface error integral. '
              'Contact, lift, RGB stability and physical calibration remain untested.')
    report['comparison'] = result
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--allow-timestep-change', action='store_true')
    args = parser.parse_args()
    compare(args.report, allow_timestep_change=args.allow_timestep_change)
