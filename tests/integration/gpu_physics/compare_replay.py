"""Compare complete captures in the moving finger frames, including contact."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from soft_sim.storage import Storage


def compare(path, *, allow_iteration_change=False):
    path = Path(path)
    report = json.loads(path.read_text())
    before, after = (report['runs'][key] for key in ('baseline', 'candidate'))
    iterations = {label: entry['result']['config']['simulation']['solver_iterations']
                  for label, entry in (('baseline', before), ('candidate', after))}
    if allow_iteration_change:
        assert iterations == report['intended_solver_iterations'] == dict(baseline=80, candidate=40)
    else:
        assert all(value == 80 for value in iterations.values())
    configurations = []
    for entry in (before, after):
        assert entry['returncode'] == 0 and entry['capture']['status'] == 'committed'
        assert entry['timing']['timestep_s'] == .001
        assert entry['result']['executed_plans'][0]['sha256'] == report['plan_sha256']
        assert entry['capture']['camera_order'] == ['left_rgb', 'right_rgb', 'external_rgb']
        configuration = copy.deepcopy(entry['result']['config'])
        configuration['mesh'].pop('tetra_mesh', None)
        configuration['recording'].pop('dataset_id', None)
        configuration['task'].setdefault('robot', 'x5')
        if allow_iteration_change:
            configuration['simulation'].pop('solver_iterations')
        configurations.append(configuration)
    assert configurations[0] == configurations[1], 'Unexpected configuration difference'
    for key in ('physics_steps', 'simulation_s', 'captured_frames', 'fps', 'timestep_s', 'follow_backend', 'rendering'):
        assert before['timing'][key] == after['timing'][key], key
    assert [(p['phase'], p['physics_steps']) for p in before['timing']['phases']] == [
        (p['phase'], p['physics_steps']) for p in after['timing']['phases']]
    rows_a, rows_b = before['capture']['frames'], after['capture']['frames']
    assert [(r['time_s'],r['phase']) for r in rows_a] == [(r['time_s'],r['phase']) for r in rows_b]
    phases = np.asarray([r['phase'] for r in rows_a])
    times = np.asarray([r['time_s'] for r in rows_a])
    storage = Storage.create()
    references = []
    for entry in (before, after):
        file = storage.run_dir(entry['run_id'])/'capture/finger_reference.npz'
        with np.load(file) as archive:
            references.append({key:archive[key] for key in archive.files})
    side_reports = []
    with np.load(before['states']) as old, np.load(after['states']) as new:
        assert set(old.files) == set(new.files)
        for key in old.files:
            assert old[key].shape == new[key].shape, key
            assert old[key].shape[0] == len(rows_a), key
            assert np.isfinite(old[key]).all() and np.isfinite(new[key]).all(), key
        np.testing.assert_allclose(old['time_s'],new['time_s'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(old['time_s'], times, atol=1e-12, rtol=0)
        assert np.isfinite(old['joint_positions']).all() and np.isfinite(new['joint_positions']).all()
        command_error = float(np.abs(old['commanded_joint_positions']-new['commanded_joint_positions']).max())
        command_details = dict(maximum_difference_si=command_error,
                               units='radians for revolute joints; meters for prismatic joints')
        if allow_iteration_change:
            calibration = before['capture']['calibration']
            assert calibration['joint_names'] == after['capture']['calibration']['joint_names']
            active_prefix = 'robot'+str(before['result']['config']['task']['active_arm'])+'/'
            active = np.asarray([name.startswith(active_prefix) for name in calibration['joint_names']])
            assert active.any() and (~active).any()
            difference = np.abs(old['commanded_joint_positions']-new['commanded_joint_positions'])
            active_error, parked_error = float(difference[:,active].max()),float(difference[:,~active].max())
            assert active_error <= 1e-12
            assert before['result']['config']['task'].get('robot', 'x5') == 'x5'
            parked_slides = np.asarray([not is_active and name.rsplit('/',1)[-1] in ('joint7','joint8')
                                       for is_active,name in zip(active,calibration['joint_names'])])
            assert parked_slides.sum() == 2
            parked_rotations = ~active & ~parked_slides
            assert float(difference[:,parked_rotations].max()) <= 1e-12
            # The parked arm holds its measured post-settle pose. Different
            # iteration counts can shift that initial pose by a few micrometers.
            # Its targets must remain constant; keep the active plan exact.
            for archive in (old,new):
                targets = archive['commanded_joint_positions'][:,~active]
                np.testing.assert_allclose(targets,np.broadcast_to(targets[0],targets.shape),atol=1e-12,rtol=0)
            assert parked_error <= 1e-5
            command_details.update(active_arm_maximum_difference_si=active_error,
                                   parked_arm_maximum_difference_si=parked_error,
                                   parked_finger_slide_offset_limit_m=1e-5,
                                   rule='Active plan and parked rotations <=1e-12; constant parked finger slide targets <=1e-5 m offset')
        else:
            assert command_error <= 1e-12
        for side, name in enumerate(('left','right')):
            rest, rest_new = (r['finger_reference_'+name] for r in references)
            pin, pin_new = (r['finger_pin_'+name] for r in references)
            np.testing.assert_array_equal(rest,rest_new)
            np.testing.assert_array_equal(pin,pin_new)
            responses = []
            for archive in (old,new):
                nodes = archive['finger_nodes_'+name]
                mounts = archive['finger_mounts'][:,side]
                local = np.einsum('tni,tij->tnj',nodes-mounts[:,None,:3,3],mounts[:,:3,:3])
                assert np.isfinite(local).all()
                responses.append((local-rest)[:,~pin])
            a,b = responses
            amplitude_a = np.linalg.norm(a,axis=-1).max(axis=1)
            amplitude_b = np.linalg.norm(b,axis=-1).max(axis=1)
            drift = np.linalg.norm(a-b,axis=-1)
            precontact = np.flatnonzero(phases=='approach')[-1]
            contact_a = np.linalg.norm(a-a[precontact],axis=-1).max(axis=1)
            contact_b = np.linalg.norm(b-b[precontact],axis=-1).max(axis=1)
            per_phase = {}
            for phase in dict.fromkeys(phases):
                mask = phases==phase
                settled = mask & (times >= times[mask][0]+.25)
                if not settled.any():
                    settled = mask
                first,second = float(amplitude_a[settled].mean()),float(amplitude_b[settled].mean())
                phase_indices = np.flatnonzero(mask)
                worst_index = int(phase_indices[np.argmax(np.abs(amplitude_b[mask]/amplitude_a[mask]-1))])
                per_phase[str(phase)] = dict(frames=int(mask.sum()),settled_frames=int(settled.sum()),
                    baseline_mean_maximum_displacement_mm=1000*first,
                    candidate_mean_maximum_displacement_mm=1000*second,
                    mean_amplitude_change_percent=100*(second/first-1),
                    mean_amplitude_difference_mm=1000*(second-first),
                    maximum_frame_amplitude_change_percent=float((100*np.abs(amplitude_b[mask]/amplitude_a[mask]-1)).max()),
                    maximum_frame_amplitude_difference_mm=float(np.abs(amplitude_b[mask]-amplitude_a[mask]).max()*1000),
                    worst_relative_amplitude_frame=dict(time_s=float(times[worst_index]),
                        baseline_mm=float(amplitude_a[worst_index]*1000),
                        candidate_mm=float(amplitude_b[worst_index]*1000)),
                    baseline_peak_displacement_mm=float(amplitude_a[mask].max()*1000),
                    candidate_peak_displacement_mm=float(amplitude_b[mask].max()*1000),
                    maximum_node_shape_drift_mm=float(drift[mask].max()*1000),
                    shape_rms_drift_mm=float(np.sqrt(np.mean(drift[settled]**2))*1000),
                    baseline_mean_contact_deformation_mm=float(contact_a[settled].mean()*1000),
                    candidate_mean_contact_deformation_mm=float(contact_b[settled].mean()*1000),
                    mean_contact_deformation_difference_mm=float((contact_b[settled]-contact_a[settled]).mean()*1000),
                    mean_contact_deformation_change_percent=(float(100*(contact_b[settled].mean()/contact_a[settled].mean()-1))
                        if contact_a[settled].mean() > 1e-8 else None))
            worst_frame, worst_free_node = np.unravel_index(drift.argmax(), drift.shape)
            side_reports.append(dict(side=name,free_nodes=int((~pin).sum()),phases=per_phase,
                maximum_node_shape_drift_mm=float(drift.max()*1000),
                frames_over_shape_drift_limit=int((drift.max(axis=1) > .0015).sum()),
                worst_node_frame=dict(frame=int(worst_frame),time_s=float(times[worst_frame]),
                    phase=str(phases[worst_frame]),
                    original_node_index=int(np.flatnonzero(~pin)[worst_free_node]))))
        minimum_jacobians = {label:float(values['all_finger_minimum_jacobians'].min())
                             for label,values in (('baseline',old),('candidate',new))}
        maximum_pin_errors = {label:float(values['all_finger_maximum_pin_errors_m'].max())
                              for label,values in (('baseline',old),('candidate',new))}
        object_distance = float(np.linalg.norm(old['object_position']-new['object_position'],axis=-1).max())
    physics = {label:1000*sum(p['control_and_physics_s'] for p in entry['timing']['phases'])/entry['timing']['physics_steps']
               for label,entry in (('baseline',before),('candidate',after))}
    wall = {label:entry['timing']['wall_s'] for label,entry in (('baseline',before),('candidate',after))}
    lift = {label:entry['result']['object_lift_m'] for label,entry in (('baseline',before),('candidate',after))}
    stable_amplitude = max(abs(s['phases'][phase]['mean_amplitude_change_percent'])
                           for s in side_reports for phase in ('hold','verify_hold'))
    checks = dict(task_success=before['result']['success'] is True and after['result']['success'] is True,
                  finite_positive_elements=min(minimum_jacobians.values()) > 0,
                  clamp=max(maximum_pin_errors.values()) < 1e-5,
                  mean_hold_deformation=stable_amplitude <= 5,
                  maximum_shape_drift=max(s['maximum_node_shape_drift_mm'] for s in side_reports) <= 1.5,
                  lift_difference=abs(lift['candidate']-lift['baseline']) <= .0005)
    result = dict(control_and_physics_ms_per_step=physics,
        physics_time_reduction_percent=100*(1-physics['candidate']/physics['baseline']),
        collection_wall_s=wall,collection_time_reduction_percent=100*(1-wall['candidate']/wall['baseline']),
        frames=len(rows_a),simulation_s=before['timing']['simulation_s'],
        iterations=80 if iterations['baseline'] == iterations['candidate'] else None,
        solver_iterations=iterations,timestep_s=.001,
        maximum_command_difference_si=command_error,command_comparison=command_details,
        minimum_recorded_jacobians=minimum_jacobians,maximum_recorded_pin_errors_m=maximum_pin_errors,
        object_lift_m=lift,object_lift_difference_mm=abs(lift['candidate']-lift['baseline'])*1000,
        maximum_object_position_drift_mm=object_distance*1000,
        maximum_settled_hold_amplitude_change_percent=stable_amplitude,
        phase_timings=[dict(phase=a['phase'],physics_steps=a['physics_steps'],
            baseline_control_and_physics_ms_per_step=1000*a['control_and_physics_s']/a['physics_steps'],
            candidate_control_and_physics_ms_per_step=1000*b['control_and_physics_s']/b['physics_steps'],
            physics_time_reduction_percent=100*(1-b['control_and_physics_s']/a['control_and_physics_s']))
            for a,b in zip(before['timing']['phases'],after['timing']['phases'])],
        thresholds=dict(mean_settled_hold_amplitude_change_percent=5.,maximum_node_shape_drift_mm=1.5,
                        maximum_pin_error_m=1e-5,object_lift_difference_m=.0005),
        fingers=side_reports,review_checks=checks,qualifies_for_visual_review=all(checks.values()),
        scope='One complete A/B replay on one saved task/layout. All free FEM nodes are compared in '
              'their moving mount frames. Phase means exclude the first 0.25 s when possible. Contact '
              'deformation is relative to each run last approach frame. No physical material '
              'calibration or production/training acceptance is claimed. Wall time excludes '
              'initial setup and dataset finalization.')
    report['comparison']=result
    path.write_text(json.dumps(report,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report',type=Path)
    parser.add_argument('--allow-iteration-change',action='store_true',
                        help='Explicitly compare a declared 80/40 iteration experiment')
    args = parser.parse_args()
    result = compare(args.report, allow_iteration_change=args.allow_iteration_change)
    raise SystemExit(0 if result['qualifies_for_visual_review'] else 1)
