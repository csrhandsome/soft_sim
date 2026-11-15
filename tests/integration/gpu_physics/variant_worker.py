"""Apply experiment-only adapters before a probe or real capture; no upstream edits."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--replay-states', type=Path)
    args, remaining = parser.parse_known_args()
    options = json.loads(args.experiment.read_text())
    from soft_sim.sim import robodojo
    configure = robodojo.task_configuration

    def configuration(*a, **kw):
        cfg = configure(*a, **kw)
        for key, value in options.get('scene_overrides', {}).items():
            if key not in ('solver_type', 'gpu_max_num_partitions', 'solve_articulation_contact_last'):
                raise ValueError(f'Unsupported scene experiment: {key}')
            setattr(cfg.sim.physx, key, value)
        return cfg
    robodojo.task_configuration = configuration

    from soft_sim.tasks import collect as task
    original = task.MountedSoftTool

    class ExperimentalSoftTool(original):
        def __init__(self, stage, config, mounts, *a, **kw):
            # Explicit experiment assets already include their own reduction
            # and collision/surface mappings; do not apply runtime reduction.
            config.mesh.eliminate_fixed_cells = False
            if 'physics_threads' in options:
                import carb
                import omni.physx.bindings._physx as pxb
                carb.settings.get_settings().set(pxb.SETTING_NUM_THREADS, options['physics_threads'])
            if 'rest_authoring_scale' in options:
                kw['rest_authoring_scale'] = options['rest_authoring_scale']
            if 'collision_mesh' not in options:
                super().__init__(stage, config, mounts, *a, **kw)
                return
            from omni.physx.scripts import deformableUtils
            from pxr import Gf, Sdf
            add = deformableUtils.add_physx_deformable_body
            with np.load(options['collision_mesh']) as archive:
                metadata = json.loads(str(archive['metadata']))
                scale = config.mesh.scale_to_m / metadata['scale_to_m']
                points = archive['vertices'] * scale * kw.get('rest_authoring_scale', 1.)
                indices = archive['tets'].astype(np.int32).ravel().tolist()
            def separate_collision(*aa, **kk):
                kk.update(collision_rest_points=[Gf.Vec3f(*point) for point in points.tolist()],
                          collision_indices=indices)
                if options.get('explicit_embedding'):
                    with np.load(options['clamp_mapping']) as mapping:
                        kk['embedding'] = mapping['collision_embedding'].tolist()
                if options.get('batch_authoring'):
                    with Sdf.ChangeBlock():
                        return add(*aa, **kk)
                return add(*aa, **kk)
            deformableUtils.add_physx_deformable_body = separate_collision
            try:
                super().__init__(stage, config, mounts, *a, **kw)
            finally:
                deformableUtils.add_physx_deformable_body = add
            self.mesh_settings.update(collision_mesh=options['collision_mesh'],
                                      collision_nodes=len(points), collision_tets=len(indices)//4)

        def initialize(self):
            if 'clamp_mapping' not in options:
                return super().initialize()
            from soft_sim.sim import render_surface
            bind = render_surface.bind_surface
            with np.load(options['clamp_mapping']) as mapping:
                anchor = int(mapping['anchor_tet'])
            def constrained_binding(mesh, config, **kw):
                surface, faces, indices, weights = bind(mesh, config, **kw)
                from soft_sim.sim.soft_tool import base_pin_mask
                mask, _ = base_pin_mask(surface, config.pin_height_m, config.voxel_pitch_m)
                support = mesh.tets[anchor]
                cell = mesh.vertices[support]
                matrix = np.stack([cell[i]-cell[0] for i in (1, 2, 3)], axis=-1)
                tail = np.linalg.solve(matrix, (surface[mask]-cell[0]).T).T
                indices[mask] = support
                weights[mask] = np.c_[1-tail.sum(axis=1), tail]
                assert np.allclose(np.einsum('ni,nij->nj',weights,mesh.vertices[indices]),surface,atol=1e-8)
                return surface, faces, indices, weights
            render_surface.bind_surface = constrained_binding
            try:
                return super().initialize()
            finally:
                render_surface.bind_surface = bind
    task.MountedSoftTool = ExperimentalSoftTool
    print('PHYSICS_EXPERIMENT ' + json.dumps(options), flush=True)
    if args.replay_states:
        from tests.integration.gpu_follow.replay_worker import main as replay
        import sys
        sys.argv = [sys.argv[0], '--states', str(args.replay_states), *remaining]
        replay()
    else:
        from tests.integration.gpu_follow.physics_probe_worker import main as probe
        import sys
        sys.argv = [sys.argv[0], *remaining]
        probe()


if __name__ == '__main__':
    main()
