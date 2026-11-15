"""Short no-render ablations in the complete native scene, without upstream edits.

Use an export-mode task, then measure a held-pose loop before geometry export.
This screens costs; it does not replace the moving/contact full-episode test.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-min-iterations", type=int, default=80)
    parser.add_argument("--single-update", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--profile", action="store_true",
                        help="Bracket only the warmed timed loop with CUDA profiler capture and NVTX")
    parser.add_argument("--sample-every", type=int, default=0,
                        help="Save nodal checkpoints outside timing every N measured steps (0 disables)")
    args, remaining = parser.parse_known_args()
    if args.steps < 20 or args.warmup_steps < 0 or not 1 <= args.scene_min_iterations <= 255 or args.sample_every < 0:
        parser.error("Use at least 20 steps, scene iterations in [1, 255], and nonnegative sample interval")
    from soft_sim.sim import robodojo
    from soft_sim.sim.scene import SceneExporter
    configure = robodojo.task_configuration

    def configuration(*a, **kw):
        cfg = configure(*a, **kw)
        cfg.sim.physx.min_position_iteration_count = args.scene_min_iterations
        return cfg

    def probe(exporter, destination, target_label):
        import torch
        from pxr import PhysxSchema
        from soft_sim.mesh import tet_volumes
        env = exporter.env
        robots = list(env.robot_manager.robot_key)
        commands = [robot.data.joint_pos.clone() for robot in robots]
        velocities = [torch.zeros_like(command) for command in commands]
        dt = exporter.cfg.simulation.timestep_s
        # Compare scene cache invalidation against actual poses at one epoch.
        before = [robot.data.body_link_pose_w.clone() for robot in robots]
        env.sim.scene.update(dt)
        after = [robot.data.body_link_pose_w.clone() for robot in robots]
        cache_pose_error = max(float((a-b).abs().max().cpu()) for a, b in zip(before, after))
        samples, step_samples, update_samples, command_samples = [], [], [], []
        checkpoints, checkpoint_steps = {}, []
        # Time the existing env.sim_step chain without changing its callbacks,
        # episode counters, scene writes or PhysX simulate/fetch operations.
        for index in range(args.steps + args.warmup_steps):
            if index == args.warmup_steps:
                torch.cuda.synchronize()
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("soft_sim.physics_probe")
            if args.profile and index >= args.warmup_steps:
                torch.cuda.nvtx.range_push("physics_step")
            started = time.perf_counter()
            for robot, position, velocity in zip(robots, commands, velocities):
                robot.set_joint_position_target(position)
                robot.set_joint_velocity_target(velocity)
            commanded = time.perf_counter()
            if not args.single_update:
                env.sim.scene.update(dt)
            updated = time.perf_counter()
            env.sim_step(render=False)
            finished = time.perf_counter()
            if index >= args.warmup_steps:
                samples.append(finished-started)
                command_samples.append(commanded-started)
                update_samples.append(updated-commanded)
                step_samples.append(finished-updated)
            if args.profile and index >= args.warmup_steps:
                torch.cuda.nvtx.range_pop()
            measured_step = index - args.warmup_steps + 1
            if args.sample_every and measured_step > 0 and measured_step % args.sample_every == 0:
                checkpoint_steps.append(measured_step)
                for arm, tool in enumerate(env.tools):
                    for side, points in enumerate(tool.positions()):
                        checkpoints.setdefault(f"history_{arm}_{side}", []).append(points.copy())
        torch.cuda.synchronize()
        if args.profile:
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
        arrays, validation = {}, []
        arrays["checkpoint_steps"] = np.asarray(checkpoint_steps, dtype=np.int32)
        arrays.update({key: np.asarray(value) for key, value in checkpoints.items()})
        for arm, tool in enumerate(env.tools):
            arrays[f"mounts_{arm}"] = tool.mounts.copy()
            collision_nodes = tool.collision_positions()
            for side, points in enumerate(tool.positions()):
                rest, tets, mount = tool.local_nodes[side], tool.sim_tets[side], tool.mounts[side]
                local = (points-mount[:3, 3]) @ mount[:3, :3]
                jacobian = tet_volumes(local, tets) / tool.rest_volumes[side]
                pin_error = float(np.linalg.norm(local-rest, axis=1)[tool.pin_masks[side]].max())
                arrays[f"nodes_{arm}_{side}"] = points
                arrays[f"rest_{arm}_{side}"] = rest
                arrays[f"tets_{arm}_{side}"] = tets
                arrays[f"pin_{arm}_{side}"] = tool.pin_masks[side]
                arrays[f"collision_nodes_{arm}_{side}"] = collision_nodes[side]
                view = tool.views[side]
                validation.append(dict(arm=arm, side=side, minimum_jacobian=float(jacobian.min()),
                                       maximum_pin_error_m=pin_error,
                                       maximum_displacement_m=float(np.linalg.norm(local-rest, axis=1).max()),
                                       actual_collision_vertices=int(view.max_vertices_per_body),
                                       actual_collision_tets=int(view.max_elements_per_body),
                                       maximum_collision_to_sim_vertex_distance_m=(
                                           float(np.linalg.norm(collision_nodes[side]-points, axis=1).max())
                                           if collision_nodes[side].shape == points.shape else None),
                                       authored_iterations=PhysxSchema.PhysxDeformableAPI(
                                           env.stage.GetPrimAtPath(tool.paths[side])).GetSolverPositionIterationCountAttr().Get()))
        def statistics(values):
            values = np.asarray(values)*1000
            return dict(mean=float(values.mean()), median=float(np.median(values)),
                        p90=float(np.percentile(values, 90)))
        import os
        report = dict(scope="Full native scene at held joint targets; timed loop has no rendering",
                      steps=args.steps, warmup_steps=args.warmup_steps, dt=dt,
                      fem_iterations=exporter.cfg.simulation.solver_iterations,
                      scene_min_iterations=args.scene_min_iterations,
                      single_update=args.single_update,
                      profiled=args.profile,
                      checkpoint_steps=checkpoint_steps,
                      mesh=exporter.cfg.mesh.tetra_mesh,
                      mesh_topology=[tool.mesh_topology for tool in env.tools],
                      follow_backend=os.environ.get("SOFT_SIM_FOLLOW_BACKEND", "cpu"),
                      cache_pose_error_m=cache_pose_error,
                      timing_ms=dict(loop=statistics(samples), env_step=statistics(step_samples),
                                     scene_update=statistics(update_samples), commands=statistics(command_samples)),
                      fingers=validation)
        arrays["loop_ms"] = np.asarray(samples) * 1000
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
        print("PHYSICS_PROBE " + json.dumps(report), flush=True)
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {"obstacle_count": 0}, None, None

    robodojo.task_configuration, SceneExporter.export = configuration, probe
    from soft_sim.cli.collect import main as collect
    collect(remaining)


if __name__ == "__main__":
    main()
