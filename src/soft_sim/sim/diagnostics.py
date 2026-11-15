"""Isolate the mounted FEM pair from robot/scene contacts; never records training data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from soft_sim.config import ExperimentConfig
from soft_sim.mesh import tet_volumes
from soft_sim.sim.launcher import create_simulation_app
from soft_sim.sim.soft_tool import MountedSoftTool
from soft_sim.robots.fingers import mounts_from_q, scale_config
from soft_sim.robots.mechanism import mechanism

#: NVML field -> report key; empty when no driver is visible.
GPU_SAMPLE_FIELDS = (("utilization.gpu", "utilisation_percent"),
                     ("memory.used", "memory_used_mib"),
                     ("power.draw", "power_w"),
                     ("clocks.sm", "clock_mhz"),
                     ("temperature.gpu", "temperature_c"))


def sample_gpu():
    try:
        lines = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(field for field, _ in GPU_SAMPLE_FIELDS),
                                "--format=csv,noheader,nounits"], check=True,
                               capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        values = lines[0].split(",")
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        return {}
    sample = {}
    for (_, key), value in zip(GPU_SAMPLE_FIELDS, values):
        try:
            sample[key] = float(value)
        except ValueError:
            continue  # laptop drivers report [N/A] for counters they do not expose
    return sample


def second_statistics(samples):
    """Seconds per call as milliseconds, plus warm-up and steady-state means."""
    if not samples:
        return None
    values = np.asarray(samples) * 1000
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), minimum=float(values.min()),
                maximum=float(values.max()), first_ten_ms=float(values[:10].mean()),
                last_ten_ms=float(values[-10:].mean()))


def gpu_statistics(samples):
    if not samples:
        return None
    names = set().union(*(entry.keys() for entry in samples))
    return {name: dict(mean=float(np.mean([entry[name] for entry in samples if name in entry])),
                       maximum=float(np.max([entry[name] for entry in samples if name in entry])),
                       minimum=float(np.min([entry[name] for entry in samples if name in entry])),
                       samples=len([entry for entry in samples if name in entry]))
            for name in sorted(names)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=.75)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--duration", type=float, default=.3)
    parser.add_argument("--rest-authoring-scale", type=float, default=1.)
    parser.add_argument("--mesh", type=Path, help="Conforming tetrahedral mesh NPZ")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--robot", default="x5", choices=("x5", "franka"),
                        help="Finger mechanism whose wrist frame holds the isolated pair")
    args = parser.parse_args(argv)
    if args.mesh is None:
        from soft_sim.storage import Storage
        args.mesh = Storage.create().cache_dir("meshes") / "soft_finger_tetgen_4mm.npz"
    if not np.isfinite(args.gravity) or args.gravity > 0:
        parser.error("Gravity must be finite and nonpositive along world Z")
    if not np.isfinite(args.dt) or args.dt <= 0 or not np.isfinite(args.duration) or args.duration < 20*args.dt:
        parser.error("Use a positive timestep and at least twenty steps")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    cfg.mesh.tetra_mesh = str(args.mesh)
    cfg.simulation.render = False
    cfg.simulation.timestep_s = args.dt
    cfg.simulation.solver_iterations = args.iterations
    cfg.simulation.deformable_vertex_velocity_damping = 20.
    cfg.simulation.contact_offset_m = .00025
    scale_config(cfg, args.scale, out)
    app = create_simulation_app(cfg.simulation)
    try:
        from isaacsim.core.api import World
        from pxr import Gf, UsdPhysics
        from scipy.spatial.transform import Rotation

        world = World(stage_units_in_meters=1., physics_dt=args.dt,
                      rendering_dt=args.dt, backend="torch", device="cuda:0")
        physics = world.get_physics_context()
        physics.enable_gpu_dynamics(True)
        physics.set_broadphase_type("GPU")
        scene = UsdPhysics.Scene.Get(world.stage, physics.prim_path)
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        scene.CreateGravityMagnitudeAttr(abs(args.gravity))
        arm = mechanism(args.robot)
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        base[:3, 3] = [.3, -.45, .765]
        mounts = mounts_from_q(arm.closed_joint_q(), base, mechanism=arm)
        tool = MountedSoftTool(world.stage, cfg, mounts, explicit_fem_mesh=True, surface_max_edge_m=0.,
                               rest_authoring_scale=args.rest_authoring_scale)
        world.reset()
        tool.initialize()
        arrays, initialization = {}, []
        for side, view in enumerate(tool.views):
            gradients = view.get_sim_element_deformation_gradients().cpu().numpy().reshape(-1, 3, 3)
            stretches = np.linalg.svd(gradients, compute_uv=False)
            initialization.append(dict(side=side, minimum_gpu_jacobian=float(np.linalg.det(gradients).min()),
                maximum_gpu_rest_stretch_error=float(np.abs(stretches-1).max())))
            arrays[f"inverse_rest_{side}"] = view.get_sim_element_rest_poses().cpu().numpy().reshape(-1, 3, 3)
            arrays[f"reference_{side}"] = tool.local_nodes[side]
            arrays[f"tets_{side}"] = tool.sim_tets[side]
        frames, follow_times, step_times, gpu_samples = [], [], [], []
        for index in range(round(args.duration/args.dt)):
            follow_started = time.perf_counter()
            tool.follow(mounts)
            follow_times.append(time.perf_counter() - follow_started)
            step_started = time.perf_counter()
            world.step(render=False)
            step_times.append(time.perf_counter() - step_started)
            # Sampling outside the timed region keeps the physics numbers clean.
            if (index+1) % 25 == 0:
                sample = sample_gpu()
                if sample:
                    gpu_samples.append(sample)
            if (index+1) % 20 == 0:
                sides = []
                for side, nodes in enumerate(tool.positions()):
                    local = (nodes-mounts[side, :3, 3])@mounts[side, :3, :3]
                    rest, tets = tool.local_nodes[side], tool.sim_tets[side]
                    sides.append(dict(maximum_displacement_mm=float(np.linalg.norm(local-rest, axis=1).max()*1000),
                        minimum_jacobian=float((tet_volumes(local,tets)/tet_volumes(rest,tets)).min())))
                    # Every sample is kept: a single instant compares vibration phase,
                    # not solver accuracy, so the sweep averages the settled part.
                    arrays.setdefault(f"sampled_nodes_{side}", []).append(nodes.copy())
                    arrays[f"final_nodes_{side}"] = nodes
                frames.append(dict(time_s=(index+1)*args.dt, sides=sides))
        for key, value in list(arrays.items()):
            if isinstance(value, list):
                arrays[key] = np.stack(value)
        np.savez_compressed(out/"state.npz", mounts=mounts, **arrays)
        report = dict(scale=args.scale, gravity_m_s2=args.gravity, dt=args.dt, iterations=args.iterations,
                      rest_authoring_scale=args.rest_authoring_scale,
                      initialization=initialization, frames=frames,
                      timing=dict(steps=len(step_times), simulated_s=len(step_times)*args.dt,
                                  world_step_ms=second_statistics(step_times),
                                  follow_ms=second_statistics(follow_times),
                                  gpu=gpu_statistics(gpu_samples)),
                      scope="Isolated two-finger initialization diagnostic; no robot, table, objects or rendering.")
        (out/"report.json").write_text(json.dumps(report, indent=2))
        print("FEM_INITIALIZATION_PROBE", json.dumps(report), flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
