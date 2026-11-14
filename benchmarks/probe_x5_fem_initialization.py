"""Isolate X5-mounted FEM from robot/scene contacts; never records training data."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.config import ExperimentConfig
from shared.mesh import tet_volumes
from backends.isaac_sim.app_launcher import create_simulation_app
from isaac51_tool import MountedSoftTool
from x5_soft_tool import mounts_from_q, scale_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=float, default=.75)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--duration", type=float, default=.3)
    parser.add_argument("--rest-authoring-scale", type=float, default=1.)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.gravity) or args.gravity > 0:
        parser.error("Gravity must be finite and nonpositive along world Z")
    if not np.isfinite(args.dt) or args.dt <= 0 or not np.isfinite(args.duration) or args.duration < 20*args.dt:
        parser.error("Use a positive timestep and at least twenty steps")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig()
    cfg.mesh.tetra_mesh = "data/generated/soft_finger_tetgen_4mm.npz"
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
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        base[:3, 3] = [.3, -.45, .765]
        mounts = mounts_from_q(np.r_[np.zeros(6), .044, .044], base)
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
        frames = []
        for index in range(round(args.duration/args.dt)):
            tool.follow(mounts)
            world.step(render=False)
            if (index+1) % 20 == 0:
                sides = []
                for side, nodes in enumerate(tool.positions()):
                    local = (nodes-mounts[side, :3, 3])@mounts[side, :3, :3]
                    rest, tets = tool.local_nodes[side], tool.sim_tets[side]
                    sides.append(dict(maximum_displacement_mm=float(np.linalg.norm(local-rest, axis=1).max()*1000),
                        minimum_jacobian=float((tet_volumes(local,tets)/tet_volumes(rest,tets)).min())))
                    arrays[f"final_nodes_{side}"] = nodes
                frames.append(dict(time_s=(index+1)*args.dt, sides=sides))
        np.savez_compressed(out/"state.npz", mounts=mounts, **arrays)
        report = dict(scale=args.scale, gravity_m_s2=args.gravity, dt=args.dt, iterations=args.iterations,
                      rest_authoring_scale=args.rest_authoring_scale,
                      initialization=initialization, frames=frames,
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
    main()
