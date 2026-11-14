"""Unloaded native-drive full-travel diagnostic; never a task-success episode."""
import json
from pathlib import Path

import numpy as np

from soft_sim.mesh import tet_volumes
from soft_sim.geometry import camera_intrinsics
from soft_sim.execution import trajectory_command


def run_assembly_motion(env, out, dt, scene):
    import torch
    from PIL import Image
    out = Path(out) / "assembly_motion"
    out.mkdir(parents=True, exist_ok=True)
    robots = env.robot_manager.robot_key
    initial = [robot.data.joint_pos[0].cpu().numpy().copy() for robot in robots]
    references = {}
    rest_volumes = []
    for index, tool in enumerate(env.tools):
        references[f"robot{index}_fixture_paths"] = np.asarray(tool.fixtures.paths)
        rest_volumes.append([tet_volumes(local, tets) for local, tets in zip(tool.local_nodes, tool.sim_tets)])
        for side in (0, 1):
            references[f"robot{index}_reference_{side}"] = tool.local_nodes[side]
            references[f"robot{index}_tets_{side}"] = tool.sim_tets[side]
            references[f"robot{index}_pin_{side}"] = tool.pin_masks[side]
    np.savez_compressed(out / "reference.npz", **references)
    rows, step = [], 0
    last_width = .088
    worst_jacobian, worst_pin = float("inf"), 0.
    for phase, duration, width in (("close", 2., 0.), ("closed_hold", .5, 0.),
                                   ("open", 2., .088), ("open_hold", .5, .088)):
        steps = round(duration / dt)
        print("ASSEMBLY_PHASE", phase, flush=True)
        for sample in range(steps):
            for index, robot in enumerate(robots):
                path = np.repeat(initial[index][None, :-2], 2, axis=0)
                q, velocity, jaw = trajectory_command(path, steps*dt, (sample+1)*dt, last_width, width)
                robot.set_joint_position_target(torch.as_tensor(q[None], device=robot.device, dtype=torch.float32))
                robot.set_joint_velocity_target(torch.as_tensor(velocity[None], device=robot.device, dtype=torch.float32))
            env.sim.scene.update(dt)
            env.sim_step(render=False)
            step += 1
            if step % round(.1/dt) and sample != steps-1:
                continue
            state = {}
            row = dict(phase=phase, time_s=step*dt, commanded_width_m=jaw, arms=[])
            for index, (robot, tool) in enumerate(zip(robots, env.tools)):
                actual = robot.data.joint_pos[0].cpu().numpy().copy()
                state[f"robot{index}_q"] = actual
                state[f"robot{index}_mounts"] = tool.mounts.copy()
                state[f"robot{index}_cameras"] = tool.camera_poses.copy()
                state[f"robot{index}_fixtures"] = tool.fixtures.poses()
                arm = dict(slides_m=actual[-2:].tolist(), sides=[], fixture_mount_error=tool.fixtures.mount_error(tool.mounts))
                for side, points in enumerate(tool.positions()):
                    mount = tool.mounts[side]
                    local = (points-mount[:3,3])@mount[:3,:3]
                    rest, tets = tool.local_nodes[side], tool.sim_tets[side]
                    jacobian = tet_volumes(local, tets)/rest_volumes[index][side]
                    pin = float(np.linalg.norm(local-rest,axis=1)[tool.pin_masks[side]].max())
                    entry = dict(minimum_jacobian=float(jacobian.min()), maximum_pin_error_m=pin,
                        maximum_displacement_m=float(np.linalg.norm(local-rest,axis=1).max()))
                    entry["valid"] = bool(np.isfinite(jacobian).all() and jacobian.min()>0 and pin<1e-5)
                    arm["sides"].append(entry)
                    worst_jacobian, worst_pin = min(worst_jacobian, entry["minimum_jacobian"]), max(worst_pin, pin)
                    state[f"robot{index}_nodes_{side}"] = points
                row["arms"].append(arm)
            filename = f"state_{len(rows):04d}.npz"
            np.savez_compressed(out / filename, **state)
            row["state_file"] = filename
            if sample == steps-1:
                colors, depth = env.tool.capture(env.sim.sim)
                np.savez_compressed(out / f"{phase}_depth.npz", depth=depth, cameras=env.tool.camera_poses,
                    intrinsics=np.stack([camera_intrinsics(env.tool.config.camera, 2.8)]*2),
                    nodes_left=env.tool.positions()[0], nodes_right=env.tool.positions()[1])
                for name, pixels in zip(("left", "right"), colors):
                    Image.fromarray(pixels).save(out / f"{phase}_{name}.png")
            rows.append(row)
            (out / "samples.json").write_text(json.dumps(rows, indent=2))
            if not all(side["valid"] for arm in row["arms"] for side in arm["sides"]):
                raise RuntimeError("Full-travel assembly FEM check failed; diagnostic samples retained")
        last_width = width
    endpoints = [row for row in rows if row["phase"].endswith("hold")]
    tracking = max(abs(value-row["commanded_width_m"]/2) for row in endpoints
                   for arm in row["arms"] for value in arm["slides_m"])
    result = dict(scope=__doc__.strip(), task_success_evaluated=False,
        joint_control="original position/velocity targets", samples=len(rows),
        minimum_jacobian=worst_jacobian, maximum_pin_error_m=worst_pin,
        maximum_hold_slide_tracking_error_m=tracking, finger_installation=scene["finger_installation"],
        maximum_fixture_translation_error_m=max(arm["fixture_mount_error"]["maximum_translation_error_m"] for row in rows for arm in row["arms"]),
        maximum_fixture_rotation_matrix_error=max(arm["fixture_mount_error"]["maximum_rotation_matrix_error"] for row in rows for arm in row["arms"]),
        fixture_error_epoch="pre-step native pose versus same-epoch FEM mount target",
        maximum_post_step_fem_target_offset_m=max(arm["fixture_mount_error"].get("post_step_fem_target_offset_m", 0.) for row in rows for arm in row["arms"]),
        valid=bool(worst_jacobian>0 and worst_pin<1e-5 and tracking<.0001),
        collision_validation="Run check_x5_assembly_motion.py on the saved deformed samples separately.")
    result["valid"] = bool(result["valid"] and result["maximum_fixture_translation_error_m"]<1e-5
                           and result["maximum_fixture_rotation_matrix_error"]<1e-5)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print("ASSEMBLY_RESULT", json.dumps(result), flush=True)
    if not result["valid"]:
        raise RuntimeError("Native slide endpoints did not meet the assembly diagnostic")
