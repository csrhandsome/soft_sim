"""Run the original RoboDojo task managers with replacement FEM fingers.

This is an embodiment variant, not a stock dual-X5 benchmark score. The scene
is exported, planned in the planning environment, and executed in the very same
task instance. Success is only ever the upstream RewardManager verdict.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from soft_sim.config import ROOT, ExperimentConfig
from soft_sim.execution import phase_commands, pickup_phases
from soft_sim.geometry import camera_intrinsics
from soft_sim.mesh import tet_volumes
from soft_sim.recording.capture import EpisodeRecorder
from soft_sim.robots.fingers import (
    MountedFixtures, finger_in_link, link_transforms, measured_installation, mounts_from_q, scale_config,
)
from soft_sim.robots.mechanism import mechanism
from soft_sim.runtime import run as run_process
from soft_sim.sim import robodojo
from soft_sim.sim.cameras import task_camera_capture
from soft_sim.sim.launcher import create_simulation_app
from soft_sim.sim.scene import SceneExporter, pose_matrix
from soft_sim.sim.soft_tool import MountedSoftTool, capture_mounted_tools
from soft_sim.storage import RunStore, Storage


# -- configuration -------------------------------------------------------
def build_config(args):
    """Typed defaults from the config file, then explicit CLI overrides only."""
    cfg = ExperimentConfig.load(args.config, material_preset=getattr(args, "material", None))
    # The replacement fingers ride the original slides; the stock gripper model
    # is unused, so no legacy robot profile is reused.
    cfg.gripper.base_height_m = 0.
    cfgs = {
        "task.name": args.task, "task.seed": args.seed, "task.robot": args.robot,
        "task.active_arm": args.active_arm,
        "task.finger_scale": args.finger_scale, "task.target_label": args.target_label,
        "task.joint_control": args.joint_control,
        "task.adapter_offset_m": args.adapter_offset,
        "task.texture_max_resolution": args.texture_max_resolution,
        "mesh.eliminate_fixed_cells": args.fixed_cell_elimination,
        "simulation.timestep_s": args.dt, "simulation.solver_iterations": args.solver_iterations,
        "simulation.deformable_vertex_velocity_damping": args.vertex_velocity_damping,
        "planning.friction": args.planning_friction, "planning.max_attempts": args.planning_attempts,
        "planning.indent_m": args.indent,
        "recording.fps": args.fps, "recording.dataset_id": args.dataset_id,
        "recording.success_only": getattr(args, "success_only", None),
    }
    cfg.apply_overrides(cfgs)
    active = mechanism(cfg.task.robot)
    cfg.gripper.profile = f"native_{active.name}"
    cfg.gripper.open_jaw_width_m = active.max_open_width_m
    cfg.simulation.contact_offset_m = .00025
    cfg.camera.width = cfg.camera.height = 256
    cfg.object.fixed = False
    if args.contact_height is not None:
        cfg.planning.contact_height_candidates_m = (args.contact_height,)
    fem_mesh = Path(args.fem_mesh) if args.fem_mesh else (
        Storage.create(args.storage_root).cache_dir("meshes") / "soft_finger_tetgen_4mm.npz")
    if not fem_mesh.is_file():
        legacy = ROOT / "data/generated/soft_finger_tetgen_4mm.npz"
        if legacy.is_file():
            print(f"WARNING legacy mesh location in use: {legacy}", file=sys.stderr, flush=True)
            fem_mesh = legacy
    if not fem_mesh.is_file():
        raise SystemExit("Generate the conforming mesh first: "
                         "python scripts/run.py prepare-mesh --method tetgen --pitch .004")
    cfg.mesh.tetra_mesh = str(fem_mesh)
    cfg.task.target_label = cfg.task.target_label or (
        "block_1" if cfg.task.name == "stack_blocks" else "target")
    if cfg.task.joint_control == "position" and cfg.task.name not in cfg.task.supported_tasks:
        raise SystemExit(f"Unsupported task: {cfg.task.name}")
    return cfg.validate()


def validate_arguments(args):
    """Checks that need no configuration file; fail before starting Isaac Sim."""
    if args.static_camera_probe_frames not in (0,) and args.static_camera_probe_frames < 2:
        raise SystemExit("Static camera probe requires at least two frames, or zero to disable")
    if args.mode == "replay" and args.grasp_plan is None:
        raise SystemExit("Replay requires a collision-checked plan exported from this task layout")
    if args.mode == "assembly" and args.joint_control not in (None, "position"):
        raise SystemExit("Assembly sweep requires the original position drives")
    if args.mode == "collect" and args.grasp_plan is not None:
        raise SystemExit("Collect plans from the current scene; use replay for a saved plan")


def dataset_root_for(cfg, storage):
    """One collection per task and mechanism; seeds are episodes inside it."""
    dataset_id = cfg.recording.dataset_id or f"{cfg.task.robot}_{cfg.task.name}_v1"
    return storage.dataset_dir(dataset_id), dataset_id


# -- main ----------------------------------------------------------------
def run(args):
    validate_arguments(args)
    cfg = build_config(args)
    task, seed, mode = cfg.task.name, cfg.task.seed, args.mode
    if mode == "replay" and task != "general_pickup":
        raise SystemExit("Use --mode collect for stack_blocks; replay accepts a general_pickup plan")
    arm = mechanism(cfg.task.robot)
    mechanisms = [mechanism("x5") for _ in range(2)]
    mechanisms[cfg.task.active_arm] = arm
    installations = [measured_installation(cfg.task.finger_scale, mechanism=entry) for entry in mechanisms]
    installation = installations[cfg.task.active_arm]
    layout_path = (Path(args.layout) if args.layout else robodojo.layout_path(task, seed)).resolve()
    layout = json.loads(layout_path.read_text())
    layout_hash = hashlib.sha256(layout_path.read_bytes()).hexdigest()
    plan = json.loads(Path(args.grasp_plan).read_text()) if args.grasp_plan else None
    if plan and (plan.get("status") != "planned" or not plan.get("collision_checked")
                 or plan["scene"].get("layout_sha256") != layout_hash
                 or plan["scene"].get("task") != task):
        raise SystemExit("Plan must match this official task and unmodified saved layout")
    if plan and plan["scene"].get("finger_installation") != installation:
        raise SystemExit("Plan installation differs from the measured mount; re-plan it in the current scene")

    storage = Storage.create(args.storage_root)
    store = RunStore.create(storage, command=f"collect:{mode}", task=task, seed=seed,
                            run_id=args.run_id)
    store.write_config_snapshot(cfg, vars(args))
    dataset_root, dataset_id = dataset_root_for(cfg, storage)
    run_dir = store.root
    plans_dir = store.ensure("plans")
    diagnostics = store.ensure("diagnostics")
    previews = store.ensure("previews", "initial")
    capture_dir = run_dir / "capture"
    print(f"RUN {store.run_id} {run_dir}", flush=True)

    if cfg.task.finger_scale != 1:
        scale_config(cfg, cfg.task.finger_scale, store.ensure("mesh"))
    app = create_simulation_app(cfg.simulation)
    recorder = None
    completed = False
    store.update(execution="running")
    try:
        import carb
        carb.settings.get_settings().set_int(
            "/rtx-transient/resourcemanager/maxMipCount",
            int(np.log2(cfg.task.texture_max_resolution)) + 1)
        from pxr import Usd, UsdGeom, UsdPhysics
        from scipy.spatial.transform import Rotation
        import torch

        task_name, Task = robodojo.load_task_class(task)
        robot_cfg = robodojo.robot_config([entry.name for entry in mechanisms])
        # Original task scene and layout; only the embodiment and sensors change.
        config = robodojo.task_configuration(
            task_name, seed=seed, dt=cfg.simulation.timestep_s,
            solver_iterations=cfg.simulation.solver_iterations,
            texture_max_resolution=cfg.task.texture_max_resolution, robot=robot_cfg)
        missing = arm.missing_assets()
        if missing:
            raise SystemExit(f"RoboDojo snapshot is missing {arm.name} assets: {', '.join(missing)}")
        arm_index = cfg.task.active_arm
        bases = [robodojo.arm_root_transform(robot_cfg, index) for index in range(len(mechanisms))]
        base = bases[arm_index]
        # The ROM is reset to the upstream init pose; the FEM must start there,
        # not at a nominal pose the robot never holds.
        initial_q = []
        for entry in mechanisms:
            upstream = robodojo.robot_init_joint_pos(entry.name)
            missing = set(entry.joint_names) - set(upstream)
            if missing:
                raise RuntimeError(f"{entry.name} upstream init pose lacks {sorted(missing)}")
            initial_q.append(np.array([upstream[name] for name in entry.joint_names]))
        mount_in_links = [np.stack([finger_in_link(side, inst, entry) for side in (0, 1)])
                          for entry, inst in zip(mechanisms, installations)]
        follow_backend = os.environ.get("SOFT_SIM_FOLLOW_BACKEND", "cpu")
        if follow_backend not in ("cpu", "gpu"):
            raise ValueError("SOFT_SIM_FOLLOW_BACKEND must be cpu or gpu")

        class SoftTask(Task):
            tool = None
            tool_initialized = False
            tools = None
            gpu_follower = None

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                # Keep the original table camera and replace factory wrist
                # cameras with the two soft-finger interior cameras.
                self.camera_manager.init_config = robodojo.head_camera_config()

            def _post_setup_scene(self, sim):
                super()._post_setup_scene(sim)
                self.tools = []
                self.removed_finger_colliders = {}
                for index in range(len(robot_cfg["robots"])):
                    entry, inst = mechanisms[index], installations[index]
                    for name in entry.finger_links:
                        path = robodojo.robot_link_path(self.stage, index, name)
                        prim = self.stage.GetPrimAtPath(path)
                        prim.SetInstanceable(False)
                        # The USD has nested instanced collision subtrees.
                        # Material visibility alone does not remove those shapes.
                        while True:
                            instances = [child for child in Usd.PrimRange(prim) if child.IsInstance()]
                            if not instances:
                                break
                            for child in instances:
                                child.SetInstanceable(False)
                        UsdGeom.Imageable(prim).MakeVisible()
                        for child in prim.GetChildren():
                            if child.IsA(UsdGeom.Imageable):
                                UsdGeom.Imageable(child).MakeInvisible()
                        disabled = 0
                        for child in Usd.PrimRange(prim):
                            if child.HasAPI(UsdPhysics.CollisionAPI):
                                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
                                disabled += 1
                        self.removed_finger_colliders[path] = disabled
                    mounts = mounts_from_q(initial_q[index], bases[index], inst, entry)
                    filter_paths = {side: [robodojo.robot_link_path(self.stage, index, name)]
                                    for side, name in enumerate(entry.finger_links)}
                    tool_config = deepcopy(cfg)
                    tool_config.simulation.render = index == arm_index
                    tool = MountedSoftTool(self.stage, tool_config, mounts,
                        root=f"/World/SoftTool{index}", filtered_bodies=filter_paths,
                        explicit_fem_mesh=True, max_depenetration_velocity=1.,
                        surface_max_edge_m=.002 * cfg.task.finger_scale,
                        rt_subframes=4, lighting_samples_per_pixel=8)
                    self.tools.append(tool)
                    tool.fixtures = MountedFixtures(self.stage,
                        [robodojo.robot_link_path(self.stage, index, name) for name in entry.finger_links],
                        cfg.task.finger_scale, inst, tool.paths, entry)
                    tool.fixtures.follow(mounts)
                self.tool = self.tools[arm_index]

            def sim_step(self, render=True):
                if self.tool is not None:
                    if not self.tool_initialized:
                        for tool in self.tools:
                            tool.initialize()
                        self.tool_initialized = True
                    self.follow_tools(update_cameras=render)
                return super().sim_step(render=render)

            def follow_tools(self, *, update_cameras=True):
                if follow_backend == "gpu" and self.tool_initialized:
                    if self.gpu_follower is None:
                        from soft_sim.sim.gpu_follow import GpuToolFollower
                        self.gpu_follower = GpuToolFollower(self.tools,
                            list(self.robot_manager.robot_key),
                            [entry.finger_links for entry in mechanisms], mount_in_links)
                    self.gpu_follower.follow(update_cameras=update_cameras)
                    return
                for index, tool in enumerate(self.tools):
                    articulation = self.robot_manager.robot_key[index]
                    poses = articulation.data.body_link_pose_w[0].cpu().numpy()
                    links = []
                    for name in mechanisms[index].finger_links:
                        pose = poses[articulation.body_names.index(name)]
                        matrix = np.eye(4)
                        matrix[:3, 3] = pose[:3]
                        matrix[:3, :3] = Rotation.from_quat(np.r_[pose[4:7], pose[3]]).as_matrix()
                        links.append(matrix)
                    tool.follow(np.asarray(links) @ mount_in_links[index], update_cameras=update_cameras)

        env = SoftTask(config, app)
        env.success = [True]
        env.scene_manager.layout_manager.set_saved_layout(0, layout)
        print("TASK_STAGE before_reset", time.time(), flush=True)
        env.reset(seed=[seed])
        print("TASK_STAGE after_reset", time.time(), flush=True)
        env.scene_manager.apply_saved_poses([0])
        print("TASK_STAGE after_apply_saved_poses", time.time(), flush=True)
        # Fail loudly instead of planning against links the loaded USD lacks.
        for index, entry in enumerate(mechanisms):
            bodies = set(env.robot_manager.robot_key[index].body_names)
            expected_links = {entry.hand_link, *entry.finger_links}
            if not expected_links <= bodies:
                raise RuntimeError(f"robot{index} USD is missing {sorted(expected_links - bodies)} "
                                   f"for the {entry.name} finger mechanism")
        if os.environ.get("SOFT_SIM_DEBUG_JOINTS"):
            for index, articulation in enumerate(env.robot_manager.robot_key):
                print("JOINTS", index, "after_reset",
                      np.round(articulation.data.joint_pos[0].cpu().numpy(), 4).tolist(), flush=True)
        env.follow_tools()
        for tool in env.tools:
            tool.reset_to_mounts(tool.mounts)
        print("TASK_STAGE fem_reset_at_measured_mounts", time.time(), flush=True)
        for _ in range(round(.3 / cfg.simulation.timestep_s)):
            env.sim_step(render=False)
        print("TASK_STAGE after_300_steps", time.time(), flush=True)
        if os.environ.get("SOFT_SIM_DEBUG_JOINTS"):
            for index, articulation in enumerate(env.robot_manager.robot_key):
                print("JOINTS", index, "after_300",
                      np.round(articulation.data.joint_pos[0].cpu().numpy(), 4).tolist(), flush=True)
        env.robot_manager.set_origin_endpose()
        robot = env.robot_manager.robot_key[arm_index]
        # The articulation root settles under gravity and joint compliance, so
        # the nominal yml pose is only a pre-reset placeholder. Everything the
        # planner sees must use the measured base frame.
        nominal_base = base
        root_pose = np.asarray(robot.data.root_link_pose_w[0].detach().cpu().numpy(), dtype=float)
        base = pose_matrix((root_pose[:3], root_pose[3:]))
        root_body = robot.body_names[0]
        if root_body != arm.base_link:
            raise RuntimeError(f"Articulation root {root_body!r} is not the {arm.name} base link "
                               f"{arm.base_link!r}")
        external, product, external_calibration = task_camera_capture(env)

        fem_state, fem_report = {}, []
        for index, tool in enumerate(env.tools):
            sides = []
            for side, nodes in enumerate(tool.positions()):
                mount = tool.mounts[side]
                local = (nodes - mount[:3, 3]) @ mount[:3, :3]
                rest, tets = tool.local_nodes[side], tool.sim_tets[side]
                displacement = np.linalg.norm(local - rest, axis=1)
                jacobian = tet_volumes(local, tets) / tet_volumes(rest, tets)
                pin_error = float(displacement[tool.pin_masks[side]].max())
                entry = dict(minimum_jacobian=float(jacobian.min()),
                             maximum_displacement_m=float(displacement.max()),
                             maximum_pin_error_m=pin_error)
                entry["valid"] = bool(np.isfinite(jacobian).all() and jacobian.min() > 0
                                      and pin_error < 1e-5)
                sides.append(entry)
                fem_state[f"robot{index}_nodes_{side}"] = nodes
                fem_state[f"robot{index}_reference_{side}"] = rest
                fem_state[f"robot{index}_tets_{side}"] = tets
            fem_state[f"robot{index}_mounts"] = tool.mounts
            fem_report.append(sides)
        np.savez_compressed(diagnostics / "initial_fem.npz", **fem_state)
        store.write_json("diagnostics/initial_fem_validation.json", fem_report)
        fem_valid = all(side["valid"] for arm in fem_report for side in arm)
        if not fem_valid and mode != "export":
            raise RuntimeError("Initial native-task FEM contains inverted cells or clamp slip")
        for tool in env.tools:
            tool.sync_render_surface()
        rgb, depth = env.tool.capture(env.sim.sim)
        np.savez_compressed(previews / "preview_depth.npz", depth=depth,
            cameras=env.tool.camera_poses,
            intrinsics=np.stack([camera_intrinsics(cfg.camera, 2.8)] * 2),
            nodes_left=env.tool.positions()[0], nodes_right=env.tool.positions()[1])
        from PIL import Image
        for name, pixels in zip(("left", "right"), rgb):
            Image.fromarray(pixels).save(previews / f"preview_{name}.png")
        Image.fromarray(np.asarray(external.get_data())[..., :3].astype(np.uint8)).save(
            previews / "preview_external.png")
        store.record("previews.initial", previews)

        expected = mounts_from_q(robot.data.joint_pos[0].cpu().numpy(), base, installation, arm,
                                 joint_names=robot.joint_names)
        actual = env.tool.mounts
        position_error = float(np.linalg.norm(expected[:, :3, 3] - actual[:, :3, 3], axis=1).max())
        rotation_error = float(np.abs(expected[:, :3, :3] - actual[:, :3, :3]).max())
        report = dict(valid=position_error < .0001 and rotation_error < .001,
                      max_translation_error_m=position_error,
                      max_rotation_matrix_error=rotation_error,
                      joint_names=robot.joint_names, measured_mounts=actual.tolist(),
                      fk_mounts=expected.tolist(), installation=installation,
                      robot=arm.name, base_pose=base.tolist(),
                      nominal_base_pose=nominal_base.tolist(),
                      root_settle_translation_m=float(np.linalg.norm(
                          base[:3, 3] - nominal_base[:3, 3])))
        if hasattr(env.tool, "fixtures"):
            report["fixture_mount_errors"] = [tool.fixtures.mount_error(tool.mounts) for tool in env.tools]
            report["valid"] = report["valid"] and all(
                error["maximum_translation_error_m"] < 1e-5
                and error["maximum_rotation_matrix_error"] < 1e-5
                for error in report["fixture_mount_errors"])
        store.write_json("reports/initial/mount_validation.json", report)
        if not report["valid"]:
            # Per-link evidence: the mount pair alone cannot say which joint
            # or frame diverged between the URDF snapshot and the articulation.
            q = robot.data.joint_pos[0].cpu().numpy()
            links = link_transforms(q, mechanism=arm, joint_names=robot.joint_names)
            bodies = {name: index for index, name in enumerate(robot.body_names)}
            poses = robot.data.body_link_pose_w[0].cpu().numpy()
            evidence = {name: dict(fk=(base @ links[name]).tolist(),
                                   isaac=pose_matrix(robodojo.numpy_pose(
                                       (poses[index][:3], poses[index][3:]))).tolist())
                        for name, index in bodies.items() if name in links}
            store.write_json("diagnostics/mount_disagreement.json", dict(
                robot=arm.name, joint_names=list(robot.joint_names),
                joint_positions=np.asarray(q).tolist(), base=base.tolist(), links=evidence))
            raise RuntimeError(f"{arm.name} articulation and soft-finger planning mounts disagree")

        manager = env.scene_manager.layout_manager
        exporter = SceneExporter(
            env=env, cfg=cfg, layout=layout, layout_path=layout_path, layout_hash=layout_hash,
            installation=installation, base=base, arm_index=arm_index,
            mechanisms=mechanisms, installations=installations,
            fem_report=fem_report, fem_valid=fem_valid,
            finger_scale=cfg.task.finger_scale, adapter_offset_m=cfg.task.adapter_offset_m,
            texture_max_resolution=cfg.task.texture_max_resolution, task_name=task)
        # The upstream reward parser uses NumPy, whereas CUDA scene wrappers
        # return Torch poses. Normalize at this manager boundary.
        robodojo.original_reward_poses(manager)
        env.reward_manager.init_state()
        numpy_pose = robodojo.numpy_pose

        scene, target, object_pose = exporter.export(plans_dir / "operation_000",
                                                     cfg.task.target_label)
        store.record("scene", plans_dir / "operation_000")
        if mode == "assembly":
            from soft_sim.tasks.assembly import run_assembly_motion
            run_assembly_motion(env, run_dir, cfg.simulation.timestep_s, scene)
            store.record("assembly_motion", run_dir / "assembly_motion")
            completed = True
            return store.update(execution="completed")

        if args.static_camera_probe_frames:
            from soft_sim.validation.camera_stability import feature_motion
            images = {name: [] for name in ("left_rgb", "right_rgb", "external_rgb")}
            before_nodes = [tool.positions() for tool in env.tools]
            before_object = np.concatenate([np.atleast_1d(value) for value in object_pose()])
            before_cameras = env.tool.camera_poses.copy()
            for _ in range(args.static_camera_probe_frames):
                colors, _ = env.tool.capture(env.sim.sim)
                for name, pixels in zip(images, (*colors, np.asarray(external.get_data())[..., :3])):
                    images[name].append(pixels.astype(np.uint8).copy())
            after_nodes = [tool.positions() for tool in env.tools]
            node_delta = max(float(np.abs(a - b).max()) for arm_a, arm_b in zip(before_nodes, after_nodes)
                             for a, b in zip(arm_a, arm_b))
            object_delta = float(np.abs(np.concatenate([np.atleast_1d(value) for value in object_pose()])
                                   - before_object).max())
            camera_delta = float(np.abs(env.tool.camera_poses - before_cameras).max())
            report = dict(scope="Repeated zero-time raw renders, with no physics steps. Diagnostic only; "
                               "does not establish stability during task execution.",
                          frame_count=args.static_camera_probe_frames,
                          rendering=env.tool.render_settings,
                          maximum_node_component_change_m=node_delta,
                          maximum_object_pose_component_change=object_delta,
                          maximum_camera_matrix_change=camera_delta,
                          cameras={name: [feature_motion(a, b) for a, b in zip(frames, frames[1:])]
                                   for name, frames in images.items()})
            np.savez_compressed(diagnostics / "static_camera_probe.npz",
                                **{name: np.stack(frames) for name, frames in images.items()})
            store.write_json("diagnostics/static_camera_probe.json", report)
            if max(node_delta, object_delta, camera_delta) > 1e-7:
                raise RuntimeError("Static camera probe changed scene state; see diagnostics")
            print("STATIC_CAMERA_PROBE", args.static_camera_probe_frames, "frames", flush=True)
        if mode == "export":
            print("OFFICIAL_TASK_EXPORTED", task_name, scene["obstacle_count"], "obstacles", flush=True)
            completed = True
            return store.update(execution="completed")

        def plan_snapshot(snapshot, support_label):
            heights = ([cfg.planning.contact_height_candidates_m[0]]
                       if args.contact_height is not None else
                       [value * cfg.task.finger_scale for value in cfg.planning.contact_height_candidates_m])
            indent = (cfg.planning.indent_m if args.indent is not None
                      else (.004 if support_label else .008))
            for attempt, height in enumerate(heights):
                destination = snapshot / f"attempt_{attempt:03d}"
                arguments = ["--scene", snapshot, "--output", destination,
                             "--contact-height", height, "--indent", indent,
                             "--friction", cfg.planning.friction,
                             "--lift-height", cfg.planning.lift_height_m,
                             "--max-attempts", cfg.planning.max_attempts,
                             "--release-gap", cfg.planning.release_gap_m,
                             "--sample-dt", cfg.planning.sample_dt_s]
                if support_label:
                    arguments.extend(["--place-on", snapshot / f"task_object_{support_label}.npz"])
                print("TASK_PLANNING", snapshot.name, "contact_height", height, "indent", indent, flush=True)
                process = run_process("planning", "soft_sim.cli.plan", *arguments,
                                      log_path=destination / "planning.log")
                if process.ok and (destination / "plan.json").exists():
                    return destination / "plan.json"
            raise RuntimeError(f"Motion planning failed for {snapshot}; see attempt_*/planning.log")

        def read_plan(path, snapshot_scene):
            loaded = json.loads(path.read_text())
            if (loaded.get("status") != "planned" or not loaded.get("collision_checked")
                    or loaded["scene"].get("layout_sha256") != layout_hash
                    or loaded["scene"].get("robot_name") != cfg.task.robot
                    or not np.isclose(loaded["scene"].get("finger_scale", 1.), cfg.task.finger_scale)):
                raise ValueError("Plan must match the task layout, robot and finger scale")
            if loaded["scene"].get("finger_installation") != snapshot_scene.get("finger_installation"):
                raise ValueError("Plan installation differs from the actual finger/fixture transforms")
            np.testing.assert_allclose(snapshot_scene["initial_q"], loaded["initial_q"], atol=.005)
            np.testing.assert_allclose(snapshot_scene["initial_object_position"],
                                       loaded["scene"]["initial_object_position"], atol=.002)
            with np.load(path.parent / loaded["trajectory_file"]) as archive:
                trajectories = {key: archive[key].copy() for key in archive.files}
            return loaded, trajectories

        operations = ([(cfg.task.target_label, None)] if task_name != "stack_blocks" else
                      [("block_1", "block_0"), ("block_2", "block_1")])
        plan_path = (plan_snapshot(plans_dir / "operation_000", operations[0][1])
                     if mode == "collect" else Path(args.grasp_plan))
        plan, paths = read_plan(plan_path, scene)
        opened = plan["candidate"]["opened"]
        recorded_robots = env.robot_manager.robot_key
        joint_names = [f"robot{index}/{name}" for index, articulation in enumerate(recorded_robots)
                       for name in articulation.joint_names]
        parked_commands = [articulation.data.joint_pos[0].clone() for articulation in recorded_robots]
        calibration = dict(finger_intrinsics=np.stack([camera_intrinsics(cfg.camera, 2.8)] * 2).tolist(),
            finger_resolution=[256, 256], **external_calibration,
            material=cfg.to_dict()["material"], task_scene=scene,
            robot_type=arm.calibration_robot_type, joint_names=joint_names, active_arm=arm_index,
            finger_scale=cfg.task.finger_scale, joint_control=cfg.task.joint_control,
            rendering=env.tool.render_settings, finger_projection=env.tool.camera_projection)
        if hasattr(env.tool, "fixtures"):
            calibration["fixture_paths"] = env.tool.fixtures.paths
        recorder = EpisodeRecorder(capture_dir, cfg.recording.fps, calibration,
            dataset_root=dataset_root, repo_id=cfg.recording.repo_id,
            task=env.gen_instruction(0)[0], run_id=store.run_id,
            success_only=cfg.recording.success_only)
        store.update(capture="capturing")
        np.savez_compressed(capture_dir / "finger_reference.npz", **env.tool.reference_state)
        sim_step, last_width = 0, opened
        checkpoints, executed_plans = [], []
        task_labels = list(scene["task_objects"])
        task_objects = [manager.get_scene_object(
            inst_name=manager.get_instance_name(label=label, env_idx=0), env_idx=0) for label in task_labels]
        recorder.calibration["task_object_labels"] = task_labels
        # Register the task's original requirements once, as EvalEnv.run_eval
        # does. RewardManager.step evaluates their transitions during motion.
        env.run_reward()
        if hasattr(env, "get_score"):
            env.get_score()
        rejected = False
        stride = round(1 / (cfg.recording.fps * cfg.simulation.timestep_s))
        collection_started = time.perf_counter()
        phase_timings = []
        for operation_index, (target_label, support_label) in enumerate(operations):
            operation_dir = plans_dir / f"operation_{operation_index:03d}"
            if operation_index:
                current_scene, target, object_pose = exporter.export(operation_dir, target_label)
                store.record(f"scene.operation_{operation_index:03d}", operation_dir)
                try:
                    plan_path = plan_snapshot(operation_dir, support_label)
                    plan, paths = read_plan(plan_path, current_scene)
                except (RuntimeError, ValueError, AssertionError) as error:
                    # A valid recorded attempt can fail to find its next move.
                    # Keep its failure label; a physics/sensor fault instead
                    # remains incomplete through the outer exception handler.
                    failure = dict(operation_index=operation_index, target_label=target_label,
                                   message=str(error))
                    store.write_json("diagnostics/planning_failure.json", failure)
                    checkpoints.append(dict(operation_index=operation_index, target_label=target_label,
                                            phase="planning_failed"))
                    print("TASK_PLANNING_REJECTED", json.dumps(failure), flush=True)
                    rejected = True
                    break
            operation_initial_z = float(object_pose()[0][2])
            executed_plans.append(dict(operation_index=operation_index, target_label=target_label,
                support_label=support_label, plan_file=str(plan_path.relative_to(run_dir))
                if plan_path.is_relative_to(run_dir) else str(plan_path),
                sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest()))
            store.write_json("plans/executed.json", executed_plans)
            for phase, duration, trajectory, width in pickup_phases(plan, paths):
                print("TASK_PHASE", operation_index, target_label, phase, flush=True)
                phase_started = time.perf_counter()
                phase_capture_s = 0.
                phase_physics_s = 0.
                steps = round(duration / cfg.simulation.timestep_s)
                phase_positions, phase_velocities, phase_widths = phase_commands(
                    trajectory, steps, cfg.simulation.timestep_s, last_width, width)
                position_commands = torch.as_tensor(phase_positions, device=robot.device, dtype=torch.float32)
                velocity_commands = torch.as_tensor(phase_velocities, device=robot.device, dtype=torch.float32)
                zero_commands = torch.zeros_like(position_commands) if cfg.task.joint_control == "kinematic" else None
                preparation_s = time.perf_counter() - phase_started
                for index in range(steps):
                    physics_started = time.perf_counter()
                    command = position_commands[index:index + 1]
                    jaw = float(phase_widths[index])
                    if cfg.task.joint_control == "kinematic":
                        robot.write_joint_state_to_sim(command, zero_commands[index:index + 1])
                    robot.set_joint_position_target(command)
                    robot.set_joint_velocity_target(velocity_commands[index:index + 1])
                    parked_commands[arm_index] = command[0]
                    for parked_index, articulation in enumerate(recorded_robots):
                        if parked_index != arm_index:
                            articulation.set_joint_position_target(parked_commands[parked_index][None])
                    env.sim.scene.update(cfg.simulation.timestep_s)
                    env.sim_step(render=False)
                    phase_physics_s += time.perf_counter() - physics_started
                    sim_step += 1
                    if sim_step % stride == 0:
                        capture_started = time.perf_counter()
                        rgb, depth, all_nodes = capture_mounted_tools(env.tools, arm_index, env.sim.sim)
                        nodes = all_nodes[arm_index]
                        position, quat = object_pose()
                        minimum_jacobians = np.empty((len(env.tools), 2))
                        maximum_pin_errors = np.empty((len(env.tools), 2))
                        for tool_index, tool in enumerate(env.tools):
                            for side, points in enumerate(all_nodes[tool_index]):
                                mount = tool.mounts[side]
                                local = (points - mount[:3, 3]) @ mount[:3, :3]
                                rest, tets = tool.local_nodes[side], tool.sim_tets[side]
                                jacobian = tet_volumes(local, tets) / tool.rest_volumes[side]
                                pin_error = float(np.linalg.norm(local - rest, axis=1)[tool.pin_masks[side]].max())
                                minimum_jacobians[tool_index, side] = jacobian.min()
                                maximum_pin_errors[tool_index, side] = pin_error
                                if not np.isfinite(jacobian).all() or jacobian.min() <= 0 or pin_error > 1e-5:
                                    failure = dict(phase=phase, time_s=sim_step * cfg.simulation.timestep_s,
                                                   robot=tool_index, side=side,
                                                   minimum_jacobian=float(jacobian.min()),
                                                   maximum_pin_error_m=pin_error)
                                    store.write_json("diagnostics/physics_failure.json", failure)
                                    np.savez_compressed(diagnostics / "invalid_fem_sample.npz",
                                                        nodes=points, reference=rest, tets=tets, mount=mount)
                                    raise RuntimeError("FEM sample rejected; capture retained as incomplete: "
                                                       + str(failure))
                        object_poses = [numpy_pose(obj.get_world_pose()) for obj in task_objects]
                        fixture_state = {}
                        if hasattr(env.tool, "fixtures"):
                            error = env.tool.fixtures.mount_error(env.tool.mounts)
                            if (error["maximum_translation_error_m"] > 1e-5
                                    or error["maximum_rotation_matrix_error"] > 1e-5):
                                raise RuntimeError("Physical fixtures failed to follow measured mounts: "
                                                   + str(error))
                            fixture_state["fixture_to_world"] = env.tool.fixtures.poses()
                            fixture_state["fixture_pre_step_translation_error_m"] = error["maximum_translation_error_m"]
                            fixture_state["fixture_pre_step_rotation_matrix_error"] = error["maximum_rotation_matrix_error"]
                        recorder.append(*rgb, np.asarray(external.get_data())[..., :3].astype(np.uint8).copy(),
                            time_s=sim_step * cfg.simulation.timestep_s, phase=phase,
                            joint_positions=np.concatenate([articulation.data.joint_pos[0].cpu().numpy()
                                                            for articulation in recorded_robots]),
                            joint_velocities=np.concatenate([articulation.data.joint_vel[0].cpu().numpy()
                                                             for articulation in recorded_robots]),
                            commanded_joint_positions=torch.cat(parked_commands).cpu().numpy(),
                            object_position=np.asarray(position), object_orientation_wxyz=np.asarray(quat),
                            task_object_positions=np.stack([pose[0] for pose in object_poses]),
                            task_object_orientations_wxyz=np.stack([pose[1] for pose in object_poses]),
                            operation_index=operation_index,
                            all_finger_minimum_jacobians=minimum_jacobians,
                            all_finger_maximum_pin_errors_m=maximum_pin_errors,
                            finger_camera_to_world=env.tool.camera_poses, finger_depth_m=depth,
                            finger_mounts=env.tool.mounts, finger_nodes_left=nodes[0],
                            finger_nodes_right=nodes[1], width_parameter_m=jaw, **fixture_state)
                        env.reward_manager.step(env_idx_list=[0])
                        phase_capture_s += time.perf_counter() - capture_started
                phase_timings.append(dict(operation_index=operation_index, phase=phase,
                    physics_steps=steps, simulation_s=steps * cfg.simulation.timestep_s,
                    wall_s=time.perf_counter() - phase_started,
                    command_preparation_s=preparation_s, capture_and_record_s=phase_capture_s,
                    control_and_physics_s=phase_physics_s))
                lift = float(object_pose()[0][2]) - operation_initial_z
                checkpoints.append(dict(operation_index=operation_index, target_label=target_label,
                                        phase=phase, object_lift_m=lift))
                if phase == "probe_lift" and lift < .005:
                    print("GRASP_REJECTED", target_label, "Object did not follow probe lift", flush=True)
                    rejected = True
                    break
                last_width = width
            if rejected:
                break
        actual_lift = float(object_pose()[0][2]) - operation_initial_z
        collection_wall_s = time.perf_counter() - collection_started
        store.write_json("diagnostics/collection_timing.json", dict(
            scope="Collection only, excluding initial setup and dataset finalization; includes inter-operation planning",
            wall_s=collection_wall_s, physics_steps=sim_step,
            simulation_s=sim_step * cfg.simulation.timestep_s,
            captured_frames=len(recorder.rows), timestep_s=cfg.simulation.timestep_s,
            fps=cfg.recording.fps, rendering=env.tool.render_settings,
            follow_backend=follow_backend, phases=phase_timings))
        rewards = env.reward_manager.get_reward()
        success = bool(rewards[0] > .999)
        scores = env.reward_manager.get_score(reward_lst=rewards) if hasattr(env, "get_score") else None
        result = dict(task=task_name, success=success, object_lift_m=actual_lift,
            official_task_manager=True, official_benchmark_evaluation=False,
            embodiment_variant=scene["embodiment_variant"], active_robot=arm.name,
            checkpoints=checkpoints,
            joint_control=cfg.task.joint_control, task_rewards=rewards, task_scores=scores,
            task_object_labels=task_labels, executed_plans=executed_plans,
            coupling="prescribed FEM base nodes, no reaction torque feedback", config=cfg.to_dict())
        store.write_json("results/task.json", result)
        store.update(task_success=success)
        recorder.close(complete=True, result={k: v for k, v in result.items() if k != "config"})
        metadata = recorder._read_metadata()
        store.update(capture=metadata["status"],
                     datasets=([dict(dataset_id=dataset_id, dataset_root=storage.relative(dataset_root),
                                     episode_index=metadata.get("episode_index"))]
                               if recorder.committed else []))
        print("TASK_RESULT", json.dumps(result), flush=True)
        completed = True
        store.update(execution="completed")
        if recorder.committed:
            print(f"RUN_COMPLETE {store.run_id} dataset={dataset_id} "
                  f"episode={metadata.get('episode_index')} root={dataset_root}")
        else:
            print(f"RUN_DISCARDED {store.run_id} task_success={success}", flush=True)
    except BaseException as error:
        import traceback
        traceback.print_exc()
        store.write_json("run_failure.json", dict(
            error_type=type(error).__name__, message=str(error)))
        store.append_error(f"{type(error).__name__}: {error}")
        store.update(execution="failed")
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
                store.update(capture=recorder._read_metadata()["status"])
        finally:
            store.write_json("run_status.json", dict(
                completed=completed, mode=mode, task=task,
                joint_control=cfg.task.joint_control))
            sys.stdout.flush()
            sys.stderr.flush()
            # Kit's immediate shutdown can swallow an active exception and
            # return 0. A failed batch must remain failed for its caller.
            if not completed:
                os._exit(1)
        # Recording/conversion is flushed above. Skip stage teardown: legacy
        # deformable tensor views can stall Kit's graceful close after export.
        app.close(wait_for_replicator=False, skip_cleanup=True)
