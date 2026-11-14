"""Run original RoboDojo task managers/layouts with replacement FEM fingers.

This is an embodiment variant, not a stock dual-X5 benchmark score. Export the
actual task scene, plan with plan_soft_grasp.py, then replay in the same task.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "RoboDojo"))
from shared.config import ExperimentConfig
from shared.geometry import camera_intrinsics
from shared.tool_mount import mounts_from_tool
from backends.isaac_sim.app_launcher import create_simulation_app
from isaac51_tool import MountedSoftTool
from motion_execution import layout_target_parameters


def world_geometry(stage, path, world_from_root=None):
    from pxr import Usd, UsdGeom
    vertices, faces, cursor = [], [], 0
    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(path)
    correction = np.eye(4)
    if world_from_root is not None:
        correction = world_from_root @ np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root)).T)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = correction @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points = points @ transform[:3, :3].T + transform[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if np.all(counts == 3):
            faces.append(indices.reshape(-1, 3) + cursor)
        else:
            triangles, start = [], 0
            for count in counts:
                polygon = indices[start:start + count]
                triangles.extend([[cursor + polygon[0], cursor + polygon[i], cursor + polygon[i + 1]]
                                  for i in range(1, count - 1)])
                start += count
            faces.append(np.asarray(triangles, dtype=np.int32))
        vertices.append(points)
        cursor += len(points)
    if not vertices:
        raise ValueError(f"No triangle geometry: {path}")
    return np.concatenate(vertices), np.concatenate(faces).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="general_pickup")
    parser.add_argument("--layout", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot", choices=("x5", "franka"), default="x5")
    parser.add_argument("--active-arm", type=int, choices=(0, 1), default=1)
    parser.add_argument("--finger-scale", type=float, help="Uniform geometry scale (X5: 0.75; Franka: 1)")
    parser.add_argument("--mount-profile", choices=("measured", "parallel_legacy"), default="measured")
    parser.add_argument("--target-label")
    parser.add_argument("--mode", choices=("export", "assembly", "replay", "collect"), default="export")
    parser.add_argument("--grasp-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=.001)
    parser.add_argument("--adapter-offset", type=float, default=.008)
    parser.add_argument("--fem-mesh", type=Path)
    parser.add_argument("--solver-iterations", type=int, default=80)
    parser.add_argument("--vertex-velocity-damping", type=float)
    parser.add_argument("--joint-control", choices=("position", "kinematic"), default="position",
                        help="Position targets use the original X5 drives; kinematic is a diagnostic replay")
    parser.add_argument("--contact-height", type=float, help="Contact band on the scaled finger, in meters")
    parser.add_argument("--indent", type=float, help="Per-side grasp preload, in meters")
    parser.add_argument("--planning-friction", type=float, default=.51)
    parser.add_argument("--planning-attempts", type=int, default=96)
    parser.add_argument("--static-camera-probe-frames", type=int, default=0,
                        help="Optional diagnostic captures without physics steps; saved outside the dataset")
    parser.add_argument("--texture-max-resolution", type=int, choices=(512,1024,2048,4096), default=1024)
    args = parser.parse_args()
    if args.static_camera_probe_frames not in (0,) and args.static_camera_probe_frames < 2:
        parser.error("Static camera probe requires at least two frames, or zero to disable")
    if args.target_label is None:
        args.target_label = "block_1" if args.task == "stack_blocks" else "target"
    if args.finger_scale is None:
        args.finger_scale = .75 if args.robot == "x5" else 1.
    if args.robot == "x5":
        if args.vertex_velocity_damping is None:
            args.vertex_velocity_damping = 20.
        if args.fem_mesh is None:
            args.fem_mesh = ROOT / "data/generated/soft_finger_tetgen_4mm.npz"
            if not args.fem_mesh.is_file():
                parser.error("Generate a conforming mesh first: envs/meshing/.venv/bin/python scripts/prepare_mesh.py "
                             "--method tetgen --pitch .004 --output data/generated/soft_finger_tetgen_4mm.npz")
    if not np.isfinite(args.finger_scale) or not .25 <= args.finger_scale <= 1.:
        parser.error("Finger scale must be in [0.25, 1]")
    installation = None
    if args.robot == "x5" and args.mount_profile == "measured":
        from benchmarks.x5_mount_geometry import measured_installation
        installation = measured_installation(args.finger_scale)
    if args.mode == "replay" and args.grasp_plan is None:
        parser.error("Replay requires a collision-checked plan exported from this task layout")
    if args.mode == "assembly" and (args.robot != "x5" or installation is None or args.joint_control != "position"):
        parser.error("Assembly sweep requires measured X5 mounts and original position drives")
    if args.mode == "replay" and args.task != "general_pickup":
        parser.error("Use --mode collect for stack_blocks; replay accepts a general_pickup plan")
    if args.mode == "collect" and args.task not in ("general_pickup", "stack_blocks"):
        parser.error("Motion-planning collection currently supports general_pickup and stack_blocks")
    if args.mode == "collect" and args.grasp_plan is not None:
        parser.error("Collect plans from the current scene; use replay for a saved grasp plan")
    stride = round(1 / (args.fps * args.dt))
    if stride < 1 or not np.isclose(stride * args.dt, 1 / args.fps):
        parser.error("FPS must divide the physics step rate")
    layout_path = (args.layout or ROOT / f"RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0/{args.task}_{args.seed}.json").resolve()
    layout = json.loads(layout_path.read_text())
    layout_hash = hashlib.sha256(layout_path.read_bytes()).hexdigest()
    plan = json.loads(args.grasp_plan.read_text()) if args.grasp_plan else None
    if plan and (plan.get("status") != "planned" or not plan.get("collision_checked") or
                 plan["scene"].get("layout_sha256") != layout_hash or plan["scene"].get("task") != args.task):
        parser.error("Plan must match this official task and unmodified saved layout")
    if plan and plan["scene"].get("finger_installation") != installation:
        parser.error("Plan installation differs from this mount profile; legacy plans require --mount-profile parallel_legacy")
    cfg = ExperimentConfig()
    cfg.gripper.profile, cfg.gripper.open_jaw_width_m = "panda", .08
    cfg.gripper.base_height_m, cfg.gripper.adapter_offset_m = 0., args.adapter_offset
    cfg.simulation.timestep_s, cfg.simulation.solver_iterations = args.dt, args.solver_iterations
    cfg.simulation.deformable_vertex_velocity_damping = args.vertex_velocity_damping
    cfg.simulation.contact_offset_m = .00025
    cfg.camera.width = cfg.camera.height = 256
    cfg.object.fixed = False
    target_parameters = layout_target_parameters(layout, args.target_label)
    physics = target_parameters.get("physics", {})
    cfg.object.mass_kg = physics.get("mass", cfg.object.mass_kg)
    cfg.object.friction = physics.get("friction", cfg.object.friction)
    if args.fem_mesh:
        cfg.mesh.tetra_mesh = str(args.fem_mesh)
    cfg.validate()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.finger_scale != 1:
        from x5_soft_tool import scale_config
        scale_config(cfg, args.finger_scale, out)
    app = create_simulation_app(cfg.simulation)
    recorder = None
    completed = False
    try:
        import carb
        carb.settings.get_settings().set_int("/rtx-transient/resourcemanager/maxMipCount",
                                            int(np.log2(args.texture_max_resolution))+1)
        if args.robot == "franka":
            import omni.kit.app
            omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
                "isaacsim.robot_motion.motion_generation", True)
            from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
            from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
        from omegaconf import OmegaConf
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        from scipy.spatial.transform import Rotation
        import torch
        import omni.replicator.core as rep
        from task.RoboDojo.task_registry import load_task_class
        from utils.load_file import load_yaml

        def external_camera(stage):
            camera = UsdGeom.Camera.Define(stage, "/World/SoftTaskCamera")
            camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
            camera.CreateFocalLengthAttr(18.)
            camera.CreateHorizontalApertureAttr(24.)
            camera.CreateVerticalApertureAttr(18.)
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(1., -1.3, 1.6),
                Gf.Vec3d(.1, -.03, .85), Gf.Vec3d(0, 0, 1)).GetInverse())
            product = rep.create.render_product(str(camera.GetPath()), (640, 480))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([product])
            return annotator, product

        task_name, Task = load_task_class(args.task)
        sim_config = load_yaml(ROOT / "RoboDojo/env_cfg/sim/sim_config.yml")
        sim_config.update(device="cuda:0", use_fabric=True, seed=args.seed, dt=args.dt, decimation=1)
        sim_config["scene"]["num_envs"] = 1
        sim_config["physx"] = dict(min_position_iteration_count=args.solver_iterations,
            max_position_iteration_count=255,
            gpu_max_rigid_contact_count=1048576, gpu_max_rigid_patch_count=65536,
            gpu_found_lost_pairs_capacity=262144, gpu_found_lost_aggregate_pairs_capacity=1048576,
            gpu_total_aggregate_pairs_capacity=262144, gpu_collision_stack_size=33554432,
            gpu_max_soft_body_contacts=262144, gpu_max_particle_contacts=65536)
        sim_config["render"] = dict(antialiasing_mode="FXAA", rendering_mode="balanced",
            carb_settings={"/rtx-transient/resourcemanager/texturestreaming/memoryBudget": .15,
                "/rtx-transient/resourcemanager/maxMipCount": int(np.log2(args.texture_max_resolution))+1})
        # Original task scene and layout; only the robot embodiment and sensors
        # change. No policy server or stock gripper contact is used.
        robot_cfg = dict(robots=[dict(robot_type="arm", robot_name="franka", coupled=False,
            need_planner=False, default_root_pos=[0., -.45, .765],
            default_root_rot=[.7071067811865476, 0., 0., .7071067811865476],
            grasp_perfect_direction="top_down_little_right")])
        if args.robot == "x5":
            robot_cfg = load_yaml(ROOT / "RoboDojo/env_cfg/robot/dual_x5.yml")
            for robot_config in robot_cfg["robots"]:
                robot_config["need_planner"] = False
        config = OmegaConf.create(dict(sim=sim_config, robot=robot_cfg,
            scene=load_yaml(ROOT / "RoboDojo/env_cfg/scene/default.yml"),
            camera=dict(default_frequency=None, annotator=dict(common=dict(enabled=False))),
            task_env=load_yaml(ROOT / f"RoboDojo/task/RoboDojo/config/{task_name}.yml")))
        kin = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config("Franka")) if args.robot == "franka" else None
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        base[:3, 3] = [0., -.45, .765]
        q_initial = np.array([0., .19634954084936207, 0., -2.617993877991494,
                              0., 2.941592653589793, .7853981633974483, .04, .04])
        arm_index = args.active_arm if args.robot == "x5" else 0
        if args.robot == "x5":
            from x5_soft_tool import FINGER_LINKS, JOINT_NAMES, mounts_from_q, mounts_from_links
            q_initial = np.r_[np.zeros(6), .044, .044]
            base[:3, 3] = robot_cfg["robots"][arm_index]["default_root_pos"]
        finger_links = FINGER_LINKS if args.robot == "x5" else ("panda_leftfinger", "panda_rightfinger")

        def mounts_for(q):
            position, rotation = kin.compute_forward_kinematics("panda_hand", np.asarray(q[:7]))
            hand = np.eye(4)
            hand[:3, :3] = rotation @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
            hand[:3, 3] = position + rotation @ np.array([0., 0., .075])
            return mounts_from_tool(cfg.gripper, base @ hand, float(np.clip(q[-2:].sum(), 0, .08)))

        class SoftTask(Task):
            tool = None
            tool_initialized = False
            tools = None

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                # The replacement has its own two interior cameras and one
                # overview. Avoid allocating unused factory wrist products.
                self.camera_manager.init_config = OmegaConf.create(dict(
                    default_frequency=None, annotator=dict(common=dict(enabled=False))))

            def _post_setup_scene(self, sim):
                super()._post_setup_scene(sim)
                self.tools = []
                self.removed_finger_colliders = {}
                for index in range(len(robot_cfg["robots"])):
                    for name in finger_links:
                        path = f"/World/envs/env_0/robot{index}/{name}"
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
                    arm_base = base.copy()
                    arm_base[:3, 3] = robot_cfg["robots"][index]["default_root_pos"]
                    mounts = mounts_from_q(q_initial, arm_base, installation) if args.robot == "x5" else mounts_for(q_initial)
                    filtered = ("link6", "link7", "link8") if args.robot == "x5" else ("panda_hand", "panda_link7")
                    if args.robot == "x5" and installation is not None:
                        filtered = ("link7", "link8")
                    filter_paths = [f"/World/envs/env_0/robot{index}/{name}" for name in filtered]
                    if args.robot == "x5" and installation is not None:
                        filter_paths = {side: [f"/World/envs/env_0/robot{index}/{name}"] for side, name in enumerate(FINGER_LINKS)}
                    tool_config = deepcopy(cfg)
                    tool_config.simulation.render = index == arm_index
                    tool = MountedSoftTool(self.stage, tool_config, mounts,
                        root=f"/World/SoftTool{index}",
                        filtered_bodies=filter_paths,
                        explicit_fem_mesh=bool(args.fem_mesh), max_depenetration_velocity=1.,
                        surface_max_edge_m=.002 * args.finger_scale,
                        rt_subframes=4, lighting_samples_per_pixel=8)
                    self.tools.append(tool)
                    if args.robot == "x5" and installation is not None:
                        from benchmarks.x5_mount_geometry import MountedFixtures
                        tool.fixtures = MountedFixtures(self.stage,
                            [f"/World/envs/env_0/robot{index}/{name}" for name in FINGER_LINKS],
                            args.finger_scale, installation, tool.paths)
                        tool.fixtures.follow(mounts)
                self.tool = self.tools[arm_index]

            def sim_step(self, render=True):
                if self.tool is not None:
                    if not self.tool_initialized:
                        for tool in self.tools:
                            tool.initialize()
                        self.tool_initialized = True
                    self.follow_tools()
                return super().sim_step(render=render)

            def follow_tools(self):
                for index, tool in enumerate(self.tools):
                    articulation = self.robot_manager.robot_key[index]
                    if args.robot == "x5":
                        poses = articulation.data.body_link_pose_w[0].cpu().numpy()
                        links = []
                        for name in FINGER_LINKS:
                            pose = poses[articulation.body_names.index(name)]
                            matrix = np.eye(4)
                            matrix[:3, 3] = pose[:3]
                            matrix[:3, :3] = Rotation.from_quat(np.r_[pose[4:7], pose[3]]).as_matrix()
                            links.append(matrix)
                        tool.follow(mounts_from_links(links, installation))
                    else:
                        tool.follow(mounts_for(articulation.data.joint_pos[0].cpu().numpy()))

        import time
        env = SoftTask(config, app)
        env.success = [True]
        env.scene_manager.layout_manager.set_saved_layout(0, layout)
        print("TASK_STAGE before_reset", time.time(), flush=True)
        env.reset(seed=[args.seed])
        print("TASK_STAGE after_reset", time.time(), flush=True)
        env.scene_manager.apply_saved_poses([0])
        print("TASK_STAGE after_apply_saved_poses", time.time(), flush=True)
        env.follow_tools()
        for tool in env.tools:
            tool.reset_to_mounts(tool.mounts)
        print("TASK_STAGE fem_reset_at_measured_mounts", time.time(), flush=True)
        for _ in range(round(.3/args.dt)):
            env.sim_step(render=False)
        print("TASK_STAGE after_300_steps", time.time(), flush=True)
        env.robot_manager.set_origin_endpose()
        robot = env.robot_manager.robot_key[arm_index]
        external, product = external_camera(env.stage)
        fem_state, fem_report = {}, []
        from shared.mesh import tet_volumes
        for index, tool in enumerate(env.tools):
            sides = []
            for side, nodes in enumerate(tool.positions()):
                mount = tool.mounts[side]
                local = (nodes-mount[:3,3])@mount[:3,:3]
                rest, tets = tool.local_nodes[side], tool.sim_tets[side]
                displacement = np.linalg.norm(local-rest,axis=1)
                jacobian = tet_volumes(local,tets)/tet_volumes(rest,tets)
                pin_error = float(displacement[tool.pin_masks[side]].max())
                entry = dict(minimum_jacobian=float(jacobian.min()),
                    maximum_displacement_m=float(displacement.max()), maximum_pin_error_m=pin_error)
                entry["valid"] = bool(np.isfinite(jacobian).all() and jacobian.min()>0 and pin_error<1e-5)
                sides.append(entry)
                fem_state[f"robot{index}_nodes_{side}"] = nodes
                fem_state[f"robot{index}_reference_{side}"] = rest
                fem_state[f"robot{index}_tets_{side}"] = tets
            fem_state[f"robot{index}_mounts"] = tool.mounts
            fem_report.append(sides)
        np.savez_compressed(out / "initial_fem.npz", **fem_state)
        (out / "initial_fem_validation.json").write_text(json.dumps(fem_report,indent=2))
        fem_valid = all(side["valid"] for arm in fem_report for side in arm)
        if not fem_valid and args.mode != "export":
            raise RuntimeError("Initial native-task FEM contains inverted cells or clamp slip")
        for tool in env.tools:
            tool.sync_render_surface()
        rgb, depth = env.tool.capture(env.sim.sim)
        np.savez_compressed(out / "preview_depth.npz", depth=depth, cameras=env.tool.camera_poses,
            intrinsics=np.stack([camera_intrinsics(cfg.camera, 2.8)]*2),
            nodes_left=env.tool.positions()[0], nodes_right=env.tool.positions()[1])
        from PIL import Image
        for name, pixels in zip(("left", "right"), rgb):
            Image.fromarray(pixels).save(out / f"preview_{name}.png")
        Image.fromarray(np.asarray(external.get_data())[...,:3].astype(np.uint8)).save(out / "preview_external.png")
        if args.robot == "x5":
            expected = mounts_from_q(robot.data.joint_pos[0].cpu().numpy(), base, installation)
            actual = env.tool.mounts
            position_error = float(np.linalg.norm(expected[:,:3,3]-actual[:,:3,3],axis=1).max())
            rotation_error = float(np.abs(expected[:,:3,:3]-actual[:,:3,:3]).max())
            report = dict(valid=position_error < .0001 and rotation_error < .001,
                          max_translation_error_m=position_error, max_rotation_matrix_error=rotation_error,
                          joint_names=robot.joint_names, measured_mounts=actual.tolist(),
                          fk_mounts=expected.tolist(), installation=installation)
            if hasattr(env.tool, "fixtures"):
                report["fixture_mount_errors"] = [tool.fixtures.mount_error(tool.mounts) for tool in env.tools]
                report["valid"] = report["valid"] and all(error["maximum_translation_error_m"]<1e-5
                    and error["maximum_rotation_matrix_error"]<1e-5 for error in report["fixture_mount_errors"])
            (out / "mount_validation.json").write_text(json.dumps(report, indent=2))
            if not report["valid"]:
                raise RuntimeError("Native X5 articulation and soft-finger planning mounts disagree")
        manager = env.scene_manager.layout_manager
        # The upstream reward parser uses NumPy, whereas CUDA scene wrappers
        # return Torch poses. Normalize at this manager boundary.
        def numpy_pose(pose):
            return tuple(v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v) for v in pose)
        original_instance_pose = manager.get_instance_pose
        manager.get_instance_pose = lambda *a, **kw: numpy_pose(original_instance_pose(*a, **kw))
        env.reward_manager.init_state()
        def export_scene(destination, target_label):
            out = Path(destination)
            out.mkdir(parents=True, exist_ok=True)
            target_parameters = layout_target_parameters(layout, target_label)
            target_name = manager.get_instance_name(label=target_label, env_idx=0)
            target = manager.get_scene_object(inst_name=target_name, env_idx=0)
            def object_pose():
                return numpy_pose(target.get_world_pose())
            target_path = target.prim_path
            def pose_matrix(pose):
                position, quaternion = numpy_pose(pose)
                result = np.eye(4)
                result[:3, 3] = position
                result[:3, :3] = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_matrix()
                return result
            vertices, faces = world_geometry(env.stage, target_path, pose_matrix(object_pose()))
            np.savez_compressed(out / "object_geometry.npz", vertices=vertices, faces=faces)
            task_objects = {}
            labels = ("block_0", "block_1", "block_2") if args.task == "stack_blocks" else (target_label,)
            for label in labels:
                instance = manager.get_instance_name(label=label, env_idx=0)
                obj = manager.get_scene_object(inst_name=instance, env_idx=0)
                position, quaternion = numpy_pose(obj.get_world_pose())
                v, f = world_geometry(env.stage, obj.prim_path, pose_matrix((position, quaternion)))
                filename = f"task_object_{label}.npz"
                np.savez_compressed(out / filename, vertices=v, faces=f)
                task_objects[label] = dict(prim_path=obj.prim_path, position=position.tolist(),
                    quaternion_wxyz=quaternion.tolist(), geometry_file=filename)
            print("TASK_TARGET_GEOMETRY", len(vertices), len(faces), flush=True)
            obstacles = {}
            count = 0
            exported = {target_path}
            for obj in env.scene_manager.get_objects([0]).values():
                if obj.prim_path in exported:
                    continue
                exported.add(obj.prim_path)
                print("TASK_EXPORT_GEOMETRY", obj.prim_path, flush=True)
                try:
                    actual_pose = pose_matrix(obj.get_world_pose()) if hasattr(obj, "get_world_pose") else None
                    v, f = world_geometry(env.stage, obj.prim_path, actual_pose)
                except ValueError:
                    continue
                import trimesh
                hull = trimesh.Trimesh(v, f, process=False).convex_hull
                v, f = hull.vertices, hull.faces
                obstacles.update({f"vertices_{count}": v, f"faces_{count}": f})
                count += 1
                print("TASK_OBSTACLE", count, obj.prim_path, flush=True)
            if args.robot == "x5":
                import trimesh
                other_index = 1-arm_index
                other = env.robot_manager.robot_key[other_index]
                poses = other.data.body_link_pose_w[0].cpu().numpy()
                for name, pose in zip(other.body_names, poses):
                    if name in FINGER_LINKS:
                        continue
                    actual_pose = pose_matrix((pose[:3], pose[3:]))
                    v, f = world_geometry(env.stage, f"/World/envs/env_0/robot{other_index}/{name}", actual_pose)
                    hull = trimesh.Trimesh(v, f, process=False).convex_hull
                    obstacles.update({f"vertices_{count}": hull.vertices, f"faces_{count}": hull.faces})
                    count += 1
                for nodes in env.tools[other_index].positions():
                    hull = trimesh.convex.convex_hull(nodes)
                    obstacles.update({f"vertices_{count}": hull.vertices, f"faces_{count}": hull.faces})
                    count += 1
                if installation is not None:
                    from benchmarks.x5_mount_geometry import fixture_meshes
                    from benchmarks.x5_soft_tool import finger_in_link
                    for side in (0, 1):
                        pose = env.tools[other_index].mounts[side] @ np.linalg.inv(finger_in_link(side, installation))
                        for mesh in fixture_meshes(args.finger_scale, side, installation).values():
                            mesh.apply_transform(pose)
                            obstacles.update({f"vertices_{count}": mesh.vertices, f"faces_{count}": mesh.faces})
                            count += 1
            np.savez_compressed(out / "obstacle_geometry.npz", count=count, **obstacles)
            scene = dict(task=task_name, layout=str(layout_path), layout_sha256=layout_hash,
                task_objects=task_objects,
                obstacle_count=count,
                target_parameters=target_parameters,
                target_label=target_label, target_prim=target_path,
                initial_q=robot.data.joint_pos[0].cpu().numpy().tolist(),
                initial_object_position=object_pose()[0].tolist(),
                table_z=.765, table_center_m=[0., -.05, .74], table_size_m=[1.4, 1.1, .05],
                world_from_robot_base=base.tolist(), hand_mount_offset_m=.075,
                adapter_offset_m=args.adapter_offset, finger_mesh=cfg.mesh.stl,
                collision_scope="task rigid and geometry objects + table; convex hulls; room excluded",
                embodiment_variant="single Franka with FEM soft fingers; original layout retained")
            scene.update(robot_name=args.robot, active_arm=arm_index, finger_scale=args.finger_scale,
                         finger_scale_to_m=cfg.mesh.scale_to_m, target_geometry_sha256=hashlib.sha256(
                             (out / "object_geometry.npz").read_bytes()).hexdigest())
            scene.update(material=cfg.to_dict()["material"],
                         fem_initialized_after_task_reset=True,
                         removed_factory_finger_colliders=env.removed_finger_colliders,
                         initial_fem_validation=fem_report,
                         physics_initialization_valid=fem_valid,
                         texture_max_resolution=args.texture_max_resolution,
                         deformable_mesh=[tool.mesh_settings for tool in env.tools],
                         mesh_topology=[tool.mesh_topology for tool in env.tools])
            if args.robot == "x5":
                scene["collision_scope"] += "; parked second X5 links and deformed soft fingers included"
            if args.robot == "x5":
                from x5_soft_tool import MOUNT_X, HALF_ROOT_SPACING, INWARD_OFFSET
                scene.update(hand_mount_offset_m=MOUNT_X, adapter_offset_m=HALF_ROOT_SPACING-INWARD_OFFSET,
                    embodiment_variant="original dual X5 with replacement FEM fingers on native slides",
                    joint_names=robot.joint_names)
                scene["finger_installation"] = installation
                from benchmarks.x5_soft_tool import finger_in_link
                scene["finger_in_slide_transforms"] = [finger_in_link(side, installation).tolist() for side in (0, 1)]
                if installation is not None:
                    scene["adapter_offset_m"] = installation["closed_half_spacing_m"]
                    scene["hand_mount_offset_m"] += installation["forward_offset_m"]
            (out / "scene.json").write_text(json.dumps(scene, indent=2))
            return scene, target, object_pose

        scene, target, object_pose = export_scene(out, args.target_label)
        if args.mode == "assembly":
            from benchmarks.x5_assembly_motion import run_assembly_motion
            run_assembly_motion(env, out, args.dt, scene)
            completed = True
            return
        if args.static_camera_probe_frames:
            from check_camera_stability import feature_motion
            images = {name: [] for name in ("left_rgb", "right_rgb", "external_rgb")}
            before_nodes = [tool.positions() for tool in env.tools]
            before_object = np.r_[*object_pose()]
            before_cameras = env.tool.camera_poses.copy()
            for _ in range(args.static_camera_probe_frames):
                colors, _ = env.tool.capture(env.sim.sim)
                for name, pixels in zip(images, (*colors, np.asarray(external.get_data())[..., :3])):
                    images[name].append(pixels.astype(np.uint8).copy())
            after_nodes = [tool.positions() for tool in env.tools]
            node_delta = max(float(np.abs(a-b).max()) for arm_a, arm_b in zip(before_nodes, after_nodes)
                             for a, b in zip(arm_a, arm_b))
            object_delta = float(np.abs(np.r_[*object_pose()]-before_object).max())
            camera_delta = float(np.abs(env.tool.camera_poses-before_cameras).max())
            report = dict(scope="Repeated zero-time raw renders, with no physics steps. Diagnostic only; "
                          "does not establish stability during task execution.",
                          frame_count=args.static_camera_probe_frames, rendering=env.tool.render_settings,
                          maximum_node_component_change_m=node_delta,
                          maximum_object_pose_component_change=object_delta,
                          maximum_camera_matrix_change=camera_delta,
                          cameras={name: [feature_motion(a, b) for a, b in zip(frames, frames[1:])]
                                   for name, frames in images.items()})
            np.savez_compressed(out / "static_camera_probe.npz", **{name: np.stack(frames)
                                for name, frames in images.items()})
            (out / "static_camera_probe.json").write_text(json.dumps(report, indent=2))
            if max(node_delta, object_delta, camera_delta) > 1e-7:
                raise RuntimeError("Static camera probe changed scene state; see static_camera_probe.json")
            print("STATIC_CAMERA_PROBE", args.static_camera_probe_frames, "frames", flush=True)
        if args.mode == "export":
            print("OFFICIAL_TASK_EXPORTED", task_name, scene["obstacle_count"], "obstacles", flush=True)
            completed = True
            return
        from motion_execution import pickup_phases, trajectory_command

        def plan_snapshot(snapshot, support_label):
            heights = [args.contact_height] if args.contact_height is not None else [
                value*args.finger_scale for value in (.073, .055, .080)]
            indent = args.indent if args.indent is not None else (.004 if support_label else .008)
            for attempt, height in enumerate(heights):
                destination = snapshot / f"motion_plan_{attempt}"
                command = [str(ROOT / "envs/planning/.venv/bin/python"), str(ROOT / "benchmarks/plan_soft_grasp.py"),
                    "--scene", str(snapshot), "--output", str(destination), "--contact-height", str(height),
                    "--indent", str(indent), "--friction", str(args.planning_friction),
                    "--lift-height", ".14", "--max-attempts", str(args.planning_attempts)]
                if support_label:
                    command.extend(["--place-on", str(snapshot / f"task_object_{support_label}.npz")])
                print("TASK_PLANNING", snapshot.name, "contact_height", height, "indent", indent, flush=True)
                with (snapshot / f"planning_{attempt}.log").open("w") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode == 0 and (destination / "plan.json").exists():
                    return destination / "plan.json"
            raise RuntimeError(f"Motion planning failed for {snapshot}; see planning_*.log")

        def read_plan(path, snapshot_scene):
            loaded = json.loads(path.read_text())
            if (loaded.get("status") != "planned" or not loaded.get("collision_checked")
                    or loaded["scene"].get("layout_sha256") != layout_hash
                    or loaded["scene"].get("robot_name", "franka") != args.robot
                    or not np.isclose(loaded["scene"].get("finger_scale", 1.), args.finger_scale)):
                raise ValueError("Plan must match the task layout, robot and finger scale")
            if loaded["scene"].get("finger_installation") != snapshot_scene.get("finger_installation"):
                raise ValueError("Plan installation differs from the actual finger/fixture transforms")
            np.testing.assert_allclose(snapshot_scene["initial_q"], loaded["initial_q"], atol=.005)
            np.testing.assert_allclose(snapshot_scene["initial_object_position"],
                                       loaded["scene"]["initial_object_position"], atol=.002)
            with np.load(path.parent / loaded["trajectory_file"]) as archive:
                trajectories = {key: archive[key].copy() for key in archive.files}
            return loaded, trajectories

        operations = [(args.target_label, None)] if task_name != "stack_blocks" else [
            ("block_1", "block_0"), ("block_2", "block_1")]
        plan_path = plan_snapshot(out, operations[0][1]) if args.mode == "collect" else args.grasp_plan
        plan, paths = read_plan(plan_path, scene)
        opened = plan["candidate"]["opened"]
        from episode_recording import EpisodeRecorder
        recorded_robots = env.robot_manager.robot_key
        joint_names = [f"robot{index}/{name}" for index, articulation in enumerate(recorded_robots)
                       for name in articulation.joint_names]
        parked_commands = [articulation.data.joint_pos[0].clone() for articulation in recorded_robots]
        calibration = dict(finger_intrinsics=np.stack([camera_intrinsics(cfg.camera, 2.8)]*2).tolist(),
            finger_resolution=[256,256], external_resolution=[640,480],
            external_intrinsics=[[480.,0.,320.],[0.,480.,240.],[0.,0.,1.]],
            material=cfg.to_dict()["material"], task_scene=scene,
            robot_type="dual_x5_softfinger" if args.robot == "x5" else "franka_softfinger",
            joint_names=joint_names, active_arm=arm_index, finger_scale=args.finger_scale,
            joint_control=args.joint_control,
            rendering=env.tool.render_settings,
            finger_projection=env.tool.camera_projection)
        if hasattr(env.tool, "fixtures"):
            calibration["fixture_paths"] = env.tool.fixtures.paths
        recorder = EpisodeRecorder(out / "recording", args.fps, calibration,
            dataset_root=args.dataset_root, task=env.gen_instruction(0)[0])
        np.savez_compressed(out / "recording/finger_reference.npz", **env.tool.reference_state)
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
        for operation_index, (target_label, support_label) in enumerate(operations):
            if operation_index:
                snapshot = out / f"operation_{operation_index}_{target_label}"
                current_scene, target, object_pose = export_scene(snapshot, target_label)
                operation_initial_z = float(object_pose()[0][2])
                try:
                    plan_path = plan_snapshot(snapshot, support_label)
                    plan, paths = read_plan(plan_path, current_scene)
                except (RuntimeError, ValueError, AssertionError) as error:
                    # A valid recorded attempt can fail to find its next move.
                    # Keep its failure label; a physics/sensor fault instead
                    # remains incomplete through the outer exception handler.
                    failure = dict(operation_index=operation_index, target_label=target_label,
                                   message=str(error))
                    (out / "planning_failure.json").write_text(json.dumps(failure, indent=2))
                    checkpoints.append(dict(operation_index=operation_index, target_label=target_label,
                                            phase="planning_failed"))
                    print("TASK_PLANNING_REJECTED", json.dumps(failure), flush=True)
                    rejected = True
                    break
            operation_initial_z = float(object_pose()[0][2])
            executed_plans.append(dict(operation_index=operation_index, target_label=target_label,
                support_label=support_label, plan_file=str(plan_path.relative_to(out))
                if plan_path.is_relative_to(out) else str(plan_path),
                sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest()))
            (out / "executed_plans.json").write_text(json.dumps(executed_plans, indent=2))
            for phase, duration, trajectory, width in pickup_phases(plan, paths):
                print("TASK_PHASE", operation_index, target_label, phase, flush=True)
                steps = round(duration / args.dt)
                for index in range(steps):
                    positions, velocities, jaw = trajectory_command(
                        trajectory, steps*args.dt, (index+1)*args.dt, last_width, width)
                    command = torch.as_tensor(positions[None], device=robot.device, dtype=torch.float32)
                    if args.joint_control == "kinematic":
                        robot.write_joint_state_to_sim(command, torch.zeros_like(command))
                    robot.set_joint_position_target(command)
                    robot.set_joint_velocity_target(torch.as_tensor(velocities[None], device=robot.device,
                                                                   dtype=torch.float32))
                    parked_commands[arm_index] = command[0]
                    for parked_index, articulation in enumerate(recorded_robots):
                        if parked_index != arm_index:
                            articulation.set_joint_position_target(parked_commands[parked_index][None])
                    env.sim.scene.update(args.dt)
                    env.sim_step(render=False)
                    sim_step += 1
                    if sim_step % stride == 0:
                        for tool in env.tools:
                            tool.sync_render_surface()
                        rgb, depth = env.tool.capture(env.sim.sim)
                        position, quat = object_pose()
                        nodes = env.tool.positions()
                        minimum_jacobians = np.empty((len(env.tools), 2))
                        maximum_pin_errors = np.empty((len(env.tools), 2))
                        for tool_index, tool in enumerate(env.tools):
                            for side, points in enumerate(nodes if tool_index == arm_index else tool.positions()):
                                mount = tool.mounts[side]
                                local = (points-mount[:3,3])@mount[:3,:3]
                                rest, tets = tool.local_nodes[side], tool.sim_tets[side]
                                jacobian = tet_volumes(local,tets)/tet_volumes(rest,tets)
                                pin_error = float(np.linalg.norm(local-rest,axis=1)[tool.pin_masks[side]].max())
                                minimum_jacobians[tool_index, side] = jacobian.min()
                                maximum_pin_errors[tool_index, side] = pin_error
                                if not np.isfinite(jacobian).all() or jacobian.min()<=0 or pin_error>1e-5:
                                    failure = dict(phase=phase, time_s=sim_step*args.dt, robot=tool_index, side=side,
                                        minimum_jacobian=float(jacobian.min()), maximum_pin_error_m=pin_error)
                                    (out/"physics_failure.json").write_text(json.dumps(failure,indent=2))
                                    np.savez_compressed(out/"invalid_fem_sample.npz", nodes=points, reference=rest,
                                                        tets=tets, mount=mount)
                                    raise RuntimeError("FEM sample rejected; capture retained as incomplete: "+str(failure))
                        object_poses = [numpy_pose(obj.get_world_pose()) for obj in task_objects]
                        fixture_state = {}
                        if hasattr(env.tool, "fixtures"):
                            error = env.tool.fixtures.mount_error(env.tool.mounts)
                            if error["maximum_translation_error_m"]>1e-5 or error["maximum_rotation_matrix_error"]>1e-5:
                                raise RuntimeError("Physical fixtures failed to follow measured mounts: "+str(error))
                            fixture_state["fixture_to_world"] = env.tool.fixtures.poses()
                            fixture_state["fixture_pre_step_translation_error_m"] = error["maximum_translation_error_m"]
                            fixture_state["fixture_pre_step_rotation_matrix_error"] = error["maximum_rotation_matrix_error"]
                        recorder.append(*rgb, np.asarray(external.get_data())[...,:3].astype(np.uint8).copy(),
                            time_s=sim_step*args.dt, phase=phase,
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
                            finger_mounts=env.tool.mounts, finger_nodes_left=nodes[0], finger_nodes_right=nodes[1],
                            width_parameter_m=jaw, **fixture_state)
                        env.reward_manager.step(env_idx_list=[0])
                lift = float(object_pose()[0][2])-operation_initial_z
                checkpoints.append(dict(operation_index=operation_index, target_label=target_label,
                                        phase=phase, object_lift_m=lift))
                if phase == "probe_lift" and lift < .005:
                    print("GRASP_REJECTED", target_label, "Object did not follow probe lift", flush=True)
                    rejected = True
                    break
                last_width = width
            if rejected:
                break
        actual_lift = float(object_pose()[0][2])-operation_initial_z
        rewards = env.reward_manager.get_reward()
        success = bool(rewards[0] > .999)
        scores = env.reward_manager.get_score(reward_lst=rewards) if hasattr(env, "get_score") else None
        result = dict(task=task_name, success=success, object_lift_m=actual_lift,
            official_task_manager=True, official_benchmark_evaluation=False,
            embodiment_variant=scene["embodiment_variant"], checkpoints=checkpoints,
            joint_control=args.joint_control, task_rewards=rewards, task_scores=scores,
            task_object_labels=task_labels, executed_plans=executed_plans,
            coupling="prescribed FEM base nodes, no reaction torque feedback", config=cfg.to_dict())
        (out / "result.json").write_text(json.dumps(result, indent=2))
        recorder.close(complete=True, result={k:v for k,v in result.items() if k != "config"})
        print("TASK_RESULT", json.dumps(result), flush=True)
        completed = True
    except BaseException as error:
        import traceback
        traceback.print_exc()
        (out / "run_failure.json").write_text(json.dumps(dict(
            error_type=type(error).__name__, message=str(error)), indent=2))
        raise
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            (out / "run_status.json").write_text(json.dumps(dict(
                completed=completed, mode=args.mode, task=args.task,
                joint_control=args.joint_control), indent=2))
            sys.stdout.flush()
            sys.stderr.flush()
            # Kit's immediate shutdown can swallow an active exception and
            # return 0. A failed batch must remain failed for its caller.
            if not completed:
                os._exit(1)
        # Recording/conversion is flushed above. Skip stage teardown: legacy
        # deformable tensor views can stall Kit's graceful close after export.
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
