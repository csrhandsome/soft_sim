"""Isaac Sim 5.1: RoboDojo Franka asset + shared FEM fingers and two cameras.

This is an asset/soft-tool integration demo, separate from official benchmark
scores. The stock benchmark managers are not launched. Objects remain dynamic;
only the soft-finger base nodes are prescribed from the measured arm pose.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from shared.config import ExperimentConfig
from shared.geometry import matrix_to_quat_wxyz, camera_intrinsics, transform_points
from shared.tool_mount import mounts_from_tool
from backends.isaac_sim.app_launcher import create_simulation_app, add_livestream_arguments, configure_visualization
from isaac51_tool import MountedSoftTool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_livestream_arguments(parser)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--closeup", action="store_true", help="Frame the object and soft fingers in the external camera")
    parser.add_argument("--mode", choices=("smoke", "pick", "squeeze", "no-contact", "export"), default="pick")
    parser.add_argument("--squeeze-width", type=float, default=.05,
                        help="Closed width for a stationary squeeze/release test using a grasp plan")
    parser.add_argument("--elasticity-damping", type=float,
                        help="Override PhysX elasticity damping (an uncalibrated material parameter)")
    parser.add_argument("--vertex-velocity-damping", type=float,
                        help="Artificial FEM nodal velocity damping in 1/s for quasistatic diagnostics")
    parser.add_argument("--sleep-threshold", type=float, default=0.,
                        help="FEM sleeping speed threshold in m/s (zero disables sleeping)")
    parser.add_argument("--settling-threshold", type=float, default=0.,
                        help="FEM speed threshold for additional settling damping, m/s")
    parser.add_argument("--sleep-damping", type=float,
                        help="Additional FEM settling damping in 1/s (unset uses PhysX default)")
    parser.add_argument("--frozen-render-check", action="store_true",
                        help="Repeat captures at the end of hold without any physics steps")
    parser.add_argument("--max-depenetration-velocity", type=float,
                        help="Limit contact penetration-correction speed for FEM and object, m/s")
    parser.add_argument("--record-collision-nodes", action="store_true",
                        help="Also record the actual FEM collision mesh for contact diagnostics")
    parser.add_argument("--rt-subframes", type=int, default=2,
                        help="Render subframes at the same physical state per RGB capture")
    parser.add_argument("--lighting-samples", type=int, choices=(1, 2, 4, 8),
                        help="RTX direct/reflection samples per pixel; unset keeps renderer defaults")
    parser.add_argument("--local-physics-origin", action="store_true",
                        help="Center the squeeze scene on the object to improve float32 FEM coordinate precision")
    parser.add_argument("--arm-control", choices=("kinematic", "drive"), default="kinematic")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--check-mesh-only", action="store_true", help="Save cooked topology and exit before the episode")
    parser.add_argument("--stop-after-settle", action="store_true", help="Record initialization only, without closing")
    parser.add_argument("--primitive-box", action="store_true", help="Use an analytic box for physics diagnostics")
    parser.add_argument("--box-size", type=float, nargs=3, default=(.036, .036, .044), metavar=("X", "Y", "Z"),
                        help="Rigid diagnostic box dimensions in meters")
    parser.add_argument("--object-mass", type=float, default=.03, help="Dynamic object mass in kilograms")
    parser.add_argument("--object-width", type=float, default=.04, help="USD object width along the closing world-Y axis, meters")
    parser.add_argument("--object-usd", default="RoboDojo/Assets/Object/RoboDojo/Clutter/ball/00000/object.usdz")
    parser.add_argument("--dt", type=float, default=.0005)
    parser.add_argument("--jaw-width", type=float, default=.026,
                        help="Legacy shared width parameter; base separation = width + 2*adapter-offset")
    parser.add_argument("--adapter-offset", type=float, default=.008)
    parser.add_argument("--render-every", type=int, default=100)
    parser.add_argument("--solver-iterations", type=int, default=200)
    parser.add_argument("--simulation-resolution", type=int, default=44,
                        help="PhysX simulation cells along the longest STL axis (44 is about 2 mm)")
    parser.add_argument("--fem-mesh", help="Explicit shared NPZ tetrahedral mesh for both FEM and collision")
    parser.add_argument("--pin-height-m", type=float, default=.004,
                        help="Height of the rigidly prescribed mounting region, meters")
    parser.add_argument("--self-collision", action="store_true", help="Enable soft-finger internal collision")
    parser.add_argument("--surface-max-edge-m", type=float, default=.002,
                        help="Refine display edges and embed them in the FEM; collision geometry is preserved, zero keeps the baseline")
    parser.add_argument("--grasp-plan", help="plan.json from plan_soft_grasp.py; replay its checked joint paths")
    parser.add_argument("--record-episode", action="store_true",
                        help="Save synchronized RGB/state using LeRobot v3 under OUTPUT/recording")
    parser.add_argument("--dataset-root", type=Path,
                        help="Append episodes to a shared LeRobot v3 dataset (same sensor schema)")
    parser.add_argument("--video-fps", type=float, default=20., help="RGB sampling and playback rate in simulation time")
    parser.add_argument("--output", default="data/runs/robodojo_soft_pick")
    args = parser.parse_args()
    if not np.isfinite(args.pin_height_m) or not 0 < args.pin_height_m < .088:
        parser.error("Pin height must be finite and between 0 and 0.088 m")
    if args.simulation_resolution < 2 or not np.isfinite(args.surface_max_edge_m) or args.surface_max_edge_m < 0:
        parser.error("Simulation resolution must be >= 2 and surface edge limit finite and nonnegative")
    if not all(np.isfinite(x) and .01 <= x <= .15 for x in args.box_size) or not np.isfinite(args.object_mass) or args.object_mass <= 0:
        parser.error("Box dimensions must be in [0.01, 0.15] m and object mass must be positive")
    if args.elasticity_damping is not None and (not np.isfinite(args.elasticity_damping) or args.elasticity_damping < 0):
        parser.error("Elasticity damping must be finite and nonnegative")
    if args.vertex_velocity_damping is not None and (not np.isfinite(args.vertex_velocity_damping) or args.vertex_velocity_damping < 0):
        parser.error("Vertex velocity damping must be finite and nonnegative")
    if any(value is not None and (not np.isfinite(value) or value < 0)
           for value in (args.sleep_threshold, args.settling_threshold, args.sleep_damping)):
        parser.error("Sleeping and settling parameters must be finite and nonnegative")
    if args.sleep_threshold > args.settling_threshold:
        parser.error("Settling threshold must be at least the sleeping threshold")
    if args.frozen_render_check and not args.record_episode:
        parser.error("--frozen-render-check requires --record-episode")
    if args.record_collision_nodes and not args.record_episode:
        parser.error("--record-collision-nodes requires --record-episode")
    if args.max_depenetration_velocity is not None and (not np.isfinite(args.max_depenetration_velocity) or args.max_depenetration_velocity <= 0):
        parser.error("Maximum depenetration velocity must be positive and finite")
    if args.rt_subframes < 1:
        parser.error("Render subframes must be positive")
    if args.mode == "squeeze" and not args.grasp_plan:
        parser.error("--mode squeeze requires a plan for the current object")
    if args.local_physics_origin and args.mode != "squeeze":
        parser.error("--local-physics-origin is currently validated only for --mode squeeze")
    if not .0001 <= args.dt <= .001 or not 0 <= args.jaw_width <= .08 or args.adapter_offset < 0 or args.render_every < 1 or args.solver_iterations < 1 or not .01 <= args.object_width <= .08:
        parser.error("Invalid dt, width, adapter offset, solver iterations or render stride")
    if args.jaw_width + 2 * args.adapter_offset < .041:
        parser.error("Base separation must exceed 41 mm so the 40 mm finger bases do not overlap")
    record_stride = 0
    if args.record_episode:
        if args.no_cameras or not np.isfinite(args.video_fps) or args.video_fps <= 0:
            parser.error("--record-episode requires cameras and a positive finite --video-fps")
        interval_steps = 1 / (args.video_fps * args.dt)
        record_stride = round(interval_steps)
        if record_stride < 1 or not np.isclose(interval_steps, record_stride, atol=1e-7, rtol=0):
            parser.error("--video-fps must divide the physics step rate exactly")
        if args.mode == "export":
            parser.error("--record-episode requires a simulated episode, not --mode export")
    grasp_plan = None
    if args.grasp_plan:
        plan_file = ROOT / args.grasp_plan
        grasp_plan = json.loads(plan_file.read_text())
        if grasp_plan.get("status") != "planned" or not grasp_plan.get("collision_checked"):
            parser.error("Expected a collision-checked soft-tool grasp plan")
        args.object_usd = grasp_plan["scene"]["object_asset"]
        args.object_width = grasp_plan["scene"]["object_width_m"]
        args.jaw_width = grasp_plan["candidate"]["closed"]
        args.adapter_offset = grasp_plan["scene"]["adapter_offset_m"]
        if args.primitive_box and not grasp_plan["scene"].get("primitive_box", False):
            parser.error("This plan was generated for a USD object; export and plan the box scene first")
        args.primitive_box = grasp_plan["scene"].get("primitive_box", False)
        args.box_size = grasp_plan["scene"].get("object_size_m", args.box_size)
        args.object_mass = grasp_plan["scene"].get("object_mass_kg", args.object_mass)
        if args.mode not in ("pick", "squeeze"):
            parser.error("--grasp-plan requires --mode pick or squeeze")
        if args.mode == "squeeze":
            if not np.isfinite(args.squeeze_width) or not args.jaw_width <= args.squeeze_width < grasp_plan["candidate"]["opened"]:
                parser.error("Squeeze width must be between the planned closed and open widths")
            args.jaw_width = args.squeeze_width
    if args.primitive_box:
        args.object_width = args.box_size[1]
    mount_offset = grasp_plan["scene"]["hand_mount_offset_m"] if grasp_plan else .075
    cfg = ExperimentConfig()
    cfg.mesh.pin_height_m = args.pin_height_m
    if args.fem_mesh:
        cfg.mesh.tetra_mesh = args.fem_mesh
    if args.elasticity_damping is not None:
        cfg.material.damping = args.elasticity_damping
    cfg.gripper.profile = "panda"
    cfg.gripper.base_height_m = 0.
    cfg.gripper.open_jaw_width_m = .08
    cfg.gripper.adapter_offset_m = args.adapter_offset
    cfg.simulation.timestep_s = args.dt
    cfg.simulation.solver_iterations = args.solver_iterations
    cfg.simulation.deformable_vertex_velocity_damping = args.vertex_velocity_damping
    cfg.simulation.contact_offset_m = .00025
    cfg.simulation.render = not args.no_cameras
    cfg.simulation.headless = not args.gui
    cfg.object.mass_kg = args.object_mass
    cfg.object.fixed = False
    cfg.camera.width = cfg.camera.height = 256
    configure_visualization(parser, args, cfg.simulation)
    cfg.validate()
    asset = ROOT / "RoboDojo/Assets/Robots/franka/Franka.usd"
    if not asset.is_file():
        raise FileNotFoundError("Run python3 scripts/download_benchmark_assets.py first")
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.record_episode and (out / "recording").exists():
        raise FileExistsError(f"Recording already exists: {out / 'recording'}; choose a new --output")
    app = create_simulation_app(cfg.simulation)
    tool = None
    recorder = None
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.robot_motion.motion_generation", True)
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema
        import omni.usd
        import torch
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        world = World(stage_units_in_meters=1., physics_dt=args.dt, rendering_dt=args.dt,
                      backend="torch", device="cuda:0")
        physics = world.get_physics_context()
        physics.enable_gpu_dynamics(True)
        physics.set_broadphase_type("GPU")
        # FEM tensors use world-space float32 positions. Keep the small cage
        # close to zero without changing any relative geometry or material.
        # The robot/object/camera stay authored in their original scene frame;
        # the FEM is authored separately in the translated world frame below.
        origin_offset = -np.array([.42, -.12, .12 + args.box_size[2] / 2]) if args.local_physics_origin else np.zeros(3)
        if args.local_physics_origin:
            scene_root = UsdGeom.Xform.Define(stage, "/World")
            scene_root.AddTranslateOp().Set(Gf.Vec3d(*origin_offset))
        # GroundPlane takes a world-space height, unlike the authored table
        # and object below; its default would leave it above the shifted table.
        world.scene.add_ground_plane(z_position=float(origin_offset[2]))
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(1500.)
        table_z = .12
        base = np.array([0., 0., table_z])
        hand_r = np.diag([1., -1., -1.])
        tool_in_hand = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        robot_prim = stage.DefinePrim("/World/Franka")
        robot_prim.GetReferences().AddReference(str(asset))
        root_xform = UsdGeom.Xformable(robot_prim)
        root_xform.ClearXformOpOrder()
        root_xform.AddTranslateOp().Set(Gf.Vec3d(*base))
        # Instance proxies must be expanded before overriding their colliders.
        while True:
            instances = [p for p in Usd.PrimRange(robot_prim) if p.IsInstance()]
            if not instances:
                break
            for p in instances:
                p.SetInstanceable(False)
        body_paths = {}
        for p in Usd.PrimRange(robot_prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                body_paths[p.GetName()] = str(p.GetPath())
                PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
            if p.GetName() in ("panda_leftfinger", "panda_rightfinger"):
                UsdGeom.Imageable(p).MakeInvisible()
                for child in Usd.PrimRange(p):
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        robot = world.scene.add(SingleArticulation(prim_path="/World/Franka", name="benchmark_franka"))
        table = UsdGeom.Cube.Define(stage, "/World/Table")
        table.CreateSizeAttr(1.)
        table.AddTranslateOp().Set(Gf.Vec3d(.35, 0., table_z - .025))
        table.AddScaleOp().Set(Gf.Vec3f(1.05, .8, .05))
        table.CreateDisplayColorAttr([Gf.Vec3f(.28, .21, .15)])
        UsdPhysics.CollisionAPI.Apply(table.GetPrim())
        table_mdl = ROOT / "RoboDojo/Assets/Material/material_0122/Mahogany_Planks.mdl"
        if table_mdl.is_file():
            from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
            create_mdl_material(str(table_mdl), "Mahogany_Planks", "/World/TableMaterial")
            UsdShade.MaterialBindingAPI.Apply(table.GetPrim()).Bind(UsdShade.Material.Get(stage, "/World/TableMaterial"))
        obj_root = UsdGeom.Xform.Define(stage, "/World/Object")
        start = np.array([.42, -.12, table_z + .0225])
        obj_root.AddTranslateOp().Set(Gf.Vec3d(*start))
        if args.primitive_box:
            obj = UsdGeom.Cube.Define(stage, "/World/Object/Shape")
            obj.CreateSizeAttr(1.)
            obj.AddScaleOp().Set(Gf.Vec3f(*args.box_size))
            obj.CreateDisplayColorAttr([Gf.Vec3f(.7, .2, .15)])
            UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
            object_asset = "primitive:box"
            start[2] = table_z + args.box_size[2] / 2 + .0005
            UsdGeom.Xformable(obj_root).GetOrderedXformOps()[0].Set(Gf.Vec3d(*start))
        else:
            object_path = ROOT / args.object_usd
            if not object_path.is_file():
                raise FileNotFoundError(object_path)
            visual = stage.DefinePrim("/World/Object/Shape")
            visual.GetReferences().AddReference(str(object_path))
            while True:
                instances = [p for p in Usd.PrimRange(visual) if p.IsInstance()]
                if not instances:
                    break
                for p in instances:
                    p.SetInstanceable(False)
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeRelativeBound(visual, obj_root.GetPrim()).ComputeAlignedRange()
            center = np.asarray(bounds.GetMidpoint())
            size = np.asarray(bounds.GetSize())
            if not np.isfinite(size).all() or max(size) <= 0:
                raise RuntimeError("Object USD has no usable bounds")
            if size[1] <= 0:
                raise RuntimeError("Object has zero width along the jaw closing axis")
            scale = args.object_width / size[1]
            print("OBJECT_SIZE_M", (size * scale).tolist(), flush=True)
            xf = UsdGeom.Xformable(visual)
            original = np.asarray(xf.GetLocalTransformation())
            normalize = np.eye(4)
            normalize[:3, :3] *= scale
            normalize[3, :3] = -center * scale
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(Gf.Matrix4d(*(original @ normalize).ravel().tolist()))
            start[2] = table_z + size[2] * scale / 2 + .0005
            UsdGeom.Xformable(obj_root).GetOrderedXformOps()[0].Set(Gf.Vec3d(*start))
            for child in Usd.PrimRange(visual):
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    child.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if child.IsA(UsdPhysics.Joint):
                    child.SetActive(False)
                if child.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(child)
                    UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr("convexHull")
            obj = UsdGeom.Xformable(visual)
            object_asset = str(object_path)
        for child in Usd.PrimRange(obj_root.GetPrim()):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(args.mode != "no-contact")
                collision = PhysxSchema.PhysxCollisionAPI.Apply(child)
                collision.CreateContactOffsetAttr(cfg.simulation.contact_offset_m)
                collision.CreateRestOffsetAttr(0.)
        rb = UsdPhysics.RigidBodyAPI.Apply(obj_root.GetPrim())
        if args.max_depenetration_velocity is not None:
            PhysxSchema.PhysxRigidBodyAPI.Apply(obj_root.GetPrim()).CreateMaxDepenetrationVelocityAttr(args.max_depenetration_velocity)
        if args.mode == "no-contact":
            rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(obj_root.GetPrim()).CreateMassAttr(args.object_mass)
        mat = UsdShade.Material.Define(stage, "/World/ObjectMaterial")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(1.)
        pm.CreateDynamicFrictionAttr(1.)
        UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat, materialPurpose="physics")
        object_view = world.scene.add(SingleRigidPrim(prim_path="/World/Object", name="object"))
        # start and FK below are world-space; authored object transforms above
        # remain in the source scene frame inherited from /World.
        start = start + origin_offset
        kin = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config("Franka"))
        kin.set_robot_base_pose(base + origin_offset, np.array([1., 0., 0., 0.]))
        q = np.array([0., -.5, 0., -2.2, 0., 1.8, .8])
        hand = np.array([.42, -.12, .44]) + origin_offset
        def solve(target):
            nonlocal q
            q, ok = kin.compute_inverse_kinematics("panda_hand", np.asarray(target),
                matrix_to_quat_wxyz(hand_r), warm_start=q, position_tolerance=.0001, orientation_tolerance=.005)
            if not ok:
                raise RuntimeError("Arm IK failed: " + str(target))
            return q
        def mount_frames(position, rotation, width):
            matrix = np.eye(4)
            matrix[:3, :3] = rotation @ tool_in_hand
            matrix[:3, 3] = position + rotation @ np.array([0., 0., mount_offset])
            return mounts_from_tool(cfg.gripper, matrix, width)
        if grasp_plan:
            q = np.array(grasp_plan["initial_q"][:7])
            initial_width = float(sum(grasp_plan["initial_q"][-2:]))
            if args.mode == "squeeze":
                with np.load(plan_file.parent / grasp_plan["trajectory_file"]) as planned_paths:
                    q = planned_paths["approach"][-1].copy()
                initial_width = grasp_plan["candidate"]["opened"]
        else:
            solve(hand)
            initial_width = .08
        hand, measured_r = kin.compute_forward_kinematics("panda_hand", q)
        if args.mode == "export":
            vertices, triangles = [], []
            count = 0
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            for child in Usd.PrimRange(obj_root.GetPrim()):
                if not child.IsA(UsdGeom.Mesh) and not child.IsA(UsdGeom.Cube):
                    continue
                if child.IsA(UsdGeom.Cube):
                    import trimesh
                    cube = trimesh.creation.box(extents=[UsdGeom.Cube(child).GetSizeAttr().Get()] * 3)
                    points = np.asarray(cube.vertices)
                    counts = np.full(len(cube.faces), 3)
                    indices = np.asarray(cube.faces).ravel()
                else:
                    mesh = UsdGeom.Mesh(child)
                    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
                    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
                matrix = np.asarray(cache.GetLocalToWorldTransform(child))
                points = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
                cursor = 0
                for length in counts:
                    face = indices[cursor:cursor+length]
                    for j in range(1, length-1):
                        triangles.append([count+face[0], count+face[j], count+face[j+1]])
                    cursor += length
                vertices.append(points)
                count += len(points)
            np.savez_compressed(out / "object_geometry.npz", vertices=np.concatenate(vertices), faces=np.asarray(triangles))
            metadata = dict(object_asset=object_asset, object_width_m=args.object_width,
                primitive_box=args.primitive_box, object_size_m=list(args.box_size) if args.primitive_box else (size * scale).tolist(),
                object_mass_kg=args.object_mass,
                initial_object_position=start.tolist(), table_z=table_z,
                initial_q=np.r_[q, .04, .04].tolist(), hand_mount_offset_m=.075,
                adapter_offset_m=args.adapter_offset, finger_mesh=cfg.mesh.stl)
            (out / "scene.json").write_text(json.dumps(metadata, indent=2))
            print("GEOMETRY_EXPORTED", str(out), flush=True)
            return
        tool = MountedSoftTool(stage, cfg, mount_frames(hand, measured_r, initial_width),
            root="/SoftTool" if args.local_physics_origin else "/World/SoftTool",
            sleep_threshold=args.sleep_threshold, settling_threshold=args.settling_threshold,
            sleep_damping=args.sleep_damping,
            max_depenetration_velocity=args.max_depenetration_velocity,
            rt_subframes=args.rt_subframes, lighting_samples_per_pixel=args.lighting_samples,
            simulation_resolution=args.simulation_resolution, surface_max_edge_m=args.surface_max_edge_m,
            explicit_fem_mesh=bool(args.fem_mesh), self_collision=args.self_collision,
            filtered_bodies=[body_paths[name] for name in ("panda_hand", "panda_link7") if name in body_paths])
        task_camera = UsdGeom.Camera.Define(stage, "/World/TaskCamera")
        task_camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
        eye, target = ((.78, -.48, .48), (.42, -.12, .23)) if args.closeup else ((1.05, -1.15, .92), (.25, 0., .32))
        task_camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0., 0., 1.)).GetInverse())
        if args.record_episode:
            import omni.replicator.core as rep
            from episode_recording import EpisodeRecorder
            external_size = (640, 480)
            external_focal = 35. if args.closeup else 18.
            task_camera.CreateFocalLengthAttr(external_focal)
            task_camera.CreateHorizontalApertureAttr(24.)
            task_camera.CreateVerticalApertureAttr(18.)
            external_product = rep.create.render_product(str(task_camera.GetPath()), external_size)
            external_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            external_rgb.attach([external_product])
            external_pose = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(task_camera.GetPrim())).T.copy()
            external_pose[:3, :3] = external_pose[:3, :3] @ np.diag([1., -1., -1.])
            calibration = dict(
                finger_intrinsics=np.stack([camera_intrinsics(cfg.camera, 2.8)] * 2).tolist(),
                external_intrinsics=[[640. * external_focal / 24., 0., 320.],
                                     [0., 480. * external_focal / 18., 240.], [0., 0., 1.]],
                external_camera_to_world=external_pose.tolist(),
                finger_resolution=[cfg.camera.width, cfg.camera.height],
                finger_projection=tool.camera_projection,
                rendering=tool.render_settings,
                deformable_mesh=tool.mesh_settings,
                deformable_dynamics=dict(material=cfg.to_dict()["material"],
                    vertex_velocity_damping_override_s_inv=cfg.simulation.deformable_vertex_velocity_damping,
                    solver_position_iterations=args.solver_iterations,
                    sleep_threshold_m_s=args.sleep_threshold, settling_threshold_m_s=args.settling_threshold,
                    sleep_damping_override_s_inv=args.sleep_damping,
                    max_depenetration_velocity_override_m_s=args.max_depenetration_velocity,
                    damping_calibrated=False),
                world_origin_offset_m=origin_offset.tolist(),
                external_resolution=list(external_size), physics_dt_s=args.dt,
                collision_nodes_recorded=args.record_collision_nodes,
                rigid_object_max_depenetration_velocity_override_m_s=args.max_depenetration_velocity,
                record_stride=record_stride, grasp_plan=str(plan_file) if grasp_plan else None)
            recorder = EpisodeRecorder(out / "recording", args.video_fps, calibration,
                dataset_root=args.dataset_root,
                task="squeeze and release" if args.mode == "squeeze" else "pick and lift the object")
        from omni.kit.viewport.utility import get_active_viewport
        viewport = get_active_viewport()
        if viewport:
            viewport.camera_path = "/World/TaskCamera"
        world.reset()
        q9 = torch.as_tensor(np.r_[q, initial_width/2, initial_width/2], dtype=torch.float32, device="cuda:0")
        robot.set_joint_positions(q9)
        robot.set_joint_velocities(torch.zeros_like(q9))
        robot.get_articulation_controller().set_gains(kps=torch.full((9,), 2000., device="cuda:0"),
                                                     kds=torch.full((9,), 150., device="cuda:0"))
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q9))
        tool.initialize()
        np.savez_compressed(out / "finger_reference.npz", **tool.reference_state)
        (out / "mesh_topology.json").write_text(json.dumps(tool.mesh_topology, indent=2))
        if not all(row["valid"] for row in tool.mesh_topology):
            raise RuntimeError("Cooked FEM contains nodes disconnected from the pinned base; see mesh_topology.json")
        if args.check_mesh_only:
            return
        if recorder is not None:
            np.savez_compressed(out / "recording/finger_reference.npz", **tool.reference_state)
        pick = start + [0., 0., mount_offset + .073]
        # Keep the complete 88.18 mm STL tip above the table. Short objects
        # cannot use the same center-based grasp height as tall ones.
        pick[2] = max(pick[2], table_z + mount_offset + .088180001 + .002)
        place = pick + [0., .28, 0.]
        above_pick, above_place = pick + [0., 0., .13], place + [0., 0., .13]
        phases = [("settle", .3, hand.copy(), .08), ("approach", 1.5, above_pick, .08),
            ("descend", 1.5, pick, .08), ("close", 2., pick, args.jaw_width),
            ("hold", .5, pick, args.jaw_width), ("lift", 2., above_pick, args.jaw_width),
            ("transfer", 2., above_place, args.jaw_width), ("lower", 2., place, args.jaw_width),
            ("release", 1.5, place, .08), ("retreat", 1.5, above_place, .08), ("settle_place", .5, above_place, .08)]
        if grasp_plan:
            joint_paths = np.load(plan_file.parent / grasp_plan["trajectory_file"])
            opened, closed = grasp_plan["candidate"]["opened"], grasp_plan["candidate"]["closed"]
            def still(value):
                return np.repeat(np.asarray(value)[None], 2, axis=0)
            pre, approach, probe, lift = (joint_paths[name] for name in ("pregrasp", "approach", "probe", "lift"))
            phases = [("settle", .3, still(q), opened),
                ("pregrasp", max(.5,(len(pre)-1)*grasp_plan["sample_dt_s"]), pre, opened),
                ("approach", max(1.,(len(approach)-1)*grasp_plan["sample_dt_s"]), approach, opened),
                ("close", 2., still(approach[-1]), closed),
                ("hold", 1., still(approach[-1]), closed),
                ("probe_lift", 1.5, probe, closed),
                ("lift", max(2.,(len(lift)-1)*grasp_plan["sample_dt_s"]), lift, closed),
                ("verify_hold", 1., still(lift[-1]), closed)]
            if args.mode == "squeeze":
                # Initialize at the checked, open final approach pose. Keep
                # the arm stationary to isolate contact deformation and decay.
                phases = [("settle", 1., still(q), opened),
                          ("close", 3., still(q), args.squeeze_width),
                          ("hold", 2., still(q), args.squeeze_width),
                          ("release", 3., still(q), opened),
                          ("settle_release", 1., still(q), opened)]
        if args.mode == "smoke":
            phases = phases[:1]
        elif args.mode == "no-contact":
            phases = phases[:5]
        if args.stop_after_settle:
            phases = phases[:1]
        samples, max_lift, initial_z, sim_step = [], 0., None, 0
        last_pos, last_width = hand.copy(), initial_width
        for name, duration, goal, width in phases:
            print("PHASE", name, flush=True)
            for i in range(round(duration / args.dt)):
                if not app.is_running():
                    raise RuntimeError("Simulator closed during task")
                u = (i + 1) / round(duration / args.dt)
                jaw_u = u*u*(3-2*u)
                jaw = last_width + jaw_u * (width-last_width)
                if grasp_plan:
                    location = min(u*(len(goal)-1), len(goal)-1)
                    lower = min(int(location), len(goal)-2)
                    fraction = location-lower
                    q = (1-fraction)*goal[lower]+fraction*goal[lower+1]
                    q9 = torch.as_tensor(np.r_[q, jaw/2, jaw/2], dtype=torch.float32, device="cuda:0")
                    robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q9))
                else:
                    target = last_pos + jaw_u * (goal-last_pos)
                    if sim_step % 20 == 0:
                        solve(target)
                        q9 = torch.as_tensor(np.r_[q, jaw/2, jaw/2], dtype=torch.float32, device="cuda:0")
                        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q9))
                if args.arm_control == "kinematic":
                    robot.set_joint_positions(q9)
                    robot.set_joint_velocities(torch.zeros_like(q9))
                actual_q = robot.get_joint_positions().cpu().numpy()
                hand, measured_r = kin.compute_forward_kinematics("panda_hand", actual_q[:7])
                tool.follow(mount_frames(hand, measured_r, float(np.clip(actual_q[-2:].sum(), 0., .08))))
                world.step(render=False)
                sim_step += 1
                if sim_step % args.render_every == 0:
                    tool.sync_render_surface()
                    world.render()
                    points = tool.positions()
                    if any(not np.isfinite(p).all() or np.max(np.abs(p)) > 5 for p in points):
                        raise RuntimeError("Unstable soft-finger physics")
                    pose, _ = object_view.get_world_pose()
                    pose = pose.cpu().numpy()
                    if initial_z is not None:
                        max_lift = max(max_lift, float(pose[2] - initial_z))
                    rest = [transform_points(nodes, m[:3, 3], m[:3, :3]) for nodes, m in zip(tool.local_nodes, tool.mounts)]
                    samples.append(dict(time_s=sim_step * args.dt, phase=name, object_position=pose.tolist(),
                        hand_position=hand.tolist(), width_parameter_m=jaw, base_separation_m=jaw+2*args.adapter_offset,
                        max_displacement_mm=float(max(np.linalg.norm(p-r, axis=-1).max() for p,r in zip(points,rest))*1000)))
                if recorder is not None and sim_step % record_stride == 0:
                    # Flush the current physics state before the zero-time
                    # Replicator step renders all three products.
                    # Read the external annotator before any further physics step.
                    rgb, depth = tool.capture(world)
                    external = np.asarray(external_rgb.get_data())[..., :3].astype(np.uint8).copy()
                    object_position, object_orientation = object_view.get_world_pose()
                    recorded_nodes = tool.positions()
                    collision_state = {}
                    if args.record_collision_nodes:
                        collision_nodes = tool.collision_positions()
                        collision_state = dict(finger_collision_nodes_left=collision_nodes[0],
                                               finger_collision_nodes_right=collision_nodes[1])
                    recorder.append(rgb[0], rgb[1], external, time_s=sim_step * args.dt, phase=name,
                        joint_positions=robot.get_joint_positions().cpu().numpy(),
                        joint_velocities=robot.get_joint_velocities().cpu().numpy(),
                        commanded_joint_positions=q9.cpu().numpy(),
                        object_position=object_position.cpu().numpy(),
                        object_orientation_wxyz=object_orientation.cpu().numpy(),
                        width_parameter_m=jaw, finger_camera_to_world=tool.camera_poses,
                        finger_depth_m=depth,
                        finger_nodes_left=recorded_nodes[0], finger_nodes_right=recorded_nodes[1],
                        finger_mounts=tool.mounts, **collision_state)
            pose, _ = object_view.get_world_pose()
            print("CHECKPOINT", name, "object", pose.tolist(), "hand", hand.tolist(), flush=True)
            # A cheap mechanics checkpoint also works without RGB cameras,
            # enabling isolated clamp/mesh comparisons before a full recording.
            checkpoint_nodes = tool.positions()
            if any(not np.isfinite(nodes).all() for nodes in checkpoint_nodes):
                raise RuntimeError("Non-finite FEM state at phase " + name)
            np.savez_compressed(out / (name + '_fem.npz'),
                time_s=sim_step * args.dt, finger_mounts=tool.mounts,
                finger_nodes_left=checkpoint_nodes[0], finger_nodes_right=checkpoint_nodes[1],
                object_position=pose.cpu().numpy())
            if name == "settle":
                initial_z = float(pose[2])
            if not grasp_plan and np.linalg.norm(hand-goal) > .025:
                raise RuntimeError("Arm tracking error after " + name)
            (out / "trajectory.json").write_text(json.dumps(samples, indent=2))
            if cfg.simulation.render and name in ("settle", "hold", "transfer", "settle_place", "verify_hold"):
                rgb, depth = tool.capture(world)
                np.savez_compressed(out / (name+"_cameras.npz"), rgb=rgb, depth_m=depth,
                    intrinsics=np.stack([camera_intrinsics(cfg.camera, 2.8)]*2), camera_to_world=tool.camera_poses)
                from shared.dataset import write_png
                write_png(out / (name+"_left.png"), rgb[0])
                write_png(out / (name+"_right.png"), rgb[1])
            if args.frozen_render_check and name in ("hold", "verify_hold"):
                from benchmarks.check_frozen_render import probe
                def frozen_capture():
                    images, _ = tool.capture(world)
                    return dict(left_rgb=images[0], right_rgb=images[1],
                                external_rgb=np.asarray(external_rgb.get_data())[..., :3].astype(np.uint8).copy())
                def frozen_state():
                    pos, rot = object_view.get_world_pose()
                    nodes = tool.positions()
                    return dict(time_s=np.asarray(world.current_time),
                        finger_nodes_left=nodes[0], finger_nodes_right=nodes[1],
                        finger_mounts=tool.mounts.copy(), finger_camera_to_world=tool.camera_poses.copy(),
                        joint_positions=robot.get_joint_positions().cpu().numpy().copy(),
                        object_position=pos.cpu().numpy().copy(), object_orientation=rot.cpu().numpy().copy())
                report = probe(frozen_capture, frozen_state, out / (name + "_frozen_render"))
                print("FROZEN_RENDER", json.dumps(report), flush=True)
            last_pos, last_width = goal.copy(), width
            if grasp_plan and name == "probe_lift" and float(pose[2])-initial_z < .005:
                print("GRASP_REJECTED", "Object did not follow the 12 mm probe lift", flush=True)
                break
        final, _ = object_view.get_world_pose()
        final = final.cpu().numpy()
        max_lift = max(max_lift, float(final[2]-initial_z))
        target_xy = start[:2] if grasp_plan else (start+[0., .28, 0.])[:2]
        xy_error = float(np.linalg.norm(final[:2] - target_xy))
        held = [sample for sample in samples if sample["phase"] == "verify_hold"]
        min_held_lift = min(sample["object_position"][2]-initial_z for sample in held) if held else None
        hold_duration = next((duration for phase,duration,_,_ in phases if phase=="verify_hold"),0.) if held else 0.
        squeeze = args.mode == "squeeze"
        result = dict(mode=args.mode, arm_control=args.arm_control, robot_asset=str(asset),
            world_origin_offset_m=origin_offset.tolist(),
            fem_settling=dict(sleep_threshold_m_s=args.sleep_threshold,
                settling_threshold_m_s=args.settling_threshold, sleep_damping_override_s_inv=args.sleep_damping),
            contact_solver=dict(max_depenetration_velocity_override_m_s=args.max_depenetration_velocity,
                                applies_to=["soft_fingers", "rigid_object"]),
            deformable_mesh=tool.mesh_settings,
            object_asset=object_asset, object_mass_kg=args.object_mass,
            object_size_m=list(args.box_size) if args.primitive_box else None,
            isaacsim_version="5.1.0", max_lift_m=max_lift, final_position=final.tolist(), final_xy_error_m=xy_error,
            target_xy_m=target_xy.tolist(), initial_object_z_m=initial_z,
            final_lift_m=float(final[2]-initial_z), min_held_lift_m=min_held_lift,
            verified_hold_duration_s=hold_duration,
            success=None if squeeze else bool(max_lift > .06 and ((min_held_lift is not None and min_held_lift > .06 and xy_error < .035)
                if grasp_plan else (args.mode == "pick" and xy_error < .035 and abs(final[2]-start[2]) < .015))),
            grasp_plan=str(plan_file) if grasp_plan else None,
            task="initialization_check" if args.stop_after_settle else ("stationary_squeeze_release" if squeeze else ("grasp_and_hold_after_lift" if grasp_plan else "pick_and_place")),
            coupling="one-way prescribed FEM base nodes; arm does not receive FEM reaction torque",
            planner=grasp_plan["planner"] if grasp_plan else "Lula IK with clear-table scripted waypoints; no collision-aware trajectory planner in this demo",
            official_benchmark_evaluation=False, config=cfg.to_dict())
        (out / "result.json").write_text(json.dumps(result, indent=2))
        if recorder is not None:
            recorder.close(complete=True, result={k: v for k, v in result.items() if k != "config"})
            print("LEROBOT_DATASET", str(recorder.root), "frames", len(recorder.rows), flush=True)
        print("RESULT", json.dumps({k:v for k,v in result.items() if k != "config"}), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if recorder is not None:
            recorder.close()
        # Closing Kit owns the render graph teardown. Detaching shared
        # annotator graphs manually during Kit shutdown is broken in 5.1.
        app.close()


if __name__ == "__main__":
    main()
