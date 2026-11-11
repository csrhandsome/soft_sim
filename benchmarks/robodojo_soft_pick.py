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
sys.path[:0] = [str(ROOT), str(ROOT / "isaacsim")]
import numpy as np
from shared.config import ExperimentConfig
from shared.geometry import matrix_to_quat_wxyz, camera_intrinsics, transform_points
from shared.tool_mount import mounts_from_tool
from app_launcher import create_simulation_app, add_livestream_arguments, configure_visualization
from isaac51_tool import MountedSoftTool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_livestream_arguments(parser)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--mode", choices=("smoke", "pick", "no-contact", "export"), default="pick")
    parser.add_argument("--arm-control", choices=("kinematic", "drive"), default="kinematic")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--primitive-box", action="store_true", help="Use an analytic box for physics diagnostics")
    parser.add_argument("--object-width", type=float, default=.04, help="USD object width along the closing world-Y axis, meters")
    parser.add_argument("--object-usd", default="RoboDojo/Assets/Object/RoboDojo/Clutter/ball/00000/object.usdz")
    parser.add_argument("--dt", type=float, default=.0005)
    parser.add_argument("--jaw-width", type=float, default=.026,
                        help="Legacy shared width parameter; base separation = width + 2*adapter-offset")
    parser.add_argument("--adapter-offset", type=float, default=.008)
    parser.add_argument("--render-every", type=int, default=100)
    parser.add_argument("--solver-iterations", type=int, default=200)
    parser.add_argument("--grasp-plan", help="plan.json from plan_soft_grasp.py; replay its checked joint paths")
    parser.add_argument("--output", default="data/runs/robodojo_soft_pick")
    args = parser.parse_args()
    if not .0001 <= args.dt <= .001 or not 0 <= args.jaw_width <= .08 or args.adapter_offset < 0 or args.render_every < 1 or args.solver_iterations < 1 or not .01 <= args.object_width <= .08:
        parser.error("Invalid dt, width, adapter offset, solver iterations or render stride")
    if args.jaw_width + 2 * args.adapter_offset < .041:
        parser.error("Base separation must exceed 41 mm so the 40 mm finger bases do not overlap")
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
        if args.primitive_box or args.mode != "pick":
            parser.error("--grasp-plan requires --mode pick and its original USD object")
    mount_offset = grasp_plan["scene"]["hand_mount_offset_m"] if grasp_plan else .075
    cfg = ExperimentConfig()
    cfg.gripper.profile = "panda"
    cfg.gripper.base_height_m = 0.
    cfg.gripper.open_jaw_width_m = .08
    cfg.gripper.adapter_offset_m = args.adapter_offset
    cfg.simulation.timestep_s = args.dt
    cfg.simulation.solver_iterations = args.solver_iterations
    cfg.simulation.contact_offset_m = .00025
    cfg.simulation.render = not args.no_cameras
    cfg.simulation.headless = not args.gui
    cfg.camera.width = cfg.camera.height = 256
    configure_visualization(parser, args, cfg.simulation)
    cfg.validate()
    asset = ROOT / "RoboDojo/Assets/Robots/franka/Franka.usd"
    if not asset.is_file():
        raise FileNotFoundError("Run python3 scripts/download_benchmark_assets.py first")
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    app = create_simulation_app(cfg.simulation)
    tool = None
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
        world.scene.add_ground_plane()
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
            obj.AddScaleOp().Set(Gf.Vec3f(.036, .036, .044))
            obj.CreateDisplayColorAttr([Gf.Vec3f(.7, .2, .15)])
            UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
            object_asset = "primitive:box"
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
        if args.mode == "no-contact":
            rb.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(obj_root.GetPrim()).CreateMassAttr(.03)
        mat = UsdShade.Material.Define(stage, "/World/ObjectMaterial")
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr(1.)
        pm.CreateDynamicFrictionAttr(1.)
        UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat, materialPurpose="physics")
        object_view = world.scene.add(SingleRigidPrim(prim_path="/World/Object", name="object"))
        kin = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config("Franka"))
        kin.set_robot_base_pose(base, np.array([1., 0., 0., 0.]))
        q = np.array([0., -.5, 0., -2.2, 0., 1.8, .8])
        hand = np.array([.42, -.12, .44])
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
        else:
            solve(hand)
            initial_width = .08
        hand, measured_r = kin.compute_forward_kinematics("panda_hand", q)
        if args.mode == "export":
            vertices, triangles = [], []
            count = 0
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            for child in Usd.PrimRange(obj_root.GetPrim()):
                if not child.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(child)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                matrix = np.asarray(cache.GetLocalToWorldTransform(child))
                points = (np.c_[points, np.ones(len(points))] @ matrix)[:, :3]
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
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
                initial_object_position=start.tolist(), table_z=table_z,
                initial_q=np.r_[q, .04, .04].tolist(), hand_mount_offset_m=.075,
                adapter_offset_m=args.adapter_offset, finger_mesh=cfg.mesh.stl)
            (out / "scene.json").write_text(json.dumps(metadata, indent=2))
            print("GEOMETRY_EXPORTED", str(out), flush=True)
            return
        tool = MountedSoftTool(stage, cfg, mount_frames(hand, measured_r, initial_width),
            filtered_bodies=[body_paths[name] for name in ("panda_hand", "panda_link7") if name in body_paths])
        task_camera = UsdGeom.Camera.Define(stage, "/World/TaskCamera")
        task_camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
        task_camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.05, -1.15, .92), Gf.Vec3d(.25, 0., .32), Gf.Vec3d(0., 0., 1.)).GetInverse())
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
        if args.mode == "smoke":
            phases = phases[:1]
        elif args.mode == "no-contact":
            phases = phases[:5]
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
            pose, _ = object_view.get_world_pose()
            print("CHECKPOINT", name, "object", pose.tolist(), "hand", hand.tolist(), flush=True)
            if name == "settle":
                initial_z = float(pose[2])
            if not grasp_plan and np.linalg.norm(hand-goal) > .025:
                raise RuntimeError("Arm tracking error after " + name)
            (out / "trajectory.json").write_text(json.dumps(samples, indent=2))
            if cfg.simulation.render and name in ("settle", "hold", "transfer", "settle_place", "verify_hold"):
                rgb, depth = tool.capture()
                np.savez_compressed(out / (name+"_cameras.npz"), rgb=rgb, depth_m=depth,
                    intrinsics=np.stack([camera_intrinsics(cfg.camera, 2.8)]*2), camera_to_world=tool.camera_poses)
                from shared.dataset import write_png
                write_png(out / (name+"_left.png"), rgb[0])
                write_png(out / (name+"_right.png"), rgb[1])
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
        result = dict(mode=args.mode, arm_control=args.arm_control, robot_asset=str(asset),
            object_asset=object_asset, isaacsim_version="5.1.0", max_lift_m=max_lift, final_position=final.tolist(), final_xy_error_m=xy_error,
            target_xy_m=target_xy.tolist(), initial_object_z_m=initial_z,
            final_lift_m=float(final[2]-initial_z), min_held_lift_m=min_held_lift,
            verified_hold_duration_s=hold_duration,
            success=bool(max_lift > .06 and ((min_held_lift is not None and min_held_lift > .06 and xy_error < .035)
                if grasp_plan else (args.mode == "pick" and xy_error < .035 and abs(final[2]-start[2]) < .015))),
            grasp_plan=str(plan_file) if grasp_plan else None,
            task="grasp_and_hold_after_lift" if grasp_plan else "pick_and_place",
            coupling="one-way prescribed FEM base nodes; arm does not receive FEM reaction torque",
            planner=grasp_plan["planner"] if grasp_plan else "Lula IK with clear-table scripted waypoints; no collision-aware trajectory planner in this demo",
            official_benchmark_evaluation=False, config=cfg.to_dict())
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print("RESULT", json.dumps({k:v for k,v in result.items() if k != "config"}), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Closing Kit owns the render graph teardown. Detaching shared
        # annotator graphs manually during Kit shutdown is broken in 5.1.
        app.close()


if __name__ == "__main__":
    main()
