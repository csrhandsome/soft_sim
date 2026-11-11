"""Franka arm + existing FEM fingers: contact-driven pick and place, with WebRTC."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
import numpy as np

from shared.config import ExperimentConfig
from shared.experiment import EpisodeSpec, GripperState
from shared.geometry import finger_pose, transform_points, matrix_to_quat_wxyz
from soft_gripper_env import IsaacSimGripperEnv
from app_launcher import add_livestream_arguments, configure_visualization

ROOT = Path(__file__).resolve().parents[1]
TABLE_Z = .12
BASE = np.array([0., 0., TABLE_Z])
HAND_R = np.diag([1., -1., -1.])
# The Panda jaws move along hand Y; the existing soft fingers close along X.
TOOL_IN_HAND = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
MOUNT_Z = .045
GRASP_Z = .073


class FrankaSoftEnv(IsaacSimGripperEnv):
    def __init__(self, config, robot_usd=None):
        self.robot_usd = robot_usd
        self.hand_position = np.array([.42, -.12, .44])
        self.hand_rotation = HAND_R.copy()
        self.robot = None
        super().__init__(config)
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.robot_motion.motion_generation', True)
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        self.kin = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config('Franka'))
        self.kin.set_robot_base_pose(BASE, np.array([1., 0., 0., 0.]))
        self.q = np.array([0., -.5, 0., -2.2, 0., 1.8, .8])

    def finger_pose(self, side):
        local, rotation = finger_pose(self.config.gripper, side, self.state.angles_deg[side],
                                      self.state.base_separation_m)
        local[2] = MOUNT_Z
        tool_rotation = self.hand_rotation @ TOOL_IN_HAND
        return self.hand_position + tool_rotation @ local, tool_rotation @ rotation

    def solve(self, position):
        q, ok = self.kin.compute_inverse_kinematics('panda_hand', np.asarray(position),
                   matrix_to_quat_wxyz(HAND_R), warm_start=self.q,
                   position_tolerance=.0005, orientation_tolerance=.01)
        if not ok:
            raise RuntimeError(f'Franka IK failed at {position}')
        self.q = q
        return q

    def build_scene(self, episode):
        self.q = self.solve(self.hand_position)
        self.hand_position, self.hand_rotation = self.kin.compute_forward_kinematics('panda_hand', self.q)
        super().build_scene(episode)
        self.robot.set_dof_positions(np.r_[self.q, 0., 0.][None])
        self.robot.set_dof_velocities(np.zeros((1, 9)))
        self.robot.set_dof_gains(stiffnesses=1000., dampings=80., dof_indices=list(range(7)))
        self.robot.set_dof_position_targets(np.r_[self.q, 0., 0.][None])
        self.sync_hand()
        self.set_gripper_state(self.state)
        from render_surface import create_surfaces
        create_surfaces(self)
        self.set_camera()

    def _build_object(self, episode):
        super()._build_object(episode)
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from isaacsim.core.experimental.prims import Articulation
        from assets import add_usd_reference
        from presentation import material, box, bind
        prim = add_usd_reference(self.stage, self.robot_usd or 'panda', '/World/Franka')
        # Replace the factory fingertips visually and physically, keeping its hand.
        for name in ('panda_leftfinger', 'panda_rightfinger'):
            root = self.stage.GetPrimAtPath('/World/Franka/' + name)
            if root:
                if root.IsInstance():
                    root.SetInstanceable(False)
                UsdGeom.Imageable(root).MakeInvisible()
                for child in Usd.PrimRange(root):
                    if child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        root_xform = UsdGeom.Xformable(prim)
        root_xform.ClearXformOpOrder()
        root_xform.AddTranslateOp().Set(Gf.Vec3d(*BASE))
        self.robot = Articulation('/World/Franka')
        # Attached FEM roots must not collide with their own mounting hand.
        from pxr import Sdf
        for name in ('left', 'right'):
            UsdPhysics.FilteredPairsAPI.Apply(self.stage.GetPrimAtPath('/World/Finger_' + name)).CreateFilteredPairsRel().SetTargets([
                Sdf.Path('/World/Franka/panda_hand'), Sdf.Path('/World/Franka/panda_link7')])
        # Adequate drive authority for a slow tabletop trajectory.
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI.Get(child, 'angular')
                if drive:
                    drive.CreateStiffnessAttr(2000.)
                    drive.CreateDampingAttr(150.)
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(child).CreateDisableGravityAttr(True)
        wood = material(self.stage, 'Table', (.27, .19, .12), .55)
        metal = material(self.stage, 'Adapter', (.18, .21, .24), .3, .7)
        box(self.stage, 'Desktop', (.35, 0., TABLE_Z - .025), (1.05, .8, .05), wood, True)
        target = material(self.stage, 'Target', (.08, .4, .18), .7)
        # Thin marker has no collision: the same table supports pick and place.
        box(self.stage, 'PlaceZone', (.42, .16, TABLE_Z + .0001), (.10, .10, .0001), target)
        self.mount_ops = []
        for side, name in enumerate(('left', 'right')):
            root = UsdGeom.Xform.Define(self.stage, '/World/Mount_' + name)
            self.mount_ops.append(root.AddTransformOp())
            clamp = UsdGeom.Cube.Define(self.stage, str(root.GetPath()) + '/Clamp')
            clamp.CreateSizeAttr(1.)
            clamp.AddTranslateOp().Set(Gf.Vec3d(0., 0., -.012))
            clamp.AddScaleOp().Set(Gf.Vec3f(.040, .040, .024))
            bind(clamp.GetPrim(), metal)
            bind(self.stage.GetPrimAtPath('/World/Finger_' + name),
                 material(self.stage, 'TPU_' + name, (.10, .55, .60) if side == 0 else (.85, .35, .10)))
        # Sliding supports and bridge remain one visible end effector assembly.
        self.bridge = UsdGeom.Xform.Define(self.stage, '/World/ToolBridge').AddTransformOp()
        bridge = UsdGeom.Cube.Define(self.stage, '/World/ToolBridge/Body')
        bridge.CreateSizeAttr(1.)
        bridge.AddScaleOp().Set(Gf.Vec3f(.15, .044, .018))
        bind(bridge.GetPrim(), metal)

    def sync_hand(self):
        self.hand_position, self.hand_rotation = self.kin.compute_forward_kinematics(
            'panda_hand', self.robot.get_dof_positions().numpy()[0, :7])

    def set_gripper_state(self, state):
        super().set_gripper_state(state)
        if hasattr(self, 'bridge') and self.bridge:
            from pxr import Gf
            matrix = np.eye(4)
            matrix[:3, :3] = self.hand_rotation @ TOOL_IN_HAND
            matrix[:3, 3] = self.hand_position + self.hand_rotation @ np.array([0., 0., .014])
            self.bridge.Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))

    def set_camera(self):
        from pxr import Gf, UsdGeom
        from omni.kit.viewport.utility import get_active_viewport
        camera = UsdGeom.Camera.Define(self.stage, '/World/TaskCamera')
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20.))
        camera.CreateFocalLengthAttr(24.)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.05, -1.15, .92), Gf.Vec3d(.25, 0., .32), Gf.Vec3d(0, 0, 1)).GetInverse())
        viewport = get_active_viewport()
        if viewport:
            viewport.camera_path = str(camera.GetPath())

    def tick(self, target, width, i):
        if i % 20 == 0:
            self.robot.set_dof_position_targets(np.r_[self.solve(target), 0., 0.][None])
        self.sync_hand()
        self.state = GripperState(np.zeros(2), width, width + 2 * self.config.gripper.adapter_offset_m, 0.)
        self.set_gripper_state(self.state)
        self.world.step(render=False)
        self.step_index += 1
        self.time_s = self.step_index * self.config.simulation.timestep_s


def run_episode(env, seed, out, args):
    cfg = env.config
    episode = EpisodeSpec.sample(cfg, seed)
    episode.object_shape = args.object
    episode.object_radius_m = .018
    episode.object_height_m = .044
    halfheight = .018 if args.object == 'sphere' else .022
    episode.object_position_m = (.42, -.12, TABLE_Z + halfheight + .0005)
    episode.object_rpy_deg = (0., 0., 0.)
    episode.mount_angles_deg = (0., 0.)
    episode.final_jaw_width_m = args.jaw_width
    env.step_index = 0
    env.time_s = 0.
    env.episode = episode
    env.state = GripperState(np.zeros(2), .08, .12, 0.)
    env.hand_position = np.array([.42, -.12, .44])
    env.build_scene(episode)
    from render_surface import update_surfaces
    import omni.ui as ui
    window = ui.Window('Franka soft-finger pick and place', width=360, height=160)
    window.position_x = 900
    window.position_y = 60
    state = {'pause': False, 'view': 'overview'}
    with window.frame:
        with ui.VStack(spacing=5):
            label = ui.Label('Preparing', height=28)
            ui.Label('Franka + TPU fingers | physical free object', height=22)
            ui.Button('Pause / resume', height=26, clicked_fn=lambda: state.update(pause=not state['pause']))
            ui.Button('Overview / fingers', height=26, clicked_fn=lambda: state.update(view='fingers' if state['view']=='overview' else 'overview'))
    start = np.array(episode.object_position_m)
    pick = start + np.array([0., 0., MOUNT_Z + GRASP_Z])
    place = pick + np.array([0., .28, 0.])
    above_pick = pick + [0., 0., .13]
    above_place = place + [0., 0., .13]
    phases = [
        ('settle', .5, env.hand_position.copy(), .08),
        ('approach', 2., above_pick, .08),
        ('descend', 2., pick, .08),
        ('close', 2., pick, args.jaw_width),
        ('grip_hold', .6, pick, args.jaw_width),
        ('lift', 2.5, above_pick, args.jaw_width),
        ('transfer', 3., above_place, args.jaw_width),
        ('lower', 2.5, place, args.jaw_width),
        ('release', 1.5, place, .08),
        ('retreat', 2., above_place, .08),
        ('settle_place', 1., above_place, .08),
    ]
    path = out / f'episode_{seed:04d}'
    path.mkdir(parents=True, exist_ok=True)
    samples = []
    initial_z = None
    lifted = False
    lift_delta = 0.
    previous, previous_width = env.hand_position.copy(), .08
    step = 0
    if cfg.simulation.livestream and args.start_delay:
        print(f'Ready for WebRTC; starting in {args.start_delay}s', flush=True)
        until = time.monotonic() + args.start_delay
        while env.app.is_running() and time.monotonic() < until:
            env.world.render()
            time.sleep(.01)
    wall_start = time.monotonic()
    for name, duration, goal, width in phases:
        print('PICK_PLACE_PHASE', name, flush=True)
        count = round(duration / cfg.simulation.timestep_s)
        for index in range(count):
            if not env.app.is_running():
                raise RuntimeError('Simulation closed during task')
            while state['pause']:
                env.world.render()
                time.sleep(.01)
            u = (index + 1) / count
            u = u*u*(3-2*u)
            target = previous + u * (goal-previous)
            jaw = previous_width + u*(width-previous_width)
            env.tick(target, jaw, step)
            if step % args.render_every == 0:
                update_surfaces(env)
                env.world.render()
                from omni.kit.viewport.utility import get_active_viewport
                viewport = get_active_viewport()
                if viewport:
                    viewport.camera_path = '/World/Camera_0' if state['view']=='fingers' and cfg.simulation.render else '/World/TaskCamera'
                points = env.get_vertices_world()
                if not np.isfinite(points).all() or np.max(np.abs(points)) > 5:
                    raise RuntimeError('Soft finger simulation became unstable')
                pose = env.get_object_pose()
                if initial_z is not None and name in ('lift', 'transfer'):
                    lift_delta = max(lift_delta, float(pose[2] - initial_z))
                    lifted = lifted or lift_delta > .06
                samples.append(dict(time_s=env.time_s, phase=name, object_pose=pose.tolist(),
                    hand_position=env.hand_position.tolist(), joints=env.robot.get_dof_positions().numpy()[0].tolist(), jaw_width_m=jaw))
                label.text = f'{name} | sim {env.time_s:.1f}s | lifted: {lifted}'
                if args.realtime:
                    delay = wall_start + env.time_s - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, .05))
            step += 1
        (path / 'trajectory.json').write_text(json.dumps(samples, indent=2))
        print('PICK_PLACE_CHECKPOINT', name, 'object=', env.get_object_pose().tolist(),
              'hand=', env.hand_position.tolist(), flush=True)
        if args.snapshots and name in ('settle', 'grip_hold', 'transfer', 'settle_place'):
            capture_overview(env, path / (name+'_overview.png'))
        if np.linalg.norm(env.hand_position - goal) > .025:
            raise RuntimeError(f'Arm tracking error after {name}: measured {env.hand_position}, goal {goal}')
        if name == 'settle':
            initial_z = float(env.get_object_pose()[2])
        previous, previous_width = goal.copy(), width
        if cfg.simulation.render and name in ('grip_hold', 'transfer', 'settle_place'):
            rgb, depth = env.render_camera()
            np.savez_compressed(path / (name+'_cameras.npz'), rgb=rgb, depth_m=depth,
                                intrinsics=env.intrinsics, camera_to_world=env.camera_poses)
    final = env.get_object_pose()
    goal_object = start + [0., .28, 0.]
    xy_error = float(np.linalg.norm(final[:2]-goal_object[:2]))
    z_error = float(abs(final[2]-goal_object[2]))
    linear, _ = env.object_view.get_velocities()
    speed = float(np.linalg.norm(linear.numpy()[0]))
    report = dict(seed=seed, task='franka_soft_finger_pick_place', object=episode.object_shape,
        lifted=lifted, max_lift_m=lift_delta, final_pose=final.tolist(), target_position=goal_object.tolist(),
        final_xy_error_m=xy_error, final_z_error_m=z_error, final_speed_m_s=speed,
        success=bool(lifted and xy_error < .035 and z_error < .015 and speed < .03),
        grasp_mode='deformable contact and friction; no fixed joint or object teleport',
        coupling='one-way: measured Franka FK drives FEM base nodes; FEM reaction is not fed back as arm torque',
        controller='Lula IK + joint position drives; scripted clear-table trajectory, no obstacle planner',
        config=cfg.to_dict())
    (path / 'trajectory.json').write_text(json.dumps(samples, indent=2))
    (path / 'result.json').write_text(json.dumps(report, indent=2))
    label.text = f"Success: {report['success']} | XY error {xy_error*1000:.1f} mm"
    print('PICK_PLACE_RESULT', json.dumps({k:v for k,v in report.items() if k!='config'}), flush=True)
    return report


def capture_overview(env, path):
    import omni.replicator.core as rep
    from shared.dataset import write_png
    product = rep.create.render_product('/World/TaskCamera', (960, 720))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    try:
        env.world.render()
        rep.orchestrator.step(delta_time=0., pause_timeline=False, wait_for_render=True)
        data = np.asarray(rgb.get_data())
        if not data.size:
            raise RuntimeError('Overview capture returned no image')
        write_png(path, data[..., :3].astype(np.uint8))
    finally:
        rgb.detach([product])
        product.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_livestream_arguments(parser)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='data/runs/franka_pick_place')
    parser.add_argument('--robot-usd', help='Optional local Franka USD/USDZ (same Panda link/joint names)')
    parser.add_argument('--object', choices=['box', 'cylinder', 'sphere'], default='box')
    parser.add_argument('--jaw-width', type=float, default=.028, help='Closed soft-finger surface gap in meters')
    parser.add_argument('--no-cameras', action='store_true', help='Skip finger RGB/depth products; overview still renders')
    parser.add_argument('--realtime', action='store_true')
    parser.add_argument('--physics-dt', type=float, default=.001, help='Physics timestep in seconds (use .0005 for finer FEM substeps)')
    parser.add_argument('--render-every', type=int, default=100, help='Physics steps per overview refresh')
    parser.add_argument('--start-delay', type=float, default=15., help='Seconds to wait for WebRTC connection after building scene')
    parser.add_argument('--snapshots', action='store_true', help='Save overview PNG at grasp, transfer and placement')
    parser.add_argument('--exit-after', action='store_true', help='Close after finite validation run, including in streaming mode')
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or not 0 <= args.jaw_width <= .08 or not np.isfinite(args.start_delay) or args.start_delay < 0:
        parser.error('Require episodes > 0, seed >= 0, and jaw width in [0,.08]')
    if not np.isfinite(args.physics_dt) or not .0001 <= args.physics_dt <= .001 or args.render_every < 1:
        parser.error('Require physics dt in [.0001,.001] and render-every >= 1')
    cfg = ExperimentConfig()
    cfg.gripper.profile = 'panda'
    cfg.gripper.base_height_m = 0.
    cfg.gripper.open_jaw_width_m = .08
    cfg.gripper.mount_tilt_range_deg = (0., 0.)
    cfg.object.fixed = False
    cfg.object.friction = 1.
    cfg.material.friction = 1.
    cfg.simulation.timestep_s = args.physics_dt
    cfg.simulation.solver_iterations = 200
    cfg.simulation.render = not args.no_cameras
    cfg.simulation.headless = not args.gui
    configure_visualization(parser, args, cfg.simulation)
    cfg.validate()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with FrankaSoftEnv(cfg, args.robot_usd) as env:
        reports = [run_episode(env, args.seed+i, out, args) for i in range(args.episodes)]
        (out / 'summary.json').write_text(json.dumps(reports, indent=2))
        if (cfg.simulation.livestream or args.gui) and not args.exit_after:
            env.wait_for_close()


if __name__ == '__main__':
    main()
