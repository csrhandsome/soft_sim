"""Isaac Sim 6.2 / PhysX backend; imports Kit APIs only after SimulationApp starts."""

from importlib.metadata import version
import numpy as np

from shared.environment import BaseGripperEnv, ForceReading
from shared.force import QuasistaticForceEstimator
from shared.geometry import transform_points


class IsaacSimGripperEnv(BaseGripperEnv):
    backend_name = 'isaacsim'

    def __init__(self, config):
        super().__init__(config)
        self.app = self.world = None
        self.views = []
        self.products = []
        self.annotators = []
        self.camera_ops = []
        self.mount_ops = []
        self.estimators = []
        self.object_view = None
        from app_launcher import create_simulation_app
        self.app = create_simulation_app(config.simulation)
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        for name in ('isaacsim.core.api', 'isaacsim.core.experimental.prims', 'omni.replicator.core'):
            manager.set_extension_enabled_immediate(name, True)
            if not manager.is_extension_enabled(name):
                self.close()
                raise RuntimeError(f'Isaac extension could not be enabled: {name}')

    @property
    def backend_metadata(self):
        return dict(version=version('isaacsim'), soft_body='PhysX volume deformable / OmniPhysics schema',
                    force_source='quasistatic_corotated_fem_estimate', native_contact_labels=False,
                    force_limitations='Estimated resultant: ignores inertia and solver damping; cannot separate friction. Use settled frames only.',
                    usd_object_collision='Per-mesh convex hull; source dimensions fitted to sampled object diameter',
                    damping_mapping='PhysX linearDamping (not numerically equivalent to MuJoCo elasticity damping)',
                    boundary_control='kinematic base nodes; simplified shared gripper profile, not a native articulation')

    def build_scene(self, episode):
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.experimental.prims import DeformablePrim
        from omni.physx.scripts import deformableUtils, physicsUtils
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, Vt, PhysxSchema

        self._clear_scene()
        omni.usd.get_context().new_stage()
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        cfg = self.config
        self.world = World(stage_units_in_meters=1., physics_dt=cfg.simulation.timestep_s,
                           rendering_dt=cfg.simulation.timestep_s, backend='numpy', device='cuda:0')
        physics = self.world.get_physics_context()
        physics.enable_gpu_dynamics(True)
        physics.set_broadphase_type('GPU')
        gravity = np.asarray(cfg.simulation.gravity_m_s2, dtype=float)
        scene = UsdPhysics.Scene.Get(self.stage, physics.prim_path)
        norm = float(np.linalg.norm(gravity))
        scene.CreateGravityDirectionAttr(Gf.Vec3f(*(gravity / norm if norm else [0, 0, -1])))
        scene.CreateGravityMagnitudeAttr(norm)
        # Disable contact-rest offsets comparable to the cage's thin beams.
        # Build the plane locally so offline runs do not require the asset server.
        self.world.scene.add_ground_plane()
        light = UsdLux.DomeLight.Define(self.stage, '/World/Light')
        light.CreateIntensityAttr(900.)
        if not deformableUtils.add_deformable_material(
            self.stage, '/World/TPU85A', density=cfg.material.density_kg_m3,
            youngs_modulus=cfg.material.young_modulus_pa, poissons_ratio=cfg.material.poisson_ratio,
            static_friction=cfg.material.friction, dynamic_friction=cfg.material.friction,
        ):
            raise RuntimeError('Failed to create TPU volume material')
        self.views, self.estimators = [], []
        self._initial_points = []
        for side, name in enumerate(('left', 'right')):
            path = f'/World/Finger_{name}'
            points = transform_points(self.mesh.vertices, *self.finger_pose(side))
            self._initial_points.append(points)
            mesh = UsdGeom.TetMesh.Define(self.stage, path)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(self.mesh.tets.astype(np.int32)))
            mesh.CreateDisplayColorAttr([Gf.Vec3f(.12, .70, .76) if side == 0 else Gf.Vec3f(.92, .54, .20)])
            if not deformableUtils.set_physics_volume_deformable_body(self.stage, mesh.GetPath()):
                raise RuntimeError(f'Failed to create volume deformable {path}')
            prim = mesh.GetPrim()
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(cfg.simulation.contact_offset_m)
            collision.CreateRestOffsetAttr(cfg.simulation.rest_offset_m)
            if not prim.ApplyAPI('PhysxBaseDeformableBodyAPI'):
                raise RuntimeError('PhysxBaseDeformableBodyAPI is unavailable; this backend requires Isaac Sim 6.2')
            for key, value in dict(selfCollision=False, solverPositionIterationCount=cfg.simulation.solver_iterations,
                                   linearDamping=cfg.material.damping, sleepThreshold=0., settlingThreshold=0.).items():
                prim.GetAttribute(f'physxDeformableBody:{key}').Set(value)
            physicsUtils.add_physics_material_to_prim(self.stage, prim, '/World/TPU85A')
            self.views.append(DeformablePrim(path))
            self.estimators.append(QuasistaticForceEstimator(points, self.mesh.tets, self.pin_mask,
                                                            cfg.material, gravity))
        self._build_object(episode)
        if cfg.simulation.scene_preset != 'plain':
            from presentation import build_presentation
            build_presentation(self)
        rep.orchestrator.set_capture_on_play(False)
        self.camera_ops = []
        if cfg.simulation.render:
            for side in (0, 1):
                camera = UsdGeom.Camera.Define(self.stage, f'/World/Camera_{side}')
                # USD optical lengths are tenths of a scene unit. With a meter
                # stage, 1 mm = .01 camera attribute units (FOV uses their ratio).
                camera.CreateFocalLengthAttr(float(episode.focal_lengths_mm[side]) * .01)
                camera.CreateHorizontalApertureAttr(cfg.camera.sensor_width_mm * .01)
                camera.CreateVerticalApertureAttr(cfg.camera.sensor_width_mm * cfg.camera.height / cfg.camera.width * .01)
                camera.CreateClippingRangeAttr(Gf.Vec2f(cfg.camera.near_m, cfg.camera.far_m))
                self.camera_ops.append(camera.AddTransformOp())
                product = rep.create.render_product(str(camera.GetPath()), (cfg.camera.width, cfg.camera.height))
                self.products.append(product)
                rgb = rep.AnnotatorRegistry.get_annotator('rgb')
                depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
                for annotator in (rgb, depth):
                    annotator.attach([product])
                self.annotators.append((rgb, depth))
        self.world.reset()
        for side, view in enumerate(self.views):
            if not view.is_physics_tensor_entity_valid():
                raise RuntimeError('PhysX deformable tensor view is unavailable. Check GPU/driver and Isaac logs.')
            positions, _, rest = view.get_nodal_positions()
            if positions.shape != (1, len(self.mesh.vertices), 3):
                raise RuntimeError('PhysX changed the volume mesh node count; refusing misaligned deformation labels')
            if not np.allclose(rest.numpy()[0], self._initial_points[side], atol=2e-6):
                raise RuntimeError('PhysX rest-node ordering differs from the shared mesh; refusing misaligned labels')
            view.set_nodal_positions(self._initial_points[side][None].astype(np.float32))
            view.set_nodal_velocities(np.zeros_like(self._initial_points[side][None], dtype=np.float32))
        self.set_gripper_state(self.state)
        if cfg.simulation.livestream:
            # An overview camera is independent of the two training cameras.
            from omni.kit.viewport.utility import get_active_viewport
            target = Gf.Vec3d(0., 0., cfg.gripper.base_height_m + float(self.mesh.vertices[:, 2].mean()))
            eye = target + Gf.Vec3d(.28, -.36, .23)
            camera = UsdGeom.Camera.Define(self.stage, '/World/OverviewCamera')
            camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 10.))
            camera.CreateFocalLengthAttr(35.)
            camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = str(camera.GetPath())
            self.world.render()

    def _build_object(self, episode):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from assets import add_usd_reference
        path = '/World/Object'
        self.object_root = UsdGeom.Xform.Define(self.stage, path)
        r, h = episode.object_radius_m, episode.object_height_m
        position = Gf.Vec3d(*episode.object_position_m)
        self.object_root.AddTranslateOp().Set(position)
        self.object_root.AddRotateXYZOp().Set(Gf.Vec3f(*episode.object_rpy_deg))
        if episode.object_shape.startswith('usd:'):
            prim = add_usd_reference(self.stage, episode.object_shape[4:], path + '/Asset')
            # Fit the mesh's largest dimension to 2*r and center it on the test pose.
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render']).ComputeLocalBound(prim).ComputeAlignedRange()
            extent = np.array(bounds.GetSize())
            if not np.isfinite(extent).all() or extent.max() <= 0:
                raise RuntimeError('USD object has no measurable geometry')
            scale = 2 * r / float(extent.max())
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*(-np.array(bounds.GetMidpoint()) * scale)))
            xform.AddScaleOp().Set(Gf.Vec3f(scale))
            # Treat catalog objects as one rigid object, regardless of source physics.
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.RigidBodyAPI):
                    child.RemoveAPI(UsdPhysics.RigidBodyAPI)
                if child.IsA(UsdPhysics.Joint):
                    child.SetActive(False)
                if child.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(child)
                    UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr('convexHull')
        else:
            shape_path = path + '/Shape'
            if episode.object_shape == 'sphere':
                shape = UsdGeom.Sphere.Define(self.stage, shape_path)
                shape.CreateRadiusAttr(r)
            elif episode.object_shape == 'cylinder':
                shape = UsdGeom.Cylinder.Define(self.stage, shape_path)
                shape.CreateRadiusAttr(r)
                shape.CreateHeightAttr(h)
                shape.CreateAxisAttr('Z')
            else:
                shape = UsdGeom.Cube.Define(self.stage, shape_path)
                shape.CreateSizeAttr(1.)
                shape.AddScaleOp().Set(Gf.Vec3f(2 * r, 2 * r, h))
            shape.CreateDisplayColorAttr([Gf.Vec3f(.8, .25, .3)])
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        root = self.object_root.GetPrim()
        for child in Usd.PrimRange(root):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(self.config.object.collision)
                offsets = PhysxSchema.PhysxCollisionAPI.Apply(child)
                offsets.CreateContactOffsetAttr(self.config.simulation.contact_offset_m)
                offsets.CreateRestOffsetAttr(self.config.simulation.rest_offset_m)
        if not self.config.object.fixed:
            UsdPhysics.RigidBodyAPI.Apply(root)
            UsdPhysics.MassAPI.Apply(root).CreateMassAttr(self.config.object.mass_kg)
            from isaacsim.core.experimental.prims import RigidPrim
            self.object_view = RigidPrim(path)
        material = UsdShade.Material.Define(self.stage, '/World/ObjectMaterial')
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(self.config.object.friction)
        physics.CreateDynamicFrictionAttr(self.config.object.friction)
        UsdShade.MaterialBindingAPI.Apply(root).Bind(material, materialPurpose='physics')

    def set_gripper_state(self, state):
        from pxr import Gf
        for side, view in enumerate(self.views):
            targets = transform_points(self.mesh.vertices, *self.finger_pose(side)).astype(np.float32)
            view.set_nodal_kinematic_position_targets(targets[None], self.pin_mask[None, :, None])
        for side, op in enumerate(self.mount_ops):
            position, rotation = self.finger_pose(side)
            matrix = np.eye(4); matrix[:3,:3] = rotation; matrix[:3,3] = position
            op.Set(Gf.Matrix4d(*matrix.T.ravel().tolist()))
        for side, op in enumerate(self.camera_ops):
            pose = self.camera_poses[side].copy()
            # USD camera looks along local -Z with local +Y up.
            pose[:3, :3] = pose[:3, :3] @ np.diag([1., -1., -1.])
            op.Set(Gf.Matrix4d(*pose.T.ravel().tolist()))

    def step_physics(self):
        if not self.app.is_running():
            raise RuntimeError('Isaac Sim was closed')
        self.world.step(render=False)
        if self.config.simulation.livestream and self.step_index % self.config.simulation.capture_every_steps == 0:
            self.world.render()

    def wait_for_close(self):
        from app_launcher import keep_open
        keep_open(self.app, render=self.world.render)

    def get_vertices_world(self):
        return np.stack([view.get_nodal_positions()[0].numpy()[0].copy() for view in self.views])

    def get_object_pose(self):
        if self.object_view is not None:
            positions, orientations = self.object_view.get_world_poses()
            return np.r_[positions.numpy()[0], orientations.numpy()[0]]
        from pxr import Usd, UsdGeom
        matrix = UsdGeom.Xformable(self.object_root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        quat = matrix.ExtractRotationQuat()
        return np.r_[np.asarray(matrix.ExtractTranslation()), quat.GetReal(), np.asarray(quat.GetImaginary())]

    def read_force(self):
        positions = self.get_vertices_world()
        if not np.isfinite(positions).all():
            raise RuntimeError('Non-finite soft-body state; dataset frame rejected')
        estimates = np.stack([estimator.estimate(p) for estimator, p in zip(self.estimators, positions)])
        # Isaac rigid contact reporters do not guarantee deformable/rigid pairs.
        # A missing native report MUST NOT be interpreted as a zero force label.
        return ForceReading(np.full(2, np.nan), np.full((2, 3), np.nan),
                            'quasistatic_corotated_fem_estimate', False, estimates)

    def render_camera(self):
        # Capture-on-play is disabled, so annotators need an explicit capture.
        # Zero delta keeps the rendered frame aligned with the labeled state.
        import omni.replicator.core as rep
        self.world.render()
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
        rgbs, depths = [], []
        for rgb_annotator, depth_annotator in self.annotators:
            rgb, depth = np.asarray(rgb_annotator.get_data()), np.asarray(depth_annotator.get_data())
            if not rgb.size or not depth.size:
                raise RuntimeError('Isaac camera annotator returned no data')
            rgbs.append(rgb[..., :3].astype(np.uint8).copy())
            depth = depth.astype(np.float32).reshape(self.config.camera.height, self.config.camera.width).copy()
            depth[(depth <= 0) | (depth >= self.config.camera.far_m) | ~np.isfinite(depth)] = np.nan
            depths.append(depth)
        return np.stack(rgbs), np.stack(depths)

    def _clear_scene(self):
        for pair, product in zip(self.annotators, self.products):
            for annotator in pair:
                annotator.detach([product])
            product.destroy()
        self.annotators, self.products, self.views = [], [], []
        self.camera_ops = []
        self.mount_ops = []
        self.object_view = None
        if self.world is not None:
            self.world.stop()
            self.world.clear_instance()
            self.world = None

    def close(self):
        # SimulationApp.close may terminate Python, so report active errors first.
        import sys
        import traceback
        if sys.exc_info()[0] is not None:
            traceback.print_exc()
        try:
            self._clear_scene()
        finally:
            if self.app is not None:
                self.app.close()
                self.app = None
