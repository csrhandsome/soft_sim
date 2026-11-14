"""Shared STL/FEM fingers in an existing Isaac Sim 5.1 host scene.

The host owns the app, physics scene and clock. Create before reset(), initialize
once physics is live, follow measured mount frames before every physics step.
Only base nodes are driven; free-node deformation is solved by PhysX.
"""
import numpy as np
from soft_sim.config import resolve_asset
from soft_sim.geometry import mounted_camera_pose, transform_points
from soft_sim.mesh import TetMesh, load_stl
from soft_sim.robots.mounts import validate_mount_poses
from soft_sim.sim.cameras import configure_pinhole_camera


def base_pin_mask(local_nodes, height_m, pitch_m):
    """Keep the intended support layer despite sub-cell cooking quantization."""
    tolerance = .01 * pitch_m
    return local_nodes[:, 2] <= local_nodes[:, 2].min() + height_m + tolerance, tolerance


class MountedSoftTool:
    def __init__(self, stage, config, mounts, root="/World/SoftTool", focal_lengths_mm=(2.8, 2.8),
                 filtered_bodies=(), sleep_threshold=0., settling_threshold=0., sleep_damping=None,
                 max_depenetration_velocity=None, rt_subframes=2, lighting_samples_per_pixel=None,
                 simulation_resolution=44, surface_max_edge_m=.002, explicit_fem_mesh=False,
                 self_collision=False, rest_authoring_scale=1.):
        from omni.physx.scripts import deformableUtils, physicsUtils
        from pxr import Gf, Sdf, UsdGeom, Vt, PhysxSchema, UsdShade
        from soft_sim.sim.render_surface import create_tpu_material
        self.stage, self.config, self.root = stage, config.validate(), root
        self.explicit_fem_mesh = explicit_fem_mesh
        self.rest_authoring_scale = float(rest_authoring_scale) if explicit_fem_mesh else 1.
        if not np.isfinite(self.rest_authoring_scale) or self.rest_authoring_scale <= 0:
            raise ValueError("Rest authoring scale must be finite and positive")
        self.mounts = validate_mount_poses(mounts)
        self.mesh = TetMesh.load(config.mesh)
        self.pin = self.mesh.vertices[:, 2] <= self.mesh.vertices[:, 2].min() + config.mesh.pin_height_m + 1e-8
        # Let 5.1 cook its legacy simulation/collision topology from the same
        # shared STL. A 6.2 TetMesh is not a portable legacy soft-body asset.
        surface, faces = load_stl(resolve_asset(config.mesh.stl), config.mesh.scale_to_m)
        self.surface_max_edge_m = surface_max_edge_m
        self.render_meshes, self.surface_bindings = [], []
        self.mesh_settings = dict(simulation_hexahedral_resolution=simulation_resolution,
                                  nominal_simulation_cell_m=float(np.ptp(surface, axis=0).max() / simulation_resolution),
                                  surface_max_edge_m=surface_max_edge_m,
                                  collision_source_vertices=len(surface), collision_source_triangles=len(faces))
        self.mesh_settings.update(physics_mesh_source='explicit_tetrahedra' if explicit_fem_mesh else 'physx_voxel_cooking',
                                  rest_authoring_scale=self.rest_authoring_scale,
                                  explicit_mesh_file=config.mesh.tetra_mesh if explicit_fem_mesh else None,
                                  pin_height_m=config.mesh.pin_height_m, self_collision=self_collision,
                                  explicit_initialization='authored rest positions and zero velocity before pinning'
                                  if explicit_fem_mesh else None)
        self.initial, self.paths, self.views, self.camera_ops = [], [], [], []
        self.products, self.annotators = [], []
        self.camera_projection = []
        self.render_settings = dict(antialiasing="FXAA", rt_subframes=rt_subframes, motion_blur=False)
        material = root + "/TPU"
        deformableUtils.add_deformable_body_material(stage, material,
            youngs_modulus=config.material.young_modulus_pa, poissons_ratio=config.material.poisson_ratio,
            density=config.material.density_kg_m3, dynamic_friction=config.material.friction,
            elasticity_damping=config.material.damping)
        self.visual_material = create_tpu_material(stage, root + '/Looks/TPU', config.material)
        for side, mount in enumerate(self.mounts):
            path = root + ("/Finger_left" if side == 0 else "/Finger_right")
            points = transform_points(self.mesh.vertices, mount[:3, 3], mount[:3, :3]).astype(np.float32)
            # Keep rest geometry independent of room translation/rotation.
            # Tensor APIs still expose world nodes; display embedding uses
            # those world nodes after removing this initial rigid transform.
            # Optional compensated local units support isolated diagnostics.
            # Production uses meters. World geometry and tensor states remain
            # SI regardless of this local representation.
            skin = (surface*self.rest_authoring_scale).astype(np.float32)
            mesh = UsdGeom.Mesh.Define(stage, path)
            authored_pose = mount.copy()
            authored_pose[:3, :3] /= self.rest_authoring_scale
            mesh.AddTransformOp().Set(Gf.Matrix4d(*authored_pose.T.ravel().tolist()))
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(skin))
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*config.material.color_linear)])
            explicit = {}
            if explicit_fem_mesh:
                # Use the same conforming volume mesh for mechanics and
                # collision, following SOFA's deformable modelling examples.
                tetra_points = [Gf.Vec3f(*point) for point in (self.mesh.vertices*self.rest_authoring_scale).tolist()]
                tetra_indices = self.mesh.tets.astype(np.int32).ravel().tolist()
                explicit = dict(simulation_rest_points=tetra_points, simulation_indices=tetra_indices,
                                collision_rest_points=tetra_points, collision_indices=tetra_indices)
            if not deformableUtils.add_physx_deformable_body(stage, path,
                    simulation_hexahedral_resolution=simulation_resolution,
                    collision_simplification=False,
                    solver_position_iteration_count=config.simulation.solver_iterations,
                    vertex_velocity_damping=config.simulation.deformable_vertex_velocity_damping,
                    self_collision=self_collision, sleep_threshold=sleep_threshold,
                    settling_threshold=settling_threshold, sleep_damping=sleep_damping, **explicit):
                raise RuntimeError("Could not create legacy PhysX deformable " + path)
            if max_depenetration_velocity is not None:
                PhysxSchema.PhysxDeformableAPI(mesh.GetPrim()).CreateMaxDepenetrationVelocityAttr(max_depenetration_velocity)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            collision.CreateContactOffsetAttr(config.simulation.contact_offset_m)
            collision.CreateRestOffsetAttr(config.simulation.rest_offset_m)
            physicsUtils.add_physics_material_to_prim(stage, mesh.GetPrim(), material)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.visual_material)
            selected_filtered = filtered_bodies.get(side, ()) if isinstance(filtered_bodies, dict) else filtered_bodies
            if selected_filtered:
                from pxr import UsdPhysics
                UsdPhysics.FilteredPairsAPI.Apply(mesh.GetPrim()).CreateFilteredPairsRel().SetTargets(
                    [Sdf.Path(body) for body in selected_filtered])
            self.initial.append(points)
            self.paths.append(path)
        if config.simulation.render:
            import omni.replicator.core as rep
            rep.orchestrator.set_capture_on_play(False)
            # A moving close-up cage should not depend on a low-resolution
            # DLSS history from the previous sampled simulation state.
            rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
            import carb
            settings = carb.settings.get_settings()
            settings.set("/rtx/post/motionblur/enabled", False)
            for name, path in (("direct_lighting_samples_per_pixel", "/rtx/directLighting/sampledLighting/samplesPerPixel"),
                               ("reflection_samples_per_pixel", "/rtx/reflections/sampledLighting/samplesPerPixel")):
                if lighting_samples_per_pixel is not None:
                    settings.set_int(path, lighting_samples_per_pixel)
                self.render_settings[name] = settings.get(path)
            for side in (0, 1):
                camera = UsdGeom.Camera.Define(stage, root + f"/Camera_{side}")
                self.camera_projection.append(configure_pinhole_camera(camera, config.camera, focal_lengths_mm[side]))
                self.camera_ops.append(camera.AddTransformOp())
                product = rep.create.render_product(str(camera.GetPath()), (config.camera.width, config.camera.height))
                pair = [rep.AnnotatorRegistry.get_annotator(name) for name in ("rgb", "distance_to_image_plane")]
                for annotator in pair:
                    annotator.attach([product])
                self.products.append(product)
                self.annotators.append(pair)
        self.follow(self.mounts)

    def initialize(self):
        import torch
        import omni.physics.tensors as tensors
        self.tensor_sim = tensors.create_simulation_view("torch")
        self.tensor_sim.set_subspace_roots("/")
        if hasattr(self, "fixtures"):
            self.fixtures.initialize(self.tensor_sim)
        self.views = [self.tensor_sim.create_soft_body_view(path) for path in self.paths]
        self.indices, self.targets, self.local_nodes, self.pin_masks = [], [], [], []
        self._last_target_points = [None, None]
        self.sim_tets, self.mesh_topology = [], []
        for side, view in enumerate(self.views):
            actual = view.get_sim_nodal_positions()
            if self.explicit_fem_mesh:
                # World.reset() advances the initially unpinned body: using
                # those positions as rest bakes a ~30 um gravity drop into the
                # clamp and surface embedding. Restore the authored rest state
                # before installing the base targets. Verify ordering first.
                actual_tets = view.get_sim_element_indices().cpu().numpy()[0]
                if actual.shape[1:] != self.initial[side].shape or not np.array_equal(actual_tets, self.mesh.tets):
                    raise RuntimeError('PhysX reordered the explicit FEM; rest-state restoration needs an index mapping')
                body_indices = torch.arange(view.count, dtype=torch.int32, device=actual.device)
                rest_tensor = torch.as_tensor(self.initial[side][None], dtype=actual.dtype, device=actual.device)
                view.set_sim_nodal_positions(rest_tensor, body_indices)
                view.set_sim_nodal_velocities(torch.zeros_like(actual), body_indices)
                actual = view.get_sim_nodal_positions()
            points = actual.cpu().numpy()[0]
            pose = self.mounts[side]
            local = (points - pose[:3, 3]) @ pose[:3, :3]
            if self.explicit_fem_mesh:
                local = self.mesh.vertices.copy()
            # Cooking quantizes the nominal 2 mm lattice. Its second layer can
            # land a few micrometers above 4 mm; a strict 0.1 um cutoff drops
            # that entire layer and changes the support with world pose. Allow
            # 1% of a mesh cell so the intended base layer remains clamped.
            pin, pin_tolerance = base_pin_mask(local, self.config.mesh.pin_height_m, self.config.mesh.voxel_pitch_m)
            if not np.isfinite(points).all() or not pin.any() or pin.all():
                raise RuntimeError("Invalid cooked soft-finger mesh or pin selection")
            self.local_nodes.append(local)
            self.pin_masks.append(pin)
            tets = view.get_sim_element_indices().cpu().numpy()[0].copy()
            tets = tets[np.all(np.diff(np.sort(tets, axis=1), axis=1) != 0, axis=1)]
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            edges = np.concatenate([tets[:, [0, i]] for i in (1, 2, 3)])
            graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(local), len(local)))
            count, labels = connected_components(graph, directed=False)
            unanchored = ~np.isin(labels, np.unique(labels[pin]))
            from soft_sim.mesh import tet_volumes
            element_volumes = tet_volumes(local, tets)
            zero_free_rest_elements = 0
            if self.explicit_fem_mesh:
                inverse_rest = view.get_sim_element_rest_poses().cpu().numpy().reshape(-1, 3, 3)
                zero_free_rest_elements = int(np.count_nonzero(
                    ~np.any(inverse_rest != 0, axis=(1, 2)) & ~pin[tets].all(axis=1)))
            source_volume = self.mesh.metadata['stl_volume_m3']
            self.mesh_topology.append(dict(nodes=len(local), tets=len(tets), components=int(count),
                                           pinned_nodes=int(pin.sum()), pin_tolerance_m=pin_tolerance,
                                           effective_pin_height_m=float(np.max(local[pin, 2]) - local[:, 2].min()),
                                           unanchored_nodes=int(unanchored.sum()),
                                           reference_volume_m3=float(element_volumes.sum()),
                                           reference_volume_to_stl_ratio=float(element_volumes.sum() / source_volume),
                                           minimum_reference_tet_volume_m3=float(element_volumes.min()),
                                           tensor_reported_zero_free_rest_elements=zero_free_rest_elements,
                                           valid=not bool(unanchored.any()) and bool(np.all(element_volumes > 0))))
            self.sim_tets.append(tets)
            print("SOFT_TOOL_TOPOLOGY", side, self.mesh_topology[-1], flush=True)
            self.indices.append(torch.arange(view.count, dtype=torch.int32, device=actual.device))
            targets = view.get_sim_kinematic_targets().clone()
            targets[..., :3] = actual
            targets[..., 3] = 1.
            targets[:, pin, 3] = 0.
            self.targets.append(targets)
            print("SOFT_TOOL_MESH", side, "nodes", len(points), "pinned", int(pin.sum()), flush=True)
        self.follow(self.mounts)
        # The 5.1 cooking process changes node count and ordering.
        self.reference_state = dict(
            finger_reference_left=self.local_nodes[0], finger_reference_right=self.local_nodes[1],
            finger_pin_left=self.pin_masks[0], finger_pin_right=self.pin_masks[1],
            finger_tets_left=self.sim_tets[0], finger_tets_right=self.sim_tets[1])
        if self.surface_max_edge_m > 0:
            from types import SimpleNamespace
            from soft_sim.sim.render_surface import bind_surface
            from pxr import UsdGeom, Vt, Gf, UsdShade
            for side, path in enumerate(self.paths):
                cooked = SimpleNamespace(vertices=self.local_nodes[side], tets=self.sim_tets[side])
                surface, faces, indices, weights = bind_surface(
                    cooked, self.config.mesh, max_edge_m=self.surface_max_edge_m)
                UsdGeom.Imageable(self.stage.GetPrimAtPath(path)).CreatePurposeAttr('guide')
                visual = UsdGeom.Mesh.Define(self.stage, path + '_Surface')
                visual.CreateFaceVertexCountsAttr([3] * len(faces))
                visual.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
                visual.CreateSubdivisionSchemeAttr('none')
                visual.CreateDisplayColorAttr([Gf.Vec3f(*self.config.material.color_linear)])
                UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(self.visual_material)
                self.render_meshes.append(visual)
                self.surface_bindings.append((indices, weights))
            self.mesh_settings.update(render_vertices=len(surface), render_triangles=len(faces),
                                      render_binding='barycentric interpolation of cooked simulation tetrahedra')
            print('SOFT_TOOL_SURFACE', self.mesh_settings, flush=True)
            self.sync_render_surface()

    def sync_render_surface(self):
        """Update display vertices from the current FEM state; never step physics."""
        if not self.render_meshes:
            return
        from pxr import Vt
        for visual, (indices, weights), points in zip(self.render_meshes, self.surface_bindings, self.positions()):
            surface = np.einsum('ni,nij->nj', weights, points[indices]).astype(np.float32)
            visual.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(surface))

    def reset_to_mounts(self, mounts):
        """Initialize the unloaded FEM after the host finishes resetting robots.

        This is an initial condition, never a correction during execution.
        Keep the authored local rest mesh and install it at measured mounts;
        do not retain transients from a robot's reset or renderer warmup.
        """
        import torch
        mounts = validate_mount_poses(mounts)
        for side, (view, indices) in enumerate(zip(self.views, self.indices)):
            pose = mounts[side]
            points = transform_points(self.local_nodes[side], pose[:3, 3], pose[:3, :3]).astype(np.float32)
            state = torch.as_tensor(points[None], device=self.targets[side].device, dtype=self.targets[side].dtype)
            view.set_sim_nodal_positions(state, indices)
            view.set_sim_nodal_velocities(torch.zeros_like(state), indices)
        self._last_target_points = [None, None]
        if hasattr(self, "fixtures"):
            self.fixtures.reset_to_mounts(mounts)
        self.follow(mounts)

    def follow(self, mounts):
        import torch
        from pxr import Gf
        self.mounts = validate_mount_poses(mounts)
        if hasattr(self, "fixtures"):
            self.fixtures.follow(self.mounts)
        for side, (view, targets, indices) in enumerate(zip(self.views, getattr(self, "targets", []), getattr(self, "indices", []))):
            pose = self.mounts[side]
            points = transform_points(self.local_nodes[side], pose[:3, 3], pose[:3, :3]).astype(np.float32)
            # Rewriting identical targets can keep a stationary soft body awake.
            # Compare the actual float32 target, without a motion deadband: even
            # the smallest representable mounting movement still reaches PhysX.
            if np.array_equal(points, self._last_target_points[side]):
                continue
            targets[..., :3] = torch.as_tensor(points, dtype=targets.dtype, device=targets.device)
            view.set_sim_kinematic_targets(targets, indices)
            self._last_target_points[side] = points
        for op, pose in zip(self.camera_ops, self.camera_poses):
            usd_pose = pose.copy()
            usd_pose[:3, :3] = usd_pose[:3, :3] @ np.diag([1., -1., -1.])
            op.Set(Gf.Matrix4d(*usd_pose.T.ravel().tolist()))

    @property
    def camera_poses(self):
        return np.stack([mounted_camera_pose(self.config.camera, pose[:3, 3], pose[:3, :3]) for pose in self.mounts])

    def positions(self):
        return [view.get_sim_nodal_positions().cpu().numpy()[0].copy() for view in self.views]

    def collision_positions(self):
        return [view.get_nodal_positions().cpu().numpy()[0].copy() for view in self.views]

    def capture(self, world):
        import omni.replicator.core as rep
        before = self.positions()
        self.sync_render_surface()
        # Physics-only steps do not flush PhysX/Fabric to the renderer. The
        # cameras are authored in USD by follow(), so without this refresh they
        # can see the previous render's finger geometry from the current pose.
        # World.render() flushes Fabric without advancing physics; a zero-time
        # Replicator step alone does not perform that flush in Isaac Sim 5.1.
        # IsaacLab's headless NO_GUI_OR_RENDERING mode makes render() a no-op.
        # Replicator still renders, so explicitly flush Fabric first; otherwise
        # robot/object images can remain at reset while recorded joints move.
        forward = getattr(world, "forward", None)
        if forward is not None:
            forward()
        world.render()
        rep.orchestrator.step(rt_subframes=self.render_settings["rt_subframes"],
                              delta_time=0., pause_timeline=False, wait_for_render=True)
        after = self.positions()
        motion = max(float(np.max(np.abs(a - b))) for a, b in zip(after, before))
        if motion > 1e-7:
            raise RuntimeError(f"Rendering advanced FEM positions by {motion:g} m during a zero-time capture")
        rgb, depth = [], []
        for color, distance in self.annotators:
            c, d = np.asarray(color.get_data()), np.asarray(distance.get_data())
            if not c.size or not d.size:
                raise RuntimeError("Empty soft-finger camera frame")
            color = c[..., :3].astype(np.uint8).copy()
            if color.std() < 1.:
                raise RuntimeError("Uniform camera image; check palm occlusion and interior lighting")
            rgb.append(color)
            d = d.astype(np.float32).reshape(self.config.camera.height, self.config.camera.width).copy()
            d[(d <= 0) | (d >= self.config.camera.far_m) | ~np.isfinite(d)] = np.nan
            depth.append(d)
        return np.stack(rgb), np.stack(depth)

    # The host owns Kit and closes its render graph with SimulationApp.close().
    # Do not manually detach shared RGB/depth annotator graph nodes in 5.1.
