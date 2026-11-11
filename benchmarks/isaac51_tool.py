"""Shared STL/FEM fingers in an existing Isaac Sim 5.1 host scene.

The host owns the app, physics scene and clock. Create before reset(), initialize
once physics is live, follow measured mount frames before every physics step.
Only base nodes are driven; free-node deformation is solved by PhysX.
"""
import numpy as np
from shared.config import resolve_asset
from shared.geometry import mounted_camera_pose, transform_points
from shared.mesh import TetMesh, load_stl
from shared.tool_mount import validate_mount_poses


class MountedSoftTool:
    def __init__(self, stage, config, mounts, root="/World/SoftTool", focal_lengths_mm=(2.8, 2.8),
                 filtered_bodies=()):
        from omni.physx.scripts import deformableUtils, physicsUtils
        from pxr import Gf, Sdf, UsdGeom, Vt, PhysxSchema
        self.stage, self.config, self.root = stage, config.validate(), root
        self.mounts = validate_mount_poses(mounts)
        self.mesh = TetMesh.load(config.mesh)
        self.pin = self.mesh.vertices[:, 2] <= self.mesh.vertices[:, 2].min() + config.mesh.pin_height_m + 1e-8
        # Let 5.1 cook its legacy simulation/collision topology from the same
        # shared STL. A 6.2 TetMesh is not a portable legacy soft-body asset.
        surface, faces = load_stl(resolve_asset(config.mesh.stl), config.mesh.scale_to_m)
        self.initial, self.paths, self.views, self.camera_ops = [], [], [], []
        self.products, self.annotators = [], []
        material = root + "/TPU"
        deformableUtils.add_deformable_body_material(stage, material,
            youngs_modulus=config.material.young_modulus_pa, poissons_ratio=config.material.poisson_ratio,
            density=config.material.density_kg_m3, dynamic_friction=config.material.friction,
            elasticity_damping=config.material.damping)
        for side, mount in enumerate(self.mounts):
            path = root + ("/Finger_left" if side == 0 else "/Finger_right")
            points = transform_points(self.mesh.vertices, mount[:3, 3], mount[:3, :3]).astype(np.float32)
            skin = transform_points(surface, mount[:3, 3], mount[:3, :3]).astype(np.float32)
            mesh = UsdGeom.Mesh.Define(stage, path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(skin))
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDisplayColorAttr([Gf.Vec3f(.12, .7, .76) if side == 0 else Gf.Vec3f(.92, .54, .20)])
            if not deformableUtils.add_physx_deformable_body(stage, path,
                    simulation_hexahedral_resolution=44,
                    collision_simplification=False,
                    solver_position_iteration_count=config.simulation.solver_iterations,
                    self_collision=False, sleep_threshold=0., settling_threshold=0.):
                raise RuntimeError("Could not create legacy PhysX deformable " + path)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            collision.CreateContactOffsetAttr(config.simulation.contact_offset_m)
            collision.CreateRestOffsetAttr(config.simulation.rest_offset_m)
            physicsUtils.add_physics_material_to_prim(stage, mesh.GetPrim(), material)
            if filtered_bodies:
                from pxr import UsdPhysics
                UsdPhysics.FilteredPairsAPI.Apply(mesh.GetPrim()).CreateFilteredPairsRel().SetTargets(
                    [Sdf.Path(body) for body in filtered_bodies])
            self.initial.append(points)
            self.paths.append(path)
        if config.simulation.render:
            import omni.replicator.core as rep
            rep.orchestrator.set_capture_on_play(False)
            for side in (0, 1):
                camera = UsdGeom.Camera.Define(stage, root + f"/Camera_{side}")
                camera.CreateFocalLengthAttr(float(focal_lengths_mm[side]) * .01)
                camera.CreateHorizontalApertureAttr(config.camera.sensor_width_mm * .01)
                camera.CreateVerticalApertureAttr(config.camera.sensor_width_mm * config.camera.height / config.camera.width * .01)
                camera.CreateClippingRangeAttr(Gf.Vec2f(config.camera.near_m, config.camera.far_m))
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
        self.views = [self.tensor_sim.create_soft_body_view(path) for path in self.paths]
        self.indices, self.targets, self.local_nodes, self.pin_masks = [], [], [], []
        for side, view in enumerate(self.views):
            actual = view.get_sim_nodal_positions()
            points = actual.cpu().numpy()[0]
            pose = self.mounts[side]
            local = (points - pose[:3, 3]) @ pose[:3, :3]
            pin = local[:, 2] <= local[:, 2].min() + self.config.mesh.pin_height_m + 1e-7
            if not np.isfinite(points).all() or not pin.any() or pin.all():
                raise RuntimeError("Invalid cooked soft-finger mesh or pin selection")
            self.local_nodes.append(local)
            self.pin_masks.append(pin)
            self.indices.append(torch.arange(view.count, dtype=torch.int32, device=actual.device))
            targets = view.get_sim_kinematic_targets().clone()
            targets[..., :3] = actual
            targets[..., 3] = 1.
            targets[:, pin, 3] = 0.
            self.targets.append(targets)
            print("SOFT_TOOL_MESH", side, "nodes", len(points), "pinned", int(pin.sum()), flush=True)
        self.follow(self.mounts)

    def follow(self, mounts):
        import torch
        from pxr import Gf
        self.mounts = validate_mount_poses(mounts)
        for side, (view, targets, indices) in enumerate(zip(self.views, getattr(self, "targets", []), getattr(self, "indices", []))):
            pose = self.mounts[side]
            points = transform_points(self.local_nodes[side], pose[:3, 3], pose[:3, :3])
            targets[..., :3] = torch.as_tensor(points, dtype=targets.dtype, device=targets.device)
            view.set_sim_kinematic_targets(targets, indices)
        for op, pose in zip(self.camera_ops, self.camera_poses):
            usd_pose = pose.copy()
            usd_pose[:3, :3] = usd_pose[:3, :3] @ np.diag([1., -1., -1.])
            op.Set(Gf.Matrix4d(*usd_pose.T.ravel().tolist()))

    @property
    def camera_poses(self):
        return np.stack([mounted_camera_pose(self.config.camera, pose[:3, 3], pose[:3, :3]) for pose in self.mounts])

    def positions(self):
        return [view.get_sim_nodal_positions().cpu().numpy()[0].copy() for view in self.views]

    def capture(self):
        import omni.replicator.core as rep
        rep.orchestrator.step(delta_time=0., pause_timeline=False, wait_for_render=True)
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
