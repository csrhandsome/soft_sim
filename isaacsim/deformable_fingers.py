"""Shared Isaac 6.2 FEM authoring; never creates an app, world or physics scene."""

import numpy as np
from shared.geometry import transform_points


def create_fingers(stage, config, mesh, poses, paths, material_path):
    from isaacsim.core.experimental.prims import DeformablePrim
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import Gf, UsdGeom, Vt, PhysxSchema

    if not deformableUtils.add_deformable_material(
        stage, material_path, density=config.material.density_kg_m3,
        youngs_modulus=config.material.young_modulus_pa, poissons_ratio=config.material.poisson_ratio,
        static_friction=config.material.friction, dynamic_friction=config.material.friction,
    ):
        raise RuntimeError('Failed to create TPU volume material')
    views, initial_points = [], []
    for side, (path, pose) in enumerate(zip(paths, poses)):
        points = transform_points(mesh.vertices, pose[:3, 3], pose[:3, :3])
        initial_points.append(points)
        volume = UsdGeom.TetMesh.Define(stage, path)
        volume.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        volume.GetTetVertexIndicesAttr().Set(Vt.Vec4iArray.FromNumpy(mesh.tets.astype(np.int32)))
        volume.CreateDisplayColorAttr([Gf.Vec3f(.12, .70, .76) if side == 0 else Gf.Vec3f(.92, .54, .20)])
        if not deformableUtils.set_physics_volume_deformable_body(stage, volume.GetPath()):
            raise RuntimeError(f'Failed to create volume deformable {path}; Isaac Sim 6.2 is required')
        prim = volume.GetPrim()
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr(config.simulation.contact_offset_m)
        collision.CreateRestOffsetAttr(config.simulation.rest_offset_m)
        if not prim.ApplyAPI('PhysxBaseDeformableBodyAPI'):
            raise RuntimeError('PhysxBaseDeformableBodyAPI unavailable; Isaac Sim 6.2 is required')
        for key, value in dict(selfCollision=False, solverPositionIterationCount=config.simulation.solver_iterations,
                               linearDamping=config.material.damping, sleepThreshold=0., settlingThreshold=0.).items():
            prim.GetAttribute(f'physxDeformableBody:{key}').Set(value)
        physicsUtils.add_physics_material_to_prim(stage, prim, material_path)
        views.append(DeformablePrim(path))
    return views, initial_points


def initialize_fingers(views, initial_points):
    """Call after the host world resets/starts its physics tensor simulation."""
    for view, points in zip(views, initial_points):
        if not view.is_physics_tensor_entity_valid():
            raise RuntimeError('PhysX deformable tensor view unavailable; check GPU and host initialization')
        positions, _, rest = view.get_nodal_positions()
        if positions.shape != (1, len(points), 3):
            raise RuntimeError('PhysX changed the volume mesh node count')
        if not np.allclose(rest.numpy()[0], points, atol=2e-6):
            raise RuntimeError('PhysX rest-node ordering differs from the shared mesh')
        view.set_nodal_positions(points[None].astype(np.float32))
        view.set_nodal_velocities(np.zeros_like(points[None], dtype=np.float32))
