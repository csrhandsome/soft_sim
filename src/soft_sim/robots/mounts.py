"""Measured tool/finger frames for attaching the shared mesh to other robots."""

from dataclasses import replace
import numpy as np

from soft_sim.geometry import finger_pose


def rigid_transform(value):
    """Validate an SI, column-vector SE(3) transform; reject scale/reflection."""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite 4x4 rigid transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError('Mount transforms must have unit scale and proper rotation')
    return matrix.copy()


def validate_mount_poses(poses):
    if len(poses) != 2:
        raise ValueError('Supply left and right finger-base world transforms')
    return np.stack([rigid_transform(pose) for pose in poses])


def mounts_from_tool(gripper, world_from_tool, jaw_width_m, angles_deg=(0., 0.)):
    """Parallel supports in tool frame: close on X, fingers grow along +Z.

    Use measured jaw-link frames instead when the native mechanism is available.
    `world_from_tool` includes the calibrated TCP-to-adapter transform. The
    standalone scene's base_height_m is intentionally excluded here.
    """
    world_from_tool = rigid_transform(world_from_tool)
    if not np.isfinite(jaw_width_m) or not 0 <= jaw_width_m <= gripper.open_jaw_width_m:
        raise ValueError('Jaw width must lie between zero and the configured open width')
    angles = np.asarray(angles_deg, dtype=float)
    if angles.shape != (2,) or not np.isfinite(angles).all():
        raise ValueError('Expected two finite mount angles in degrees')
    local_config = replace(gripper, base_height_m=0.)
    result = []
    for side in (0, 1):
        position, rotation = finger_pose(local_config, side, angles[side],
                                         jaw_width_m + 2 * gripper.adapter_offset_m)
        local = np.eye(4)
        local[:3, :3], local[:3, 3] = rotation, position
        result.append(world_from_tool @ local)
    return validate_mount_poses(result)
