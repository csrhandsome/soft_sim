"""Common world frame: +Z up, fingers on +/-X, closing about opposite Y axes."""

import numpy as np


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def finger_pose(config, side, angle_deg, separation_m=None, height_offset_m=0.):
    # side 0 = left (-X), side 1 = right (+X); positive angle closes both.
    sign = 1 if side == 0 else -1
    yaw = np.deg2rad(config.finger_yaw_deg + (180 if side == 1 else 0))
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = rotation_y(sign * np.deg2rad(angle_deg)) @ yaw_rotation
    separation = config.base_separation_m if separation_m is None else separation_m
    position = np.array([-sign * separation / 2, 0, config.base_height_m + height_offset_m])
    return position, rotation


def transform_points(points, position, rotation):
    return points @ rotation.T + position


def rotation_rpy_degrees(rpy):
    roll, pitch, yaw = np.deg2rad(rpy)
    cr, sr, cy, sy = np.cos(roll), np.sin(roll), np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ rotation_y(pitch) @ np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def camera_pose(position, target):
    """Return camera-to-world matrix, optical convention +X right, +Y down, +Z forward."""
    position, target = np.asarray(position), np.asarray(target)
    forward = target - position
    length = np.linalg.norm(forward)
    if length < 1e-9:
        raise ValueError("Camera position and target coincide")
    forward /= length
    # Looking up from the gripper bottom: image top corresponds to world +Y.
    up = np.array([0., 1., 0.])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([1., 0., 0.])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    return pose


def camera_intrinsics(camera, focal_length_mm):
    focal = camera.width * focal_length_mm / camera.sensor_width_mm
    return np.array([[focal, 0, (camera.width - 1) / 2],
                     [0, focal, (camera.height - 1) / 2], [0, 0, 1.]])


def mounted_camera_pose(camera, finger_position, finger_rotation):
    """The camera has exactly the same local extrinsics in both soft fingers."""
    roll, pitch, yaw = np.deg2rad(camera.mount_rpy_deg)
    cr, sr, cy, sy = np.cos(roll), np.sin(roll), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rotation = rz @ rotation_y(pitch) @ rx @ np.diag([-1., -1., 1.])
    result = np.eye(4)
    result[:3, :3] = finger_rotation @ rotation
    result[:3, 3] = finger_position + finger_rotation @ np.asarray(camera.position_in_finger_m)
    return result


def matrix_to_quat_wxyz(matrix):
    """Stable rotation-matrix conversion, independent of either simulation SDK."""
    m = np.asarray(matrix)
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1) * 2
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[i + 1] = s / 4
        q[j + 1] = (m[j, i] + m[i, j]) / s
        q[k + 1] = (m[k, i] + m[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)
