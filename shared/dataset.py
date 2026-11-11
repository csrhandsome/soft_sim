"""Portable, versioned episode files: NPZ arrays plus human-readable JSON metadata."""

import json
from pathlib import Path
import struct
import zlib

import numpy as np


def write_png(path, rgb):
    """Small dependency-free RGB PNG writer for inspecting the two camera views."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError('Expected RGB image')
    def chunk(kind, data):
        payload = kind + data
        return struct.pack('>I', len(data)) + payload + struct.pack('>I', zlib.crc32(payload) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def save_frame(path, observation, phase):
    obs = observation
    arrays = dict(time_s=np.float64(obs.time_s), angles_deg=obs.angles_deg,
                  jaw_width_m=np.float64(obs.jaw_width_m), object_pose=obs.object_pose,
                  vertices_world_m=obs.vertices_world_m.astype(np.float32), displacement_m=obs.displacement_m.astype(np.float32),
                  normal_force_n=obs.force.normal_force_n, contact_force_world_n=obs.force.contact_force_world_n,
                  squeeze_force_n=np.float64(obs.squeeze_force_n), force_valid=np.bool_(obs.force.valid),
                  force_source=np.array(obs.force.source), estimated_force_world_n=obs.force.estimated_force_world_n,
                  camera_intrinsics=obs.camera_intrinsics, camera_to_world=obs.camera_to_world,
                  bilateral_contact=np.bool_(obs.bilateral_contact), phase=np.array(phase))
    if obs.rgb is not None:
        arrays.update(rgb=obs.rgb, depth_m=obs.depth_m.astype(np.float32))
    np.savez_compressed(path, **arrays)


def collect_episode(env, output, seed, index=0, episode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f'episode_{seed:08d}_{index:05d}'
    destination = output / name
    temporary = output / ('.' + name + '.incomplete')
    if destination.exists() or temporary.exists():
        raise FileExistsError(f'Episode already exists: {destination}; choose a new output or seed')
    temporary.mkdir()
    peak = 0.
    count = 0
    last = None
    for count, obs in enumerate(env.rollout(seed=seed, episode=episode), start=1):
        if count == 1:
            reference = dict(vertices_world_m=obs.vertices_world_m, camera_intrinsics=obs.camera_intrinsics,
                             camera_to_world=obs.camera_to_world, rest_vertices_m=env.mesh.vertices,
                             tets=env.mesh.tets, pin_mask=env.pin_mask)
            if obs.rgb is not None:
                reference.update(rgb=obs.rgb, depth_m=obs.depth_m)
                for side, image in enumerate(obs.rgb):
                    write_png(temporary / f'reference_{side}.png', image)
            np.savez_compressed(temporary / 'reference.npz', **reference)
        phase = 'hold' if obs.time_s >= env.config.gripper.close_duration_s else 'closing'
        save_frame(temporary / f'frame_{count - 1:06d}.npz', obs, phase)
        if obs.force.valid:
            peak = max(peak, obs.squeeze_force_n)
        last = obs
    if last is not None and last.rgb is not None:
        for side, image in enumerate(last.rgb):
            write_png(temporary / f'final_{side}.png', image)
    metadata = dict(schema_version=1, complete=True, backend=env.backend_name, backend_details=env.backend_metadata,
                    config=env.config.to_dict(), episode=env.episode.to_dict(), mesh=env.mesh.metadata,
                    frame_count=count, camera_order=['left', 'right'],
                    units=dict(length='m', force='N', angle='deg', focal_length='mm'),
                    force_definition='squeeze_force_n = mean of per-finger summed object contact normal magnitudes',
                    pose_convention='camera_to_world uses optical +X right +Y down +Z forward; quaternions wxyz',
                    grasp_success=None, grasp_success_note='Bilateral contact is not a verified lift/hold success test.',
                    peak_native_squeeze_force_n=peak if last is not None and last.force.valid else None)
    (temporary / 'episode.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.rename(destination)
    return destination, metadata
