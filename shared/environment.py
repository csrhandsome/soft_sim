"""Implement only engine-specific primitives in a subclass; experiments stay shared."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from .experiment import EpisodeSpec, gripper_state
from .geometry import camera_intrinsics, finger_pose, mounted_camera_pose, transform_points
from .mesh import TetMesh


@dataclass
class ForceReading:
    normal_force_n: np.ndarray  # (2,), nonnegative summed normal contact magnitude, object only
    contact_force_world_n: np.ndarray  # (2,3), force applied TO each finger BY the test object
    source: str
    valid: bool  # True only for native object/finger contact labels.
    estimated_force_world_n: np.ndarray  # (2,3), separate estimate, never silently a native label


@dataclass
class Observation:
    time_s: float
    angles_deg: np.ndarray
    jaw_width_m: float
    vertices_world_m: np.ndarray
    displacement_m: np.ndarray  # Relative to undeformed mesh transformed by current base pose.
    object_pose: np.ndarray  # xyz + quaternion wxyz
    force: ForceReading
    rgb: np.ndarray | None  # (2,H,W,3), left and right finger cameras.
    depth_m: np.ndarray | None  # (2,H,W), optical-axis depth in meters.
    camera_intrinsics: np.ndarray  # (2,3,3)
    camera_to_world: np.ndarray  # (2,4,4), moving with finger supports.
    bilateral_contact: bool

    @property
    def squeeze_force_n(self):
        # Mean of the two summed normal magnitudes; do not use opposing vector net force.
        return float(np.mean(self.force.normal_force_n)) if self.force.valid else float('nan')


class BaseGripperEnv(ABC):
    backend_name = 'abstract'

    def __init__(self, config):
        self.config = config.validate()
        self.mesh = TetMesh.load(config.mesh)
        self.pin_mask = self.mesh.vertices[:, 2] <= self.mesh.vertices[:, 2].min() + config.mesh.pin_height_m + 1e-8
        self.episode = None
        self.time_s = 0.0
        self.step_index = 0
        self.angles_deg = np.full(2, config.gripper.open_angle_deg)
        self.state = None

    def finger_pose(self, side):
        return finger_pose(self.config.gripper, side, self.state.angles_deg[side],
                           self.state.base_separation_m, self.state.height_offset_m)

    @property
    def camera_poses(self):
        return np.stack([mounted_camera_pose(self.config.camera, *self.finger_pose(side)) for side in (0, 1)])

    @property
    def intrinsics(self):
        return np.stack([camera_intrinsics(self.config.camera, f) for f in self.episode.focal_lengths_mm])

    def reset(self, seed=0, episode=None):
        self.episode = episode or EpisodeSpec.sample(self.config, seed)
        self.state = gripper_state(self.config.gripper, self.episode, 0.)
        self.angles_deg = self.state.angles_deg.copy()
        self.build_scene(self.episode)
        self.set_gripper_state(self.state)
        for _ in range(round(self.config.simulation.settle_duration_s / self.config.simulation.timestep_s)):
            self.step_physics()
        self.time_s = 0.0
        self.step_index = 0
        return self.observe()

    def step(self):
        if self.episode is None:
            raise RuntimeError('Call reset() before step()')
        self.step_index += 1
        self.time_s = self.step_index * self.config.simulation.timestep_s
        self.state = gripper_state(self.config.gripper, self.episode, self.time_s)
        self.angles_deg = self.state.angles_deg.copy()
        self.set_gripper_state(self.state)
        self.step_physics()

    def observe(self):
        if self.episode is None:
            raise RuntimeError('Call reset() before observe()')
        positions = self.get_vertices_world()
        reference = np.stack([transform_points(self.mesh.vertices, *self.finger_pose(side)) for side in (0, 1)])
        force = self.read_force()
        if not np.isfinite(positions).all():
            raise RuntimeError('Non-finite soft-body state; dataset frame rejected')
        if force.valid and (not np.isfinite(force.normal_force_n).all() or not np.isfinite(force.contact_force_world_n).all()
                            or np.any(force.normal_force_n < 0)):
            raise RuntimeError('Invalid native force data; dataset frame rejected')
        rgb, depth = self.render_camera() if self.config.simulation.render else (None, None)
        if self.config.simulation.render:
            expected = (self.config.camera.height, self.config.camera.width)
            if rgb is None or depth is None or rgb.shape != (2, *expected, 3) or depth.shape != (2, *expected):
                raise RuntimeError('Camera did not produce matching RGB/depth frames')
        return Observation(self.time_s, self.angles_deg.copy(), self.state.jaw_width_m, positions, positions - reference,
                           self.get_object_pose(), force, rgb, depth, self.intrinsics,
                           self.camera_poses, bool(force.valid and np.all(force.normal_force_n > 0.01)))

    def rollout(self, seed=0, episode=None):
        yield self.reset(seed, episode)
        duration = self.config.gripper.close_duration_s + self.config.gripper.hold_duration_s
        count = int(np.ceil(duration / self.config.simulation.timestep_s))
        for index in range(1, count + 1):
            self.step()
            if index % self.config.simulation.capture_every_steps == 0 or index == count:
                yield self.observe()

    @property
    def backend_metadata(self):
        return {}

    @abstractmethod
    def build_scene(self, episode): ...

    @abstractmethod
    def set_gripper_state(self, state): ...

    @abstractmethod
    def step_physics(self): ...

    @abstractmethod
    def get_vertices_world(self): ...

    @abstractmethod
    def get_object_pose(self): ...

    @abstractmethod
    def read_force(self) -> ForceReading: ...

    @abstractmethod
    def render_camera(self): ...

    @abstractmethod
    def close(self): ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
