"""Reproducible episode randomization and smooth closing trajectories."""

from dataclasses import asdict, dataclass
import numpy as np

@dataclass
class EpisodeSpec:
    seed: int
    object_shape: str
    object_radius_m: float
    object_height_m: float
    object_position_m: tuple[float, float, float]
    close_angles_deg: tuple[float, float]
    mount_angles_deg: tuple[float, float]
    final_jaw_width_m: float
    focal_lengths_mm: tuple[float, float]
    object_rpy_deg: tuple[float, float, float] = (0., 0., 0.)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def sample(cls, config, seed):
        rng = np.random.default_rng(seed)
        uniform = lambda interval: float(rng.uniform(*interval))
        left = uniform(config.gripper.close_angle_range_deg)
        right = left if config.gripper.symmetric else uniform(config.gripper.close_angle_range_deg)
        position = np.asarray(config.object.position_m, dtype=float).copy()
        position[:2] += rng.uniform(-config.object.lateral_jitter_m, config.object.lateral_jitter_m, 2)
        tilt = uniform(config.gripper.mount_tilt_range_deg)
        tilts = (tilt, tilt if config.gripper.symmetric else uniform(config.gripper.mount_tilt_range_deg))
        shapes = list(config.object.shapes) + ['usd:' + p for p in config.object.isaac_usd_assets]
        chosen = str(rng.choice(shapes))
        if config.object.selection == 'cycle':
            chosen = shapes[seed % len(shapes)]
        episode = cls(int(seed), chosen, uniform(config.object.radius_range_m),
                   uniform(config.object.height_range_m), tuple(position), (left, right),
                   tilts, uniform(config.gripper.final_jaw_width_range_m),
                   (uniform(config.camera.focal_length_range_mm), uniform(config.camera.focal_length_range_mm)))
        episode.object_rpy_deg = tuple(uniform(pair) for pair in config.object.rotation_rpy_range_deg)
        return episode


def closing_angles(config, episode, time_s):
    fraction = np.clip(time_s / config.close_duration_s, 0, 1)
    smooth = fraction * fraction * (3 - 2 * fraction)
    return np.full(2, config.open_angle_deg) + smooth * (
        np.asarray(episode.close_angles_deg) - config.open_angle_deg)


@dataclass
class GripperState:
    angles_deg: np.ndarray  # Absolute soft-finger tilt; independent of jaw opening.
    jaw_width_m: float
    base_separation_m: float
    height_offset_m: float


def gripper_state(config, episode, time_s):
    fraction = np.clip(time_s / config.close_duration_s, 0, 1)
    smooth = fraction * fraction * (3 - 2 * fraction)
    mounts = np.asarray(episode.mount_angles_deg)
    if config.profile == 'angular':
        return GripperState(mounts + closing_angles(config, episode, time_s),
                            float('nan'), config.base_separation_m, 0.)
    width = (1 - smooth) * config.open_jaw_width_m + smooth * episode.final_jaw_width_m
    height_offset = 0.
    if config.profile == 'robotiq_2f85':
        # Simplified parallel pinch branch, using Menagerie's spring-link vector
        # (55,37.5) mm and driver range 0..0.8 rad. This is NOT a full four-bar
        # constraint solution or an adaptive/wrapping grasp model.
        phase = np.arctan2(.0375, .055)
        length = np.hypot(.0375, .055)
        travel_fraction = (config.open_jaw_width_m - width) / .085
        angle = np.arccos(np.cos(phase) - travel_fraction * (np.cos(phase) - np.cos(phase + .8)))
        height_offset = float(length * (np.sin(angle) - np.sin(phase)))
    return GripperState(mounts, float(width), float(width + 2 * config.adapter_offset_m), height_offset)
