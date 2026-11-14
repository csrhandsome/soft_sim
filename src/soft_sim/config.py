"""All public configuration uses SI units, except explicitly named *_deg fields."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
import math
import os
from pathlib import Path

# Configuration format marker written into every resolved snapshot.
CONFIG_VERSION = 1
MATERIAL_PRESETS = ("bambu_tpu_75a", "bambu_tpu_85a", "bambu_tpu_90a")

# Repository root. The installed package lives at <root>/src/soft_sim/config.py;
# SOFT_SIM_ROOT only overrides it for unusual deployments.
ROOT = Path(os.environ.get("SOFT_SIM_ROOT") or Path(__file__).resolve().parents[2]).resolve()


def shore_a_to_young_modulus_pa(hardness):
    """Gent (1958) indentation estimate; not a calibrated TPU tensile modulus."""
    if not math.isfinite(hardness) or not 0 < hardness < 100:
        raise ValueError("Shore A hardness must be finite and in (0, 100)")
    return 1e6 * 0.0981 * (56 + 7.66 * hardness) / (0.137505 * (254 - 2.54 * hardness))


@dataclass
class MaterialConfig:
    """SI material parameters; named presets live in assets/materials/*.json."""
    name: str
    shore_a: float
    modulus_basis: str
    color_name: str
    color_srgb: tuple[float, float, float]
    density_kg_m3: float
    young_modulus_pa: float
    poisson_ratio: float
    friction: float
    damping: float
    source: str
    modulus_source: str
    color_source: str
    measured_hardness_shore_a: tuple[float, float] | None
    tensile_strength_xy_pa: float
    elongation_at_break_xy_min: float
    # Visual assumptions, independent of FEM/contact parameters and TDS data.
    roughness: float = 1.0
    specular_reflectivity: float = 0.0

    @classmethod
    def load(cls, preset="bambu_tpu_90a"):
        if preset not in MATERIAL_PRESETS:
            raise ValueError(f"Unknown material preset: {preset}")
        payload = json.loads((ROOT / "assets/materials" / f"{preset}.json").read_text())
        return cls(**payload)

    @property
    def color_linear(self):
        """USD displayColor uses linear RGB; UI color references use sRGB."""
        return tuple(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4
                     for c in self.color_srgb)


@dataclass
class MeshConfig:
    stl: str = "assets/meshes/soft_finger.stl"
    scale_to_m: float = 0.001  # STL has no units; measured bounds suggest millimeters.
    tetra_mesh: str = "assets/meshes/soft_finger_voxel2mm.npz"
    voxel_pitch_m: float = 0.002
    pin_height_m: float = 0.004


@dataclass
class GripperConfig:
    profile: str = "robotiq_2f85"
    base_height_m: float = 0.12
    base_separation_m: float = 0.10
    open_jaw_width_m: float = 0.085
    final_jaw_width_range_m: tuple[float, float] = (0.005, 0.025)
    adapter_offset_m: float = 0.020
    mount_tilt_range_deg: tuple[float, float] = (0.0, 12.0)
    open_angle_deg: float = 0.0
    close_angle_range_deg: tuple[float, float] = (15.0, 28.0)
    symmetric: bool = True
    # Local STL +Z grows up. Optional yaw aligns the cage with the object.
    finger_yaw_deg: float = 0.0
    close_duration_s: float = 1.0
    hold_duration_s: float = 0.3


@dataclass
class CameraConfig:
    width: int = 320
    height: int = 320
    sensor_width_mm: float = 4.8
    focal_length_range_mm: tuple[float, float] = (2.0, 3.6)
    # Identical fixed extrinsics in EACH soft finger's local frame.
    position_in_finger_m: tuple[float, float, float] = (0.0, 0.0, 0.006)
    mount_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    near_m: float = 0.001
    far_m: float = 2.0


@dataclass
class ObjectConfig:
    shapes: tuple[str, ...] = ("sphere", "box", "cylinder")
    radius_range_m: tuple[float, float] = (0.014, 0.022)
    height_range_m: tuple[float, float] = (0.025, 0.045)
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.18)
    lateral_jitter_m: float = 0.002
    mass_kg: float = 0.05
    fixed: bool = True  # Mounted test object for controlled force/deformation data.
    friction: float = 0.7
    isaac_usd_assets: tuple[str, ...] = ()  # e.g. banana, soup_can, or a local USD.
    selection: str = 'random'
    rotation_rpy_range_deg: tuple = ((0., 0.), (0., 0.), (0., 0.))
    collision: bool = True


@dataclass
class SimulationConfig:
    scene_preset: str = 'plain'
    contact_offset_m: float = 0.0005
    rest_offset_m: float = 0.0
    timestep_s: float = 0.001
    capture_every_steps: int = 25
    settle_duration_s: float = 0.05
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    render: bool = True
    headless: bool = True
    livestream: bool = False
    livestream_ip: str = ""
    livestream_signal_port: int = 49100
    livestream_stream_port: int = 47998
    solver_iterations: int = 100
    deformable_vertex_velocity_damping: float | None = None


@dataclass
class TaskConfig:
    """Task identity and embodiment parameters shared by every command."""
    name: str = "general_pickup"
    seed: int = 0
    active_arm: int = 1
    finger_scale: float = 0.75
    target_label: str | None = None
    joint_control: str = "position"
    adapter_offset_m: float = 0.008
    texture_max_resolution: int = 1024
    supported_tasks: tuple[str, ...] = ("general_pickup", "stack_blocks")


@dataclass
class PlanningConfig:
    friction: float = 0.51
    indent_m: float = 0.008
    contact_height_candidates_m: tuple[float, float, float] = (0.073, 0.055, 0.080)
    lift_height_m: float = 0.14
    release_gap_m: float = 0.002
    max_attempts: int = 96
    sample_dt_s: float = 0.02


@dataclass
class RecordingConfig:
    fps: int = 20
    dataset_id: str | None = None
    repo_id: str = "local/softfinger"


@dataclass
class ExperimentConfig:
    material: MaterialConfig = field(default_factory=MaterialConfig.load)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)

    def validate(self):
        def positive(name, value):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")

        def interval(name, value, positive_min=False):
            if len(value) != 2 or not all(math.isfinite(x) for x in value) or value[0] > value[1]:
                raise ValueError(f"Invalid interval: {name}")
            if positive_min and value[0] <= 0:
                raise ValueError(f"{name} must be positive")

        # Reject NaN/Inf anywhere, including optional overrides.
        def finite_tree(value):
            if isinstance(value, dict):
                for child in value.values():
                    finite_tree(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    finite_tree(child)
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Configuration contains NaN or infinity")

        finite_tree(asdict(self))
        if not 0 < self.material.shore_a < 100:
            raise ValueError("Shore A must be in (0, 100)")
        if len(self.material.color_srgb) != 3 or any(not 0 <= c <= 1 for c in self.material.color_srgb):
            raise ValueError("Material sRGB color must have three channels in [0, 1]")
        for name in ("roughness", "specular_reflectivity"):
            if not 0 <= getattr(self.material, name) <= 1:
                raise ValueError(f"Material {name} must be in [0, 1]")
        if self.simulation.scene_preset not in ('plain', 'studio', 'workshop', 'kitchen'):
            raise ValueError('Unknown scene preset')
        if self.simulation.deformable_vertex_velocity_damping is not None and self.simulation.deformable_vertex_velocity_damping < 0:
            raise ValueError('Deformable vertex velocity damping must be nonnegative')
        if not 0 <= self.simulation.rest_offset_m < self.simulation.contact_offset_m:
            raise ValueError('Require 0 <= rest offset < contact offset')
        if self.object.selection not in ('random', 'cycle'):
            raise ValueError('Object selection must be random or cycle')
        if len(self.object.rotation_rpy_range_deg) != 3:
            raise ValueError('Object rotation requires roll, pitch and yaw intervals')
        for pair in self.object.rotation_rpy_range_deg:
            interval('object rotation', pair)
        for port in (self.simulation.livestream_signal_port, self.simulation.livestream_stream_port):
            if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
                raise ValueError("Livestream ports must be integers in [1,65535]")
        if not isinstance(self.simulation.livestream_ip, str):
            raise ValueError("Livestream IP must be a string")
        for name, value in {
            "density": self.material.density_kg_m3, "Young modulus": self.material.young_modulus_pa,
            "mesh scale": self.mesh.scale_to_m, "mesh pitch": self.mesh.voxel_pitch_m,
            "pin height": self.mesh.pin_height_m, "base separation": self.gripper.base_separation_m,
            "close duration": self.gripper.close_duration_s, "timestep": self.simulation.timestep_s,
            "mass": self.object.mass_kg, "near": self.camera.near_m,
        }.items():
            positive(name, value)
        if not 0 < self.material.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio must be in (0, 0.5)")
        for x in (self.material.friction, self.material.damping, self.object.friction,
                  self.object.lateral_jitter_m, self.gripper.hold_duration_s, self.simulation.settle_duration_s):
            if x < 0:
                raise ValueError("Friction, damping, jitter and durations must be nonnegative")
        interval("close angles", self.gripper.close_angle_range_deg)
        if not 0 <= self.gripper.open_angle_deg <= self.gripper.close_angle_range_deg[0] <= self.gripper.close_angle_range_deg[1] < 80:
            raise ValueError("Require 0 <= open <= close_min <= close_max < 80 degrees")
        for name, pair in (("focal length", self.camera.focal_length_range_mm),
                           ("radius", self.object.radius_range_m), ("height", self.object.height_range_m)):
            interval(name, pair, True)
        positive("sensor width", self.camera.sensor_width_mm)
        interval("mount tilt", self.gripper.mount_tilt_range_deg)
        if not 0 <= self.gripper.mount_tilt_range_deg[0] <= self.gripper.mount_tilt_range_deg[1] < 60:
            raise ValueError("Mount tilt must be in [0,60) degrees")
        interval("final jaw width", self.gripper.final_jaw_width_range_m)
        if self.gripper.profile not in ('robotiq_2f85', 'panda', 'angular', 'native_x5'):
            raise ValueError("Profile must be robotiq_2f85, panda, angular or native_x5")
        max_width = {'panda': .080, 'native_x5': .088}.get(self.gripper.profile, .085)
        if not 0 <= self.gripper.final_jaw_width_range_m[0] <= self.gripper.final_jaw_width_range_m[1] <= self.gripper.open_jaw_width_m <= max_width:
            raise ValueError(f"Require 0 <= final jaw width <= open jaw width <= {max_width} m")
        if self.gripper.adapter_offset_m < 0:
            raise ValueError("Adapter offset must be nonnegative")
        if self.camera.far_m <= self.camera.near_m:
            raise ValueError("Invalid camera clipping planes")
        for value in (self.camera.width, self.camera.height, self.simulation.capture_every_steps, self.simulation.solver_iterations):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("Resolution, capture stride and iterations must be positive integers")
        for value in (self.camera.position_in_finger_m, self.camera.mount_rpy_deg,
                      self.object.position_m, self.simulation.gravity_m_s2):
            if len(value) != 3:
                raise ValueError("Positions and gravity must contain three values")
        if (not self.object.shapes and not self.object.isaac_usd_assets) or not set(self.object.shapes) <= {"sphere", "box", "cylinder"}:
            raise ValueError("Object shapes must be sphere, box and/or cylinder")
        if self.task.name not in self.task.supported_tasks:
            raise ValueError(f"Unsupported task: {self.task.name}")
        if self.task.joint_control not in ("position", "kinematic"):
            raise ValueError("Joint control must be position or kinematic")
        if self.task.active_arm not in (0, 1):
            raise ValueError("Active arm must be 0 or 1")
        if not math.isfinite(self.task.finger_scale) or not .25 <= self.task.finger_scale <= 1.:
            raise ValueError("Finger scale must be in [0.25, 1]")
        if not isinstance(self.task.seed, int) or self.task.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        if self.task.texture_max_resolution not in (512, 1024, 2048, 4096):
            raise ValueError("Texture resolution must be one of 512, 1024, 2048, 4096")
        if self.task.adapter_offset_m < 0:
            raise ValueError("Adapter offset must be nonnegative")
        for name, value in {
            "planning friction": self.planning.friction, "planning indent": self.planning.indent_m,
            "planning lift height": self.planning.lift_height_m, "planning sample dt": self.planning.sample_dt_s,
        }.items():
            positive(name, value)
        if self.planning.friction > 1:
            raise ValueError("Planning friction must not exceed 1")
        if self.planning.release_gap_m < 0:
            raise ValueError("Release gap must be nonnegative")
        if not self.planning.contact_height_candidates_m or any(
                not math.isfinite(h) or h <= 0 for h in self.planning.contact_height_candidates_m):
            raise ValueError("Contact height candidates must be finite and positive")
        if not isinstance(self.planning.max_attempts, int) or self.planning.max_attempts <= 0:
            raise ValueError("Planning attempts must be a positive integer")
        if not isinstance(self.recording.fps, int) or self.recording.fps <= 0:
            raise ValueError("Recording FPS must be a positive integer")
        stride = 1 / (self.recording.fps * self.simulation.timestep_s)
        if not math.isclose(round(stride) * self.simulation.timestep_s, 1 / self.recording.fps, rel_tol=0, abs_tol=1e-9) \
                or round(stride) < 1:
            raise ValueError("Recording FPS must divide the physics step rate")
        if self.recording.dataset_id is not None and (
                not isinstance(self.recording.dataset_id, str) or not self.recording.dataset_id.strip()
                or "/" in self.recording.dataset_id or self.recording.dataset_id in (".", "..")):
            raise ValueError("Dataset id must be a single non-empty path segment")
        return self

    def apply_overrides(self, overrides=None):
        """Apply explicit ``section.field`` overrides; arrays replace whole values."""
        for dotted, value in (overrides or {}).items():
            if value is None:
                continue
            parts = dotted.split(".")
            if len(parts) != 2:
                raise ValueError(f"Override must be section.field: {dotted}")
            section_name, field_name = parts
            section = getattr(self, section_name, None)
            if section is None or not is_dataclass(section):
                raise ValueError(f"Unknown configuration section: {section_name}")
            if field_name not in {f.name for f in fields(section)}:
                raise ValueError(f"Unknown configuration field: {dotted}")
            current = getattr(section, field_name)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(section, field_name, value)
        return self

    @classmethod
    def load(cls, path=None, overrides=None, *, material_preset=None):
        config = cls()
        if path:
            raw = json.loads(Path(path).read_text())
            material_values = raw.get("material", {})
            file_preset = material_values.pop("preset", None) if isinstance(material_values, dict) else None
            if file_preset:
                config.material = MaterialConfig.load(file_preset)
            for name, values in raw.items():
                if name not in {f.name for f in fields(cls)} or not isinstance(values, dict):
                    raise ValueError(f"Unknown configuration section: {name}")
                section = getattr(config, name)
                known = {f.name for f in fields(section)}
                unknown = set(values) - known
                if unknown:
                    raise ValueError(f"Unknown fields in {name}: {sorted(unknown)}")
                setattr(config, name, type(section)(**{**asdict(section), **values}))
        if material_preset:
            config.material = MaterialConfig.load(material_preset)
        config.apply_overrides(overrides)
        return config.validate()

    def to_dict(self):
        return asdict(self)


def resolve_asset(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
