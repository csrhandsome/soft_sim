"""All public configuration uses SI units, except explicitly named *_deg fields."""

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MaterialConfig:
    name: str = "Bambu TPU 85A (isotropic initial approximation)"
    density_kg_m3: float = 1180.0
    young_modulus_pa: float = 5.2e6  # TDS Z direction; XY is 6.8 MPa.
    poisson_ratio: float = 0.45  # Assumption, not a published Bambu value.
    friction: float = 0.7  # Assumption; calibrate against the actual object surface.
    damping: float = 0.0001  # Backend mapping is recorded in each episode.
    source: str = "https://cdn.shopify.com/s/files/1/0645/5876/0155/files/Bambu_TPU_85A_Technical_Data_Sheet_a7233668-497b-487f-af8a-3adb49fafaa1.pdf"


@dataclass
class MeshConfig:
    stl: str = "data/assetes/soft_finger.stl"
    scale_to_m: float = 0.001  # STL has no units; measured bounds suggest millimeters.
    tetra_mesh: str = "data/generated/soft_finger.npz"
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
    # Reduced interpolation is the tested preview/data baseline; full DOFs need
    # considerably more solver iterations for MPa-scale TPU and fine meshes.
    mujoco_dof: str = "trilinear"
    mujoco_cellcount: tuple[int, int, int] = (2, 2, 5)
    solver_iterations: int = 100


@dataclass
class ExperimentConfig:
    material: MaterialConfig = field(default_factory=MaterialConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    object: ObjectConfig = field(default_factory=ObjectConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

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
        if self.simulation.scene_preset not in ('plain', 'studio', 'workshop', 'kitchen'):
            raise ValueError('Unknown scene preset')
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
        if self.gripper.profile not in ('robotiq_2f85', 'panda', 'angular'):
            raise ValueError("Profile must be robotiq_2f85, panda or angular")
        max_width = .080 if self.gripper.profile == 'panda' else .085
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
        if self.simulation.mujoco_dof not in ("full", "trilinear", "quadratic"):
            raise ValueError("Unknown MuJoCo flex DOF mode")
        if len(self.simulation.mujoco_cellcount) != 3 or any(not isinstance(x, int) or x < 1 for x in self.simulation.mujoco_cellcount):
            raise ValueError("MuJoCo cellcount requires three positive integers")
        return self

    @classmethod
    def load(cls, path=None):
        config = cls()
        if path:
            raw = json.loads(Path(path).read_text())
            for name, values in raw.items():
                if name not in {f.name for f in fields(cls)} or not isinstance(values, dict):
                    raise ValueError(f"Unknown configuration section: {name}")
                section = getattr(config, name)
                setattr(config, name, type(section)(**{**asdict(section), **values}))
        return config.validate()

    def to_dict(self):
        return asdict(self)


def resolve_asset(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
