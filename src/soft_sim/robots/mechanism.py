"""Parallel prismatic finger slides on a wrist body: X5 and Franka.

Both robots carry two prismatic finger slides in frames parallel to the wrist,
with the closing axis along the slide frame Y. The shared soft finger is mounted
on those slides with the same derived wedge, so one mount convention covers
both robots. Only the numbers below are robot-specific; every transform is
computed from them, never duplicated.

Nothing here imports a simulator: the tables are plain data, the kinematics come
from the upstream URDF snapshots.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soft_sim.config import ROOT

ROBOTS_PATH = ROOT / "third_party/RoboDojo/Assets/Robots"

#: STL frame -> X5 finger slide frame. Tool X closes, tool Z grows.
_X5_TOOL_ROTATION = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))
#: STL frame -> Panda finger slide frame. Panda grows along the slide +Z.
_FRANKA_TOOL_ROTATION = ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))


@dataclass(frozen=True)
class Slide:
    """One prismatic finger slide as authored by the upstream URDF."""
    link: str
    joint: str
    origin_m: tuple[float, float, float]      # in the hand/wrist frame


@dataclass(frozen=True)
class FingerMechanism:
    """Everything the finger mount needs to know about one robot wrist."""
    name: str
    urdf: Path
    base_link: str
    hand_link: str
    arm_joints: tuple[str, ...]
    #: Articulation order the simulator reports (arm first, then both slides).
    joint_names: tuple[str, ...]
    #: Side 0 is the -Y slide, side 1 the +Y slide, matching the STL mirror.
    slides: tuple[Slide, Slide]
    #: TCP reference: midpoint of the two slide stations, in the hand frame.
    tool_origin_m: tuple[float, float, float]
    tool_rotation: tuple[tuple[float, float, float], ...]
    #: Lateral span the wedge foot mates with; the stock finger plate width.
    seat_width_m: float
    jaw_travel_m: float
    hand_mesh: Path
    #: Stock wrist camera links whose visual hull becomes a planning obstacle.
    camera_links: tuple[str, ...]
    self_collision_config: Path
    installation_profile: str
    calibration_robot_type: str
    embodiment: str
    #: How the upstream URDF names its link meshes.
    mesh_naming: str        # "urdf": relative filename | "link_obj": <link>_<visuals|collisions>_<link>.obj

    # -- derived values --------------------------------------------------
    @property
    def rotation(self):
        return np.asarray(self.tool_rotation, dtype=float)

    @property
    def finger_links(self):
        return tuple(slide.link for slide in self.slides)

    @property
    def finger_joints(self):
        return tuple(slide.joint for slide in self.slides)

    @property
    def growth_axis_in_hand(self):
        """STL +Z (finger growth) expressed in the hand frame."""
        return self.rotation[:, 2]

    @property
    def closing_axis_in_hand(self):
        """STL +X (closing direction) expressed in the hand frame."""
        return self.rotation[:, 0]

    @property
    def max_open_width_m(self):
        """Widest jaw the stock slides allow, as a measured jaw width."""
        return 2 * self.jaw_travel_m

    def slide_lateral_m(self, side):
        """Lateral slide station in the hand frame; the finger pose cancels it."""
        return float(self.slides[side].origin_m @ self.closing_axis_in_hand)

    def closed_joint_q(self):
        """Per-slide joint value at the stock fully-open pose, in URDF order."""
        return np.r_[np.zeros(len(self.arm_joints)), self.jaw_travel_m, self.jaw_travel_m]

    @property
    def meshes_dir(self):
        return self.urdf.parent / "meshes"

    def mesh_file(self, link_name, filename, kind):
        """Resolve one URDF mesh reference.

        The X5 URDF points at meshes/ next to itself. Franka's URDF names
        franka_description/... which the RoboDojo snapshot does not ship, so the
        upstream per-link OBJ export is used instead; its geometry equals the
        link frames of Franka.usd.
        """
        if self.mesh_naming == "urdf":
            return self.urdf.parent / filename
        return (self.meshes_dir / f"{link_name}_visuals_{link_name}.obj" if kind == "visual"
                else self.meshes_dir / f"{link_name}_collisions_collisions.obj")

    def missing_assets(self):
        """Paths the upstream snapshot must provide before a run can start."""
        paths = [self.urdf, self.hand_mesh, self.self_collision_config]
        if self.mesh_naming == "link_obj":
            paths.extend(self.mesh_file(slide.link, "", "visual") for slide in self.slides)
        return [str(path) for path in paths if not Path(path).is_file()]


X5 = FingerMechanism(
    name="x5",
    urdf=ROBOTS_PATH / "x5/X5A.urdf",
    base_link="base_link",
    hand_link="link6",
    arm_joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
    joint_names=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"),
    slides=(Slide("link8", "joint8", (.08657, -.0249, -.00024366)),
            Slide("link7", "joint7", (.08657, .024896, -.0002436))),
    tool_origin_m=(.08657, 0., -.00024363),
    tool_rotation=_X5_TOOL_ROTATION,
    seat_width_m=.024,
    jaw_travel_m=.044,
    hand_mesh=ROBOTS_PATH / "x5/meshes/link6.STL",
    camera_links=("camera", "camera_base"),
    self_collision_config=ROBOTS_PATH / "x5/curobo.yml",
    installation_profile="measured_wedge_v3",
    calibration_robot_type="dual_x5_softfinger",
    embodiment="original dual X5 with replacement FEM fingers on native slides",
    mesh_naming="urdf",
)

FRANKA = FingerMechanism(
    name="franka",
    urdf=ROBOTS_PATH / "franka/panda.urdf",
    base_link="panda_link0",
    hand_link="panda_hand",
    arm_joints=tuple(f"panda_joint{index}" for index in range(1, 8)),
    joint_names=tuple(f"panda_joint{index}" for index in range(1, 8))
                + ("panda_finger_joint1", "panda_finger_joint2"),
    # finger_joint1 moves +Y and drives panda_leftfinger, so the -Y side is
    # panda_rightfinger, whose own joint travels along -Y.
    slides=(Slide("panda_rightfinger", "panda_finger_joint2", (0., 0., .0584)),
            Slide("panda_leftfinger", "panda_finger_joint1", (0., 0., .0584))),
    tool_origin_m=(0., 0., .0584),
    tool_rotation=_FRANKA_TOOL_ROTATION,
    seat_width_m=.021,
    jaw_travel_m=.04,
    hand_mesh=ROBOTS_PATH / "franka/meshes/panda_hand_collisions_collisions.obj",
    camera_links=("camera", "camera_base"),
    self_collision_config=ROBOTS_PATH / "franka/curobo.yml",
    installation_profile="derived_wedge_v1",
    calibration_robot_type="panda_softfinger",
    embodiment="original dual-arm layout with one Franka Panda and replacement FEM fingers",
    mesh_naming="link_obj",
)

MECHANISMS = {X5.name: X5, FRANKA.name: FRANKA}
MECHANISM_NAMES = tuple(MECHANISMS)


def mechanism(name):
    if name not in MECHANISMS:
        raise ValueError(f"Unknown finger mechanism {name!r}; choose from {', '.join(MECHANISM_NAMES)}")
    return MECHANISMS[name]


def for_robot_description(robot_entry):
    """Mechanism of one ``env_cfg/robot/*.yml`` entry, by its upstream robot_name."""
    name = robot_entry.get("robot_name")
    if name not in MECHANISMS:
        raise ValueError(f"No soft-finger mechanism for upstream robot {name!r}")
    return MECHANISMS[name]
