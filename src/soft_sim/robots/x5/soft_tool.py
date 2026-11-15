"""X5-bound names for the shared finger math.

The implementation moved to :mod:`soft_sim.robots.fingers`, which takes a
mechanism. These wrappers keep the original X5 signatures for callers and saved
plans that predate the Franka support.
"""

import numpy as np

from soft_sim.robots import fingers
from soft_sim.robots.mechanism import X5

MECHANISM = X5
TOOL_ROTATION = X5.rotation
FINGER_LINKS = X5.finger_links
JOINT_NAMES = list(X5.joint_names)
MOUNT_X = float(X5.tool_origin_m[0])
MOUNT_Z = float(X5.tool_origin_m[2])
HALF_ROOT_SPACING = (abs(X5.slides[1].origin_m[1]) + abs(X5.slides[0].origin_m[1])) / 2
INWARD_OFFSET = .010


def link_transforms(q, urdf=None):
    return fingers.link_transforms(q, urdf=urdf, mechanism=X5)


def mounts_from_links(world_links, installation=None):
    return fingers.mounts_from_links(world_links, installation, X5)


def mounts_from_q(q, world_from_base, installation=None):
    return fingers.mounts_from_q(q, world_from_base, installation, X5)


def finger_in_link(side, installation=None):
    return fingers.finger_in_link(side, installation, X5)


def scale_config(config, scale, directory):
    return fingers.scale_config(config, scale, directory)


def make_x5_urdf(out, height, scale, installation=None):
    return fingers.make_planning_urdf(out, height, scale, installation, X5)
