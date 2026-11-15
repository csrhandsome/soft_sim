"""X5-bound names for the shared measured-wedge geometry.

The implementation moved to :mod:`soft_sim.robots.fingers`, which takes a
mechanism. These wrappers keep the original X5 signatures.
"""

from soft_sim.robots import fingers
from soft_sim.robots.mechanism import X5

MECHANISM = X5


def source_mesh():
    return fingers.source_mesh()


def measured_installation(scale, tilt_deg=None, gap_m=.0002):
    return fingers.measured_installation(scale, tilt_deg, gap_m, X5)


def validate_installation(installation, scale):
    return fingers.validate_installation(installation, scale, X5)


def installation_in_link(side, installation):
    return fingers.installation_in_link(side, installation, X5)


def fixture_meshes(scale, side, installation):
    return fingers.fixture_meshes(scale, side, installation, X5)


def camera_housing_surface(scale, aperture_m=.009):
    return fingers.camera_housing_surface(scale, aperture_m)


def assembly_meshes(scale, side, installation):
    return fingers.assembly_meshes(scale, side, installation, X5)


def tool_meshes(scale, side, installation, jaw=0.):
    return fingers.tool_meshes(scale, side, installation, jaw, X5)


def contact_geometry(scale, installation, height):
    return fingers.contact_geometry(scale, installation, height, X5)


class MountedFixtures(fingers.MountedFixtures):
    def __init__(self, stage, link_paths, scale, installation, finger_paths):
        super().__init__(stage, link_paths, scale, installation, finger_paths, X5)
