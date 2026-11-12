"""RTX-safe pinhole authoring; physical lens calibration stays in shared config."""
import math


def pinhole_projection_parameters(config, focal_length_mm):
    """Keep f/aperture unchanged while avoiding RTX's small-focal projection clamp.

    USD optical lengths are tenths of a stage unit. Authoring millimeter lenses
    on a meter stage literally gives f=.028 for a 2.8 mm lens. Isaac 5.1 RTX
    renders that with a much narrower FOV than USD's computed frustum. Uniform
    rescaling is equivalent for a pinhole camera (fStop=0), and f=10 works in
    the measured renderer. These attributes are projection parameters, not a
    physical depth-of-field model; use the original mm values for calibration.
    """
    if not math.isfinite(focal_length_mm) or focal_length_mm <= 0:
        raise ValueError("Focal length must be positive and finite")
    focal = 10.
    horizontal = focal * config.sensor_width_mm / focal_length_mm
    return dict(focal_length=focal, horizontal_aperture=horizontal,
                vertical_aperture=horizontal * config.height / config.width,
                physical_focal_length_mm=float(focal_length_mm),
                physical_sensor_width_mm=float(config.sensor_width_mm),
                horizontal_fov_deg=math.degrees(2 * math.atan(config.sensor_width_mm / (2 * focal_length_mm))))


def configure_pinhole_camera(camera, config, focal_length_mm):
    from pxr import Gf

    parameters = pinhole_projection_parameters(config, focal_length_mm)
    camera.CreateFocalLengthAttr(parameters["focal_length"])
    camera.CreateHorizontalApertureAttr(parameters["horizontal_aperture"])
    camera.CreateVerticalApertureAttr(parameters["vertical_aperture"])
    camera.CreateFStopAttr(0.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(config.near_m, config.far_m))
    return parameters
