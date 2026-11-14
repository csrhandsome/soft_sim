"""Measured X5 replacement geometry, shared by physics and motion planning.

SI units, column-vector poses. Fixture meshes are simulation CAD proxies; bolt
patterns and real camera packaging still require a hardware drawing.
"""
from functools import lru_cache
import hashlib

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from shared.config import ROOT


@lru_cache(maxsize=1)
def source_mesh():
    mesh = trimesh.load(ROOT / "data/assetes/soft_finger.stl", force="mesh")
    mesh.apply_scale(.001)
    return mesh


def measured_installation(scale, tilt_deg=None, gap_m=.0002):
    if not np.isfinite(scale) or not .25 <= scale <= 1:
        raise ValueError("Finger scale must be in [0.25, 1]")
    mesh = source_mesh()
    # Large exterior +X panels, excluding the base and rounded distal lip.
    mask = ((mesh.triangles_center[:, 0] > 0.)
            & (mesh.face_normals[:, 0] > .95) & (mesh.face_normals[:, 2] > .15)
            & (mesh.triangles_center[:, 2] > .015))
    normal = np.average(mesh.face_normals[mask], axis=0, weights=mesh.area_faces[mask])
    measured = float(np.rad2deg(np.arctan2(normal[2], normal[0])))
    tilt = measured if tilt_deg is None else float(tilt_deg)
    if not np.isfinite(tilt) or not 0 <= tilt <= 20 or not np.isfinite(gap_m) or gap_m <= 0:
        raise ValueError("Expected a finite 0–20 degree tilt and positive rest gap")
    rotation = Rotation.from_euler("y", tilt, degrees=True).as_matrix()
    # Use the WHOLE mesh support, including the rounded lip and base, rather
    # than assuming that the selected panel planes bound the entire finger.
    half = float((mesh.vertices @ rotation.T)[:, 0].max() * scale + gap_m / 2)
    base = mesh.vertices[mesh.vertices[:, 2] < 1e-8]
    # A tilted base extends back toward the palm. Put its rearmost corner
    # 2 mm ahead of the unchanged slide station; the wedge bottom stays there.
    forward = float(-(base @ rotation.T)[:, 2].min() * scale + .002)
    return dict(profile="measured_wedge_v3", scale=float(scale), tilt_deg=tilt,
                measured_panel_tilt_deg=measured, closed_half_spacing_m=half,
                forward_offset_m=forward,
                seat_width_m=.024,
                camera_aperture_m=.009, fixture_follow="native_slide_collision_shapes_v1",
                minimum_rest_gap_m=float(gap_m), fixture="wedge",
                source_sha256=hashlib.sha256((ROOT / "data/assetes/soft_finger.stl").read_bytes()).hexdigest())


def validate_installation(installation, scale):
    """Refuse stale source/scale/derived offsets before building a planner."""
    if installation is None:
        return
    expected = measured_installation(scale, installation["tilt_deg"], installation["minimum_rest_gap_m"])
    if set(expected) != set(installation):
        raise ValueError("Installation metadata does not match the measured mount schema")
    for key, value in expected.items():
        actual = installation[key]
        valid = (isinstance(actual, (int, float)) and np.isclose(actual, value, rtol=0., atol=1e-12)) if isinstance(value, float) else actual == value
        if not valid:
            raise ValueError(f"Installation {key} differs from the mesh/scale-derived mount")


def installation_in_link(side, installation):
    # link8 is -Y, link7 +Y; both parent frames have identical axes.
    from benchmarks.x5_soft_tool import TOOL_ROTATION
    pose = np.eye(4)
    sign = -1 if side == 0 else 1
    mirror = np.eye(3) if side == 0 else np.diag([-1., -1., 1.])
    pose[:3, :3] = TOOL_ROTATION @ mirror @ Rotation.from_euler(
        "y", installation["tilt_deg"], degrees=True).as_matrix()
    # Preserve the small factory origin asymmetry, centering the closed pair.
    origin_y = -.0249 if side == 0 else .024896
    pose[1, 3] = sign * installation["closed_half_spacing_m"] - origin_y
    pose[0, 3] = installation["forward_offset_m"]
    pose[2, 3] = .00000003 if side == 0 else -.00000003
    return pose


def fixture_meshes(scale, side, installation):
    """Return meshes in the *slide link frame*, including the tilted housing."""
    from benchmarks.x5_soft_tool import finger_in_link
    mount = finger_in_link(side, installation)
    if installation is None:
        adapter = trimesh.creation.box(extents=np.array([.04, .04, .0166]) * scale)
        adapter.apply_translation([0., 0., -.0083 * scale])
        adapter.apply_transform(mount)
    else:
        # Tilted top follows the complete base footprint. Flat bottom mates
        # with the unchanged slide, with a lateral bridge to its center.
        top = np.array([[x, y, 0.] for x in (-.020, .020) for y in (-.020, .020)]) * scale
        top = top @ mount[:3, :3].T + mount[:3, 3]
        # The native seat is not scaled with the soft finger. A 24 mm-wide
        # compact footprint stays within the stock link8 root's lateral span;
        # copying the tilted top width here creates an unnecessarily wide foot.
        bottom = np.array([[0., y, z] for y in (-installation["seat_width_m"]/2, installation["seat_width_m"]/2)
                           for z in (-.020*scale, .020*scale)])
        adapter = trimesh.convex.convex_hull(np.vstack([top, bottom]))
    camera = trimesh.creation.box(extents=np.array([.012, .012, .008]) * scale)
    camera.apply_translation([0., 0., .006 * scale])
    camera.apply_transform(mount)
    return dict(adapter=adapter, camera=camera)


def camera_housing_surface(scale, aperture_m=.009):
    """Open optical case in finger axes; collision uses its conservative box.

    Keep the established optical center at Z=6 mm. Four walls and a rear plate
    leave the entire configured camera frustum clear through the front opening.
    """
    outer, half_open = .006, aperture_m / 2
    wall = outer - half_open
    pieces = []
    for axis in (0, 1):
        for sign in (-1, 1):
            size = np.array([.012, .012, .008])
            size[axis] = wall
            center = np.array([0., 0., .006])
            center[axis] = sign * (outer-wall/2)
            box = trimesh.creation.box(extents=size*scale)
            box.apply_translation(center*scale)
            pieces.append(box)
    back = trimesh.creation.box(extents=np.array([.009, .009, .001])*scale)
    back.apply_translation(np.array([0., 0., .0025])*scale)
    return trimesh.util.concatenate([*pieces, back])


def assembly_meshes(scale, side, installation):
    from benchmarks.x5_soft_tool import finger_in_link
    finger = source_mesh().copy()
    finger.apply_scale(scale)
    finger.apply_transform(finger_in_link(side, installation))
    return dict(finger=finger, **fixture_meshes(scale, side, installation))


def tool_meshes(scale, side, installation, jaw=0.):
    """Assembly in wrist-oriented tool axes, zero at the slide station."""
    from benchmarks.x5_soft_tool import link_transforms, FINGER_LINKS, TOOL_ROTATION, MOUNT_X, MOUNT_Z
    links = link_transforms(np.r_[np.zeros(6), jaw / 2, jaw / 2])
    wrist_from_tool = np.eye(4)
    wrist_from_tool[:3, :3] = TOOL_ROTATION
    wrist_from_tool[:3, 3] = [MOUNT_X + (installation["forward_offset_m"] if installation else 0.), 0., MOUNT_Z]
    pose = np.linalg.inv(wrist_from_tool) @ np.linalg.inv(links["link6"]) @ links[FINGER_LINKS[side]]
    meshes = assembly_meshes(scale, side, installation)
    for mesh in meshes.values():
        mesh.apply_transform(pose)
    return meshes


def contact_geometry(scale, installation, height):
    """Inner surface at tool height; radius is measured from the mounted root."""
    mesh = tool_meshes(scale, 0, installation)["finger"]
    section = mesh.section([0., 0., 1.], [0., 0., height])
    if section is None:
        raise ValueError("Contact height lies outside the mounted finger")
    inner_x = float(section.vertices[:, 0].max())
    half = installation["closed_half_spacing_m"] if installation else .014898
    return half + inner_x, half, -2 * inner_x


class MountedFixtures:
    """Shapes fixed to the original slide bodies; physics owns their motion."""

    def __init__(self, stage, link_paths, scale, installation, finger_paths):
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf
        self.parents, self.paths, self.sides = list(link_paths), [], []
        self.mount_in_link = []
        from benchmarks.x5_soft_tool import finger_in_link
        for side in (0, 1):
            mount = finger_in_link(side, installation)
            self.mount_in_link.append(mount)
            path_root = link_paths[side] + "/SoftFixtures"
            frame = UsdGeom.Xform.Define(stage, path_root)
            frame.AddTransformOp().Set(Gf.Matrix4d(*mount.T.ravel().tolist()))
            for name, mesh in fixture_meshes(scale, side, installation).items():
                mesh.apply_transform(np.linalg.inv(mount))
                path = path_root + "/" + name
                for suffix, surface in (("Collision", mesh), ("Visual", camera_housing_surface(
                        scale, installation["camera_aperture_m"]) if name == "camera" else mesh)):
                    visual = UsdGeom.Mesh.Define(stage, path + "/" + suffix)
                    visual.CreatePointsAttr(surface.vertices.tolist())
                    visual.CreateFaceVertexCountsAttr([3] * len(surface.faces))
                    visual.CreateFaceVertexIndicesAttr(surface.faces.ravel().tolist())
                    visual.CreateSubdivisionSchemeAttr("none")
                    visual.CreateDisplayColorAttr([Gf.Vec3f(.18, .24, .3)])
                    if suffix == "Collision":
                        visual.CreatePurposeAttr("guide")
                        collider = visual.GetPrim()
                UsdPhysics.CollisionAPI.Apply(collider)
                UsdPhysics.MeshCollisionAPI.Apply(collider).CreateApproximationAttr("convexHull")
                collision = PhysxSchema.PhysxCollisionAPI.Apply(collider)
                collision.CreateContactOffsetAttr(.00025)
                collision.CreateRestOffsetAttr(0.)
                UsdPhysics.FilteredPairsAPI.Apply(collider).CreateFilteredPairsRel().SetTargets(
                    [Sdf.Path(finger_paths[side])])
                self.paths.append(path)
                self.sides.append(side)

    def initialize(self, simulation_view):
        self.views = [simulation_view.create_rigid_body_view(path) for path in self.parents]
        if any(view.count != 1 for view in self.views):
            raise RuntimeError("Expected the original two X5 slide bodies")

    def follow(self, mounts):
        # Intentionally no USD or rigid-state writes: these shapes are children
        # of the actuated native bodies, including their rendered Fabric poses.
        if hasattr(self, "views"):
            # Compare at the same pre-step epoch as the measured FEM targets.
            # A post-step native pose has already integrated the next dt.
            self.pre_step_error = self._pose_error(self.poses(), mounts)

    def reset_to_mounts(self, mounts):
        pass

    def poses(self):
        links = []
        for view in self.views:
            data = view.get_transforms().cpu().numpy()[0]
            pose = np.eye(4)
            pose[:3,3] = data[:3]
            pose[:3,:3] = Rotation.from_quat(data[3:]).as_matrix()
            links.append(pose)
        return np.stack([links[side] @ self.mount_in_link[side] for side in self.sides])

    def mount_error(self, mounts):
        result = dict(self.pre_step_error)
        result["post_step_fem_target_offset_m"] = self._pose_error(self.poses(), mounts)["maximum_translation_error_m"]
        return result

    def _pose_error(self, actual, mounts):
        expected = mounts[self.sides]
        return dict(maximum_translation_error_m=float(np.linalg.norm(actual[:,:3,3]-expected[:,:3,3], axis=1).max()),
                    maximum_rotation_matrix_error=float(np.abs(actual[:,:3,:3]-expected[:,:3,:3]).max()))
