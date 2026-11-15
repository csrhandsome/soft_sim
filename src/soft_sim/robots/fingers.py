"""Shared soft finger mounted on a mechanism's parallel prismatic slides.

Everything here is robot-agnostic: the measured wedge, fixture proxies, camera
proxies, forward kinematics and the planner URDF are all derived from the
mechanism tables in :mod:`soft_sim.robots.mechanism`. The X5-bound names stay
available through :mod:`soft_sim.robots.x5` for code and plans that predate the
generalisation.

SI units, column-vector poses.
"""

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from soft_sim.config import ROOT
from soft_sim.mesh import TetMesh
from soft_sim.robots.mechanism import X5, FingerMechanism


@lru_cache(maxsize=1)
def source_mesh():
    import trimesh
    mesh = trimesh.load(ROOT / "assets/meshes/soft_finger.stl", force="mesh")
    mesh.apply_scale(.001)
    return mesh


# -- installation --------------------------------------------------------
def measured_installation(scale, tilt_deg=None, gap_m=.0002, mechanism=X5):
    """Mount numbers derived from the shared STL for one mechanism's seat.

    ``tilt_deg`` and the rest gap are design choices; the closed spacing and
    forward offset follow from the STL footprint and the mechanism's seat.
    """
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
        raise ValueError("Expected a finite 0-20 degree tilt and positive rest gap")
    rotation = Rotation.from_euler("y", tilt, degrees=True).as_matrix()
    # Use the WHOLE mesh support, including the rounded lip and base, rather
    # than assuming that the selected panel planes bound the entire finger.
    half = float((mesh.vertices @ rotation.T)[:, 0].max() * scale + gap_m / 2)
    base = mesh.vertices[mesh.vertices[:, 2] < 1e-8]
    # A tilted base extends back toward the palm. Put its rearmost corner
    # 2 mm ahead of the unchanged slide station; the wedge bottom stays there.
    forward = float(-(base @ rotation.T)[:, 2].min() * scale + .002)
    return dict(profile=mechanism.installation_profile, scale=float(scale), tilt_deg=tilt,
                measured_panel_tilt_deg=measured, closed_half_spacing_m=half,
                forward_offset_m=forward,
                seat_width_m=float(mechanism.seat_width_m),
                camera_aperture_m=.009, fixture_follow="native_slide_collision_shapes_v1",
                minimum_rest_gap_m=float(gap_m), fixture="wedge",
                source_sha256=hashlib.sha256((ROOT / "assets/meshes/soft_finger.stl").read_bytes()).hexdigest())


def validate_installation(installation, scale, mechanism=X5):
    """Refuse stale source/scale/derived offsets before building a planner."""
    if installation is None:
        return
    expected = measured_installation(scale, installation["tilt_deg"], installation["minimum_rest_gap_m"], mechanism)
    if set(expected) != set(installation):
        raise ValueError("Installation metadata does not match the measured mount schema")
    for key, value in expected.items():
        actual = installation[key]
        valid = (isinstance(actual, (int, float)) and np.isclose(actual, value, rtol=0., atol=1e-12)) if isinstance(value, float) else actual == value
        if not valid:
            raise ValueError(f"Installation {key} differs from the mesh/scale-derived mount")


def installation_in_link(side, installation, mechanism=X5):
    """Mount pose of one finger in its slide link frame."""
    rotation = mechanism.rotation
    growth, normal, lateral = rotation[:, 2], rotation[:, 1], rotation[:, 0]
    sign = -1 if side == 0 else 1
    mirror = np.eye(3) if side == 0 else np.diag([-1., -1., 1.])
    pose = np.eye(4)
    pose[:3, :3] = rotation @ mirror @ Rotation.from_euler(
        "y", installation["tilt_deg"], degrees=True).as_matrix()
    # Preserve the small factory origin asymmetry, centering the closed pair.
    origin = mechanism.slide_lateral_m(side)
    pose[:3, 3] = (growth * installation["forward_offset_m"]
                   + lateral * (sign * installation["closed_half_spacing_m"] - origin)
                   + normal * (.00000003 if side == 0 else -.00000003))
    return pose


def fixture_meshes(scale, side, installation, mechanism=X5):
    """Return meshes in the *slide link frame*, including the tilted housing."""
    import trimesh
    mount = finger_in_link(side, installation, mechanism)
    if installation is None:
        adapter = trimesh.creation.box(extents=np.array([.04, .04, .0166]) * scale)
        adapter.apply_translation([0., 0., -.0083 * scale])
        adapter.apply_transform(mount)
    else:
        # Tilted top follows the complete base footprint. Flat bottom mates
        # with the unchanged slide, with a lateral bridge to its center.
        top = np.array([[x, y, 0.] for x in (-.020, .020) for y in (-.020, .020)]) * scale
        top = top @ mount[:3, :3].T + mount[:3, 3]
        # The native seat is not scaled with the soft finger. A compact
        # footprint stays within the stock slide root's lateral span; copying
        # the tilted top width here creates an unnecessarily wide foot.
        bottom = np.array([[0., y, z] for y in (-installation["seat_width_m"] / 2, installation["seat_width_m"] / 2)
                           for z in (-.020 * scale, .020 * scale)])
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
    import trimesh
    outer, half_open = .006, aperture_m / 2
    wall = outer - half_open
    pieces = []
    for axis in (0, 1):
        for sign in (-1, 1):
            size = np.array([.012, .012, .008])
            size[axis] = wall
            center = np.array([0., 0., .006])
            center[axis] = sign * (outer - wall / 2)
            box = trimesh.creation.box(extents=size * scale)
            box.apply_translation(center * scale)
            pieces.append(box)
    back = trimesh.creation.box(extents=np.array([.009, .009, .001]) * scale)
    back.apply_translation(np.array([0., 0., .0025]) * scale)
    return trimesh.util.concatenate([*pieces, back])


def finger_in_link(side, installation=None, mechanism=X5):
    if installation is not None:
        return installation_in_link(side, installation, mechanism)
    pose = np.eye(4)
    sign = -1 if side == 0 else 1
    pose[:3, :3] = mechanism.rotation @ (np.eye(3) if side == 0 else np.diag([-1., -1., 1.]))
    pose[:3, 3] = mechanism.closing_axis_in_hand * (-sign * .010)
    return pose


def assembly_meshes(scale, side, installation, mechanism=X5):
    finger = source_mesh().copy()
    finger.apply_scale(scale)
    finger.apply_transform(finger_in_link(side, installation, mechanism))
    return dict(finger=finger, **fixture_meshes(scale, side, installation, mechanism))


def tool_meshes(scale, side, installation, jaw=0., mechanism=X5):
    """Assembly in wrist-oriented tool axes, zero at the slide station."""
    links = link_transforms(np.r_[np.zeros(len(mechanism.arm_joints)), jaw / 2, jaw / 2],
                            mechanism=mechanism)
    wrist_from_tool = np.eye(4)
    wrist_from_tool[:3, :3] = mechanism.rotation
    wrist_from_tool[:3, 3] = np.asarray(mechanism.tool_origin_m) + mechanism.growth_axis_in_hand * (
        installation["forward_offset_m"] if installation else 0.)
    pose = (np.linalg.inv(wrist_from_tool) @ np.linalg.inv(links[mechanism.hand_link])
            @ links[mechanism.finger_links[side]])
    meshes = assembly_meshes(scale, side, installation, mechanism)
    for mesh in meshes.values():
        mesh.apply_transform(pose)
    return meshes


def contact_geometry(scale, installation, height, mechanism=X5):
    """Inner surface at tool height; radius is measured from the mounted root."""
    mesh = tool_meshes(scale, 0, installation, mechanism=mechanism)["finger"]
    section = mesh.section([0., 0., 1.], [0., 0., height])
    if section is None:
        raise ValueError("Contact height lies outside the mounted finger")
    inner_x = float(section.vertices[:, 0].max())
    half = installation["closed_half_spacing_m"] if installation else .014898
    return half + inner_x, half, -2 * inner_x


# -- kinematics ----------------------------------------------------------
def link_transforms(q, urdf=None, joint_names=None, mechanism=X5):
    """Independent URDF FK, also usable before Isaac articulation initialization."""
    robot = ET.parse(urdf or mechanism.urdf).getroot()
    names = tuple(joint_names or mechanism.joint_names)
    q = np.asarray(q, dtype=float)
    if q.shape != (len(names),):
        raise ValueError(f"Expected {len(names)} joint values for {mechanism.name}, got {q.shape}")
    values = dict(zip(names, q))
    poses = {mechanism.base_link: np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        previous = len(pending)
        for joint in pending[:]:
            parent = joint.find("parent").get("link")
            if parent not in poses:
                continue
            origin = joint.find("origin")
            local = np.eye(4)
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            motion = np.eye(4)
            if joint.get("type") != "fixed":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                value = values[joint.get("name")]
                if joint.get("type") == "prismatic":
                    motion[:3, 3] = axis * value
                else:
                    motion[:3, :3] = Rotation.from_rotvec(axis * value).as_matrix()
            poses[joint.find("child").get("link")] = poses[parent] @ local @ motion
            pending.remove(joint)
        if len(pending) == previous:
            raise ValueError("URDF has an unresolved parent or cycle")
    return poses


def mounts_from_links(world_links, installation=None, mechanism=X5):
    return np.stack([pose @ finger_in_link(side, installation, mechanism)
                     for side, pose in enumerate(world_links)])


def scene_joint_vector(scene, mechanism=X5):
    """Scene joint vector reordered into the mechanism's planning order."""
    q = np.asarray(scene["initial_q"], dtype=float)
    names = tuple(scene.get("joint_names") or mechanism.joint_names)
    if names == mechanism.joint_names:
        return q.copy()
    if set(names) != set(mechanism.joint_names):
        raise ValueError(f"Scene joints {names} do not match the {mechanism.name} mechanism")
    return q[[names.index(name) for name in mechanism.joint_names]]


def mounts_from_q(q, world_from_base, installation=None, mechanism=X5, joint_names=None):
    links = link_transforms(q, joint_names=joint_names, mechanism=mechanism)
    return mounts_from_links([world_from_base @ links[name] for name in mechanism.finger_links],
                             installation, mechanism)


# -- mesh scaling --------------------------------------------------------
def scale_config(config, scale, directory):
    """Scale topology/rest coordinates, support region, and camera geometry together."""
    if not np.isfinite(scale) or not .25 <= scale <= 1.:
        raise ValueError("Finger scale must be in [0.25, 1]")
    mesh = TetMesh.load(config.mesh)
    original = deepcopy(mesh.metadata)
    mesh.vertices *= scale
    for key in ("stl_volume_m3", "volume_m3", "maximum_tet_volume_m3"):
        if key in mesh.metadata:
            mesh.metadata[key] *= scale ** 3
    for key in ("pitch_m", "scale_to_m"):
        mesh.metadata[key] *= scale
    mesh.metadata["stl_bounds_m"] = (np.asarray(mesh.metadata["stl_bounds_m"]) * scale).tolist()
    mesh.metadata.update(uniform_scale=scale, unscaled_source=original)
    path = Path(directory) / "scaled_finger.npz"
    mesh.validate().save(path)
    config.mesh.scale_to_m *= scale
    config.mesh.voxel_pitch_m = original["pitch_m"] * scale
    config.mesh.pin_height_m *= scale
    config.mesh.tetra_mesh = str(path.resolve())
    config.camera.position_in_finger_m = tuple(np.asarray(config.camera.position_in_finger_m) * scale)
    config.camera.near_m *= scale
    return config


# -- planning URDF -------------------------------------------------------
def make_planning_urdf(out, height, scale, installation=None, mechanism=X5):
    """Keep original kinematics; add the same mounts, scaled STL, and explicit TCP."""
    import os
    import trimesh
    import yaml
    validate_installation(installation, scale, mechanism)

    tree = ET.parse(mechanism.urdf)
    robot = tree.getroot()
    for link in robot.findall("link"):
        for tag in ("visual", "collision"):
            for element in link.findall(tag):
                mesh = element.find("geometry/mesh")
                if mesh is None:
                    continue
                resolved = mechanism.mesh_file(link.get("name"), mesh.get("filename"), tag)
                if resolved.is_file():
                    mesh.set("filename", os.path.relpath(resolved, out))
                else:
                    # Stock wrist-camera geometry is visual-only and is not
                    # shipped for every robot; drop it instead of writing a
                    # dangling reference the planner cannot load.
                    link.remove(element)
        if link.get("name") in mechanism.finger_links:
            for element in list(link.findall("collision")) + list(link.findall("visual")):
                link.remove(element)

    def fixed(name, parent, pose, mesh_file=None, mesh_scale=1.):
        link = ET.SubElement(robot, "link", name=name)
        if mesh_file:
            for tag in ("collision", "visual"):
                geometry = ET.SubElement(ET.SubElement(link, tag), "geometry")
                ET.SubElement(geometry, "mesh", filename=os.path.relpath(mesh_file, out),
                              scale=" ".join([str(mesh_scale)] * 3))
        joint = ET.SubElement(robot, "joint", name=name + "_joint", type="fixed")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, pose[:3, 3])),
                      rpy=" ".join(map(str, Rotation.from_matrix(pose[:3, :3]).as_euler("xyz"))))

    for side, parent in enumerate(mechanism.finger_links):
        name = ("soft_left", "soft_right")[side]
        fixed(name, parent, finger_in_link(side, installation, mechanism),
              ROOT / "assets/meshes/soft_finger.stl", .001 * scale)
        for component, mesh in fixture_meshes(scale, side, installation, mechanism).items():
            path = Path(out) / (name + "_" + component + ".obj")
            mesh.export(path)
            fixed(name + "_" + component, parent, np.eye(4), path)
    tcp = np.eye(4)
    tcp[:3, :3] = mechanism.rotation
    tcp[:3, 3] = np.asarray(mechanism.tool_origin_m) + mechanism.growth_axis_in_hand * (
        height + (installation["forward_offset_m"] if installation else 0.))
    fixed("soft_grasp_tcp", mechanism.hand_link, tcp)
    # A stock wrist camera usually has only visual geometry in the URDF.
    # Include its hull as an obstacle on the moving robot when it resolves.
    for link in robot.findall("link"):
        if link.get("name") not in mechanism.camera_links:
            continue
        visual = link.find("visual")
        if visual is None:
            continue
        hull = trimesh.load(Path(out) / visual.find("geometry/mesh").get("filename"), force="mesh").convex_hull
        path = Path(out) / (link.get("name") + "_collision.obj")
        hull.export(path)
        collision = deepcopy(visual)
        collision.tag = "collision"
        collision.find("geometry/mesh").set("filename", path.name)
        link.append(collision)

    urdf_path = Path(out) / f"soft_{mechanism.name}.urdf"
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    srdf = ET.Element("robot", name=robot.get("name"))
    ignores = yaml.safe_load(mechanism.self_collision_config.read_text())["robot_cfg"]["kinematics"]["self_collision_ignore"]
    present = {link.get("name") for link in robot.findall("link")}
    for first, seconds in ignores.items():
        for second in seconds:
            if first in present and second in present:
                ET.SubElement(srdf, "disable_collisions", link1=first, link2=second,
                              reason=f"Upstream {mechanism.name.upper()}")
    for joint in robot.findall("joint"):
        ET.SubElement(srdf, "disable_collisions", link1=joint.find("parent").get("link"),
                      link2=joint.find("child").get("link"), reason="Adjacent")
    ET.ElementTree(srdf).write(Path(out) / f"soft_{mechanism.name}.srdf")
    return urdf_path


# -- physics fixtures ----------------------------------------------------
class MountedFixtures:
    """Shapes fixed to the original slide bodies; physics owns their motion."""

    def __init__(self, stage, link_paths, scale, installation, finger_paths, mechanism=X5):
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf
        self.parents, self.paths, self.sides = list(link_paths), [], []
        self.mount_in_link = []
        for side in (0, 1):
            mount = finger_in_link(side, installation, mechanism)
            self.mount_in_link.append(mount)
            path_root = link_paths[side] + "/SoftFixtures"
            frame = UsdGeom.Xform.Define(stage, path_root)
            frame.AddTransformOp().Set(Gf.Matrix4d(*mount.T.ravel().tolist()))
            for name, mesh in fixture_meshes(scale, side, installation, mechanism).items():
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
            raise RuntimeError("Expected the two actuated finger slide bodies")

    def follow(self, mounts):
        # Intentionally no USD or rigid-state writes: these shapes are children
        # of the actuated native bodies, including their rendered Fabric poses.
        if hasattr(self, "views"):
            # Compare at the same pre-step epoch as the measured FEM targets.
            # A post-step native pose has already integrated the next dt.
            self.pre_step_error = self._pose_error(self.poses(), mounts)
            self._pre_step_error_gpu = None

    def reset_to_mounts(self, mounts):
        pass

    def poses(self):
        links = []
        for view in self.views:
            data = view.get_transforms().cpu().numpy()[0]
            pose = np.eye(4)
            pose[:3, 3] = data[:3]
            pose[:3, :3] = Rotation.from_quat(data[3:]).as_matrix()
            links.append(pose)
        return np.stack([links[side] @ self.mount_in_link[side] for side in self.sides])

    def mount_error(self, mounts):
        errors = getattr(self, "_pre_step_error_gpu", None)
        if errors is None:
            result = dict(self.pre_step_error)
        else:
            values = errors.cpu().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("Nonfinite GPU fixture mount error")
            result = dict(maximum_translation_error_m=float(values[:, 0].max()),
                          maximum_rotation_matrix_error=float(values[:, 1].max()))
        result["post_step_fem_target_offset_m"] = self._pose_error(self.poses(), mounts)["maximum_translation_error_m"]
        return result

    def _pose_error(self, actual, mounts):
        expected = mounts[self.sides]
        return dict(maximum_translation_error_m=float(np.linalg.norm(actual[:, :3, 3] - expected[:, :3, 3], axis=1).max()),
                    maximum_rotation_matrix_error=float(np.abs(actual[:, :3, :3] - expected[:, :3, :3]).max()))
