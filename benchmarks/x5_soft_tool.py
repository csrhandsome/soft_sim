"""Native X5 finger mounts and a uniformly scaled copy of the shared FEM.

The factory link7/link8 slides remain actuated. Only their finger geometry is
replaced. All transforms below use the downloaded, unmodified X5A URDF.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from shared.config import ROOT
from shared.mesh import TetMesh


# Tool X closes along wrist Y; tool Z grows along wrist X.
TOOL_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
FINGER_LINKS = ("link8", "link7")
JOINT_NAMES = [f"joint{i}" for i in range(1, 9)]
MOUNT_X = .08657
MOUNT_Z = -.00024363
HALF_ROOT_SPACING = (.024896 + .0249) / 2
INWARD_OFFSET = .010


def link_transforms(q, urdf=None):
    """Independent URDF FK, also usable before Isaac articulation initialization."""
    robot = ET.parse(urdf or ROOT / "RoboDojo/Assets/Robots/x5/X5A.urdf").getroot()
    values = dict(zip(JOINT_NAMES, np.asarray(q)))
    poses = {"base_link": np.eye(4)}
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


def finger_in_link(side, installation=None):
    if installation is not None:
        from benchmarks.x5_mount_geometry import installation_in_link
        return installation_in_link(side, installation)
    pose = np.eye(4)
    sign = -1 if side == 0 else 1
    pose[:3, :3] = TOOL_ROTATION @ (np.eye(3) if side == 0 else np.diag([-1., -1., 1.]))
    pose[:3, 3] = [0., -sign * INWARD_OFFSET, 0.]
    return pose


def mounts_from_links(world_links, installation=None):
    return np.stack([pose @ finger_in_link(side, installation) for side, pose in enumerate(world_links)])


def mounts_from_q(q, world_from_base, installation=None):
    links = link_transforms(q)
    return mounts_from_links([world_from_base @ links[name] for name in FINGER_LINKS], installation)


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


def make_x5_urdf(out, height, scale, installation=None):
    """Keep original kinematics; add the same mounts, scaled STL, and explicit TCP."""
    import os
    import trimesh
    import yaml
    from benchmarks.x5_mount_geometry import validate_installation
    validate_installation(installation, scale)

    source = ROOT / "RoboDojo/Assets/Robots/x5"
    tree = ET.parse(source / "X5A.urdf")
    robot = tree.getroot()
    for link in robot.findall("link"):
        for mesh in link.findall(".//mesh"):
            mesh.set("filename", os.path.relpath(source / mesh.get("filename"), out))
        if link.get("name") in FINGER_LINKS:
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
    for side, parent in enumerate(FINGER_LINKS):
        name = ("soft_left", "soft_right")[side]
        fixed(name, parent, finger_in_link(side, installation), ROOT / "data/assetes/soft_finger.stl", .001 * scale)
        from benchmarks.x5_mount_geometry import fixture_meshes
        for component, mesh in fixture_meshes(scale, side, installation).items():
            path = Path(out) / (name + "_" + component + ".obj")
            mesh.export(path)
            fixed(name + "_" + component, parent, np.eye(4), path)
    tcp = np.eye(4)
    tcp[:3, :3] = TOOL_ROTATION
    tcp[:3, 3] = [MOUNT_X + height + (installation["forward_offset_m"] if installation else 0.), 0., MOUNT_Z]
    fixed("soft_grasp_tcp", "link6", tcp)
    # The stock wrist camera has only visual geometry in the URDF. Include its
    # hull as an obstacle on the moving robot, in addition to the soft cameras.
    for link in robot.findall("link"):
        if link.get("name") in ("camera", "camera_base"):
            visual = link.find("visual")
            filename = visual.find("geometry/mesh").get("filename")
            hull = trimesh.load(Path(out) / filename, force="mesh").convex_hull
            path = Path(out) / (link.get("name") + "_collision.obj")
            hull.export(path)
            collision = deepcopy(visual)
            collision.tag = "collision"
            collision.find("geometry/mesh").set("filename", path.name)
            link.append(collision)
    tree.write(Path(out) / "soft_x5.urdf", encoding="utf-8", xml_declaration=True)
    srdf = ET.Element("robot", name=robot.get("name"))
    ignores = yaml.safe_load((source / "curobo.yml").read_text())["robot_cfg"]["kinematics"]["self_collision_ignore"]
    for first, seconds in ignores.items():
        for second in seconds:
            ET.SubElement(srdf, "disable_collisions", link1=first, link2=second, reason="Upstream X5")
    for joint in robot.findall("joint"):
        ET.SubElement(srdf, "disable_collisions", link1=joint.find("parent").get("link"),
                      link2=joint.find("child").get("link"), reason="Adjacent")
    ET.ElementTree(srdf).write(Path(out) / "soft_x5.srdf")
    return Path(out) / "soft_x5.urdf"
