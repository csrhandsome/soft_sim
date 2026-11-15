"""Shared geometry-aware soft-finger grasp candidate generation."""

from pathlib import Path
import json
import numpy as np
import trimesh

from soft_sim.config import ROOT
from soft_sim.mesh import load_stl


def section_radius(vertices, faces, height):
    tri = vertices[faces]
    intersections = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = tri[:, i], tri[:, j]
        valid = ((a[:, 2] - height) * (b[:, 2] - height) <= 0) & (
            np.abs(b[:, 2] - a[:, 2]) > 1e-10
        )
        t = (height - a[valid, 2]) / (b[valid, 2] - a[valid, 2])
        intersections.extend(a[valid] + t[:, None] * (b[valid] - a[valid]))
    return float(np.asarray(intersections)[:, 0].max())


def surface_contacts(mesh, tcp, closing_axis):
    """Opposed first hits on actual triangles, including concave silhouettes."""
    first, _, left_tri = mesh.ray.intersects_location(
        [tcp - closing_axis * 0.2], [closing_axis], multiple_hits=False
    )
    last, _, right_tri = mesh.ray.intersects_location(
        [tcp + closing_axis * 0.2], [-closing_axis], multiple_hits=False
    )
    if not len(first) or not len(last):
        return None
    contacts = np.stack([first[0], last[0]])
    normals = mesh.face_normals[[left_tri[0], right_tri[0]]].copy()
    if normals[0] @ closing_axis > 0:
        normals[0] *= -1
    if normals[1] @ closing_axis < 0:
        normals[1] *= -1
    return contacts, normals


def centered_closure(
    contacts, closing_axis, radius, offset, indent, minimum_separation
):
    """Place the two surfaces around the measured contacts with equal preload."""
    center = np.asarray(contacts).mean(axis=0)
    width = float((contacts[1] - contacts[0]) @ closing_axis)
    closed = max(
        minimum_separation - 2 * offset, width + 2 * radius - 2 * offset - 2 * indent
    )
    return center, width, closed


def generate_candidates(
    scene_dir,
    out,
    *,
    contact_height=None,
    friction=0.7,
    indent=0.003,
):
    """Generate table-clear antipodal grasp candidates for an exported scene."""
    scene_dir = Path(scene_dir).resolve()
    out = Path(out).resolve()
    scene = json.loads((scene_dir / "scene.json").read_text())
    geometry = np.load(scene_dir / "object_geometry.npz")
    contact_mesh = trimesh.Trimesh(geometry["vertices"], geometry["faces"], process=True)
    contact_mesh.update_faces(contact_mesh.unique_faces())
    contact_mesh.remove_unreferenced_vertices()
    object_mesh = contact_mesh.convex_hull
    center = object_mesh.bounds.mean(axis=0)
    extents = np.ptp(object_mesh.bounds, axis=0)

    from soft_sim.robots.mechanism import mechanism
    from soft_sim.robots.fingers import contact_geometry, tool_meshes

    arm = mechanism(scene.get("robot_name"))
    if arm.missing_assets():
        raise SystemExit(f"Missing {arm.name} assets: {', '.join(arm.missing_assets())}")
    installation = scene.get("finger_installation")
    if installation is None:
        raise SystemExit(
            "Scene has no measured finger installation; re-export it on the measured mounts"
        )
    scale = scene.get("finger_scale", 1.0)
    vertices, faces = load_stl(
        ROOT / scene["finger_mesh"], scene.get("finger_scale_to_m", 0.001)
    )
    height = contact_height if contact_height is not None else 0.055 * scale
    radius = section_radius(vertices, faces, height)
    mount, offset = scene["hand_mount_offset_m"], scene["adapter_offset_m"]
    radius, offset, rest_gap = contact_geometry(scale, installation, height, arm)
    mounted_vertices = [
        [
            mesh.vertices.copy()
            for mesh in tool_meshes(scale, side, installation, mechanism=arm).values()
        ]
        for side in (0, 1)
    ]
    table_z = scene["table_z"]
    out.mkdir(parents=True, exist_ok=True)
    hand_mesh = trimesh.load(arm.hand_mesh, force="mesh")
    hand_vertices = np.asarray(hand_mesh.vertices)

    tool_in_hand = arm.rotation
    if "tool_origin_in_hand_m" in scene:
        tcp_origin_m = np.asarray(scene["tool_origin_in_hand_m"], dtype=float)
        tool_forward_m = float(installation.get("forward_offset_m", 0.0))
    else:
        tcp_origin_m = np.array([0.0, 0.0, arm.tool_origin_m[2]])
        tool_forward_m = float(mount) - float(tcp_origin_m @ arm.growth_axis_in_hand)
    hand_to_tcp_m = (
        float(tcp_origin_m @ arm.growth_axis_in_hand) + tool_forward_m + height
    )

    yaw_step = 15 if float(min(extents[:2])) < 0.015 else 45
    long_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    axial_offsets = (0.0,)
    if scene.get("task") == "insert_tubes" and len(contact_mesh.vertices) >= 3:
        xy_centered = contact_mesh.vertices[:, :2] - contact_mesh.vertices[:, :2].mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(np.cov(xy_centered.T))
        principal = eigvecs[:, int(np.argmax(eigvals))]
        long_axis = np.array([float(principal[0]), float(principal[1]), 0.0], dtype=float)
        span = float(np.ptp((contact_mesh.vertices - center) @ long_axis))
        if span > 0.04:
            max_shift = 0.4 * span
            axial_offsets = tuple(
                dict.fromkeys(
                    round(float(np.clip(shift, -max_shift, max_shift)), 4)
                    for shift in (0.0, -0.025, 0.025, -0.040, 0.040)
                )
            )
    candidates = []
    contact_cache = {}
    for yaw in np.arange(0, 360, yaw_step):
        c = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0.0])
        for direction in (1, -1):
            horizontal = direction * np.array([c[1], -c[0], 0.0])
            for tilt in (
                0,
                30,
                45,
                60,
                70,
                75,
                78,
                80,
                82,
                82.5,
                82.7,
                83,
                84,
                85,
                86,
                88,
            ):
                a = np.sin(np.deg2rad(tilt)) * horizontal + np.array(
                    [0.0, 0.0, -np.cos(np.deg2rad(tilt))]
                )
                r = np.column_stack([c, np.cross(a, c), a])
                hand_r = r @ tool_in_hand.T
                upper_z = max(0.0, float(object_mesh.bounds[1][2] - center[2]) - 0.0015)
                zoffsets = tuple(
                    dict.fromkeys(
                        (0.0, 0.001, 0.002, 0.003, 0.006, 0.009, upper_z * 0.5, upper_z)
                    )
                )
                for axial_offset in axial_offsets:
                    for zoffset in zoffsets:
                        tcp = center + long_axis * axial_offset + [0, 0, zoffset]
                        key = (float(yaw), float(axial_offset), float(zoffset))
                        if key not in contact_cache:
                            contact_cache[key] = surface_contacts(contact_mesh, tcp, c)
                        if contact_cache[key] is None:
                            continue
                        contacts, normals = contact_cache[key]
                        antipodal = min(-float(normals[0] @ c), float(normals[1] @ c))
                        if antipodal < 1.0 / np.sqrt(1.0 + friction**2):
                            continue
                        minimum_separation = 2 * offset
                        raw_width = float((contacts[1] - contacts[0]) @ c)
                        eff_indent = float(indent)
                        if raw_width + 2 * radius - 2 * offset - 2 * eff_indent <= 0:
                            eff_indent = min(
                                eff_indent,
                                max(0.001, (raw_width + 2 * radius - 2 * offset) * 0.25),
                            )
                        tcp, width, closed = centered_closure(
                            contacts, c, radius, offset, eff_indent, minimum_separation
                        )
                        opened = arm.max_open_width_m
                        if (
                            width + 2 * radius - closed - 2 * offset
                        ) / 2 < 0.00075 or not 0 < closed < opened:
                            continue
                        hand_pos = tcp - hand_r @ (
                            tcp_origin_m
                            + arm.growth_axis_in_hand * (tool_forward_m + height)
                        )
                        minimum = float((hand_vertices @ hand_r.T + hand_pos)[:, 2].min())
                        for jaw in (opened, closed):
                            for side in (0, 1):
                                for points in mounted_vertices[side]:
                                    local = points + [
                                        (-1 if side == 0 else 1) * jaw / 2,
                                        0.0,
                                        -height,
                                    ]
                                    minimum = min(
                                        minimum, float((local @ r.T + tcp)[:, 2].min())
                                    )
                        clearance = minimum - table_z
                        if clearance < 0.002:
                            continue
                        candidates.append(
                            dict(
                                tcp=tcp.tolist(),
                                rotation=r.tolist(),
                                opened=opened,
                                closed=closed,
                                contact_source="source_triangle_mesh",
                                contacts_world_m=contacts.tolist(),
                                normals_world=normals.tolist(),
                                width=width,
                                tilt_deg=tilt,
                                yaw_deg=float(yaw),
                                axial_offset_m=float(axial_offset),
                                clearance_m=clearance,
                                antipodal=antipodal,
                                score=float(
                                    antipodal * 10 - abs(zoffset) * 200 - tilt * 0.005
                                ),
                            )
                        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    (out / "candidates.json").write_text(json.dumps(candidates, indent=2))
    print(
        "GEOMETRIC_CANDIDATES", len(candidates), "contact_radius_m", radius, flush=True
    )
    if not candidates:
        raise RuntimeError(
            "No table-clear antipodal candidates for the current geometry"
        )
    return dict(
        scene=scene,
        contact_mesh=contact_mesh,
        object_mesh=object_mesh,
        arm=arm,
        installation=installation,
        scale=scale,
        height=height,
        radius=radius,
        offset=offset,
        rest_gap=rest_gap,
        tcp_origin_m=tcp_origin_m,
        tool_forward_m=tool_forward_m,
        hand_to_tcp_m=hand_to_tcp_m,
        candidates=candidates,
    )
