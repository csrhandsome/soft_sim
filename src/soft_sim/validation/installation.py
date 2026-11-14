"""Offline assembly audit: measured tilt, native slide sweep, gaps and fixtures.

This verifies unloaded rigid geometry, not FEM response or task success.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from mplib import Pose
from mplib.collision_detection import fcl

from soft_sim.config import ROOT
from soft_sim.robots.x5.mount_geometry import measured_installation, tool_meshes, source_mesh
from soft_sim.robots.x5.soft_tool import link_transforms, TOOL_ROTATION, MOUNT_X, MOUNT_Z


def collision_object(mesh):
    model = fcl.BVHModel()
    model.begin_model(len(mesh.faces), len(mesh.vertices))
    model.add_sub_model(list(np.asarray(mesh.vertices, dtype=float)), list(mesh.faces.astype(np.int32)))
    model.end_model()
    return fcl.CollisionObject(model)


def wrist_obstacles():
    source = ROOT / "third_party/RoboDojo/Assets/Robots/x5"
    robot = ET.parse(source / "X5A.urdf").getroot()
    links = link_transforms(np.zeros(8))
    result = {}
    for link in robot.findall("link"):
        name = link.get("name")
        if name not in ("link6", "camera", "camera_base"):
            continue
        visual = link.find("visual")
        mesh = trimesh.load(source / visual.find("geometry/mesh").get("filename"), force="mesh")
        if name != "link6":
            mesh = mesh.convex_hull  # same conservative housing as the planner
        local = np.eye(4)
        origin = visual.find("origin")
        if origin is not None:
            local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        mesh.apply_transform(np.linalg.inv(links["link6"]) @ links[name] @ local)
        result[name] = mesh
    return result


def surface_gaps(parts, scale):
    y, z = np.meshgrid(np.linspace(-.013 * scale, .013 * scale, 9),
                       np.linspace(.020 * scale, .080 * scale, 31))
    origins = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
    hits = []
    source = source_mesh()
    exterior_pad = ((source.triangles_center[:, 0] > 0.) & (source.triangles_center[:, 2] > .015)
                    & (source.face_normals[:, 0] > .95) & (source.face_normals[:, 2] > .15))
    for side in (0, 1):
        mesh = parts[side]["finger"]
        directions = np.tile([-1. if side == 0 else 1., 0., 0.], (len(origins), 1))
        points, rays, faces = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        hits.append({int(ray): (point, mesh.face_normals[face], bool(exterior_pad[face]))
                     for point, ray, face in zip(points, rays, faces)})
    shared = sorted(set(hits[0]) & set(hits[1]))
    all_gaps = np.array([hits[1][i][0][0] - hits[0][i][0][0] for i in shared])
    # Rays through lattice openings hit the far wall, not a grasp pad. Keep
    # these coverage misses explicit rather than calling them pad clearance.
    shared = [i for i in shared if hits[0][i][2] and hits[1][i][2]]
    gaps = np.array([hits[1][i][0][0] - hits[0][i][0][0] for i in shared])
    opposed = np.array([-hits[0][i][1] @ hits[1][i][1] for i in shared])
    return dict(rays_per_side=len(origins), paired_facing_pad_hits=len(shared),
                facing_pad_coverage_fraction=len(shared)/len(origins),
                cavity_ray_gap_p95_m=float(np.percentile(all_gaps, 95)),
                pad_selection="First visible exterior +X source panel; source centroid X>0, Z>15mm, normal X>0.95, Z>0.15. Lattice holes/internal walls excluded from pad statistics, included in cavity-ray statistics.",
                gap_min_m=float(gaps.min()), gap_median_m=float(np.median(gaps)),
                gap_p95_m=float(np.percentile(gaps, 95)), gap_max_m=float(gaps.max()),
                opposed_normal_cosine_min=float(opposed.min()),
                opposed_normal_cosine_median=float(np.median(opposed)))


def audit(scale, installation, obstacles, samples=45):
    parts = [tool_meshes(scale, side, installation) for side in (0, 1)]
    wrist_from_tool = np.eye(4)
    wrist_from_tool[:3, :3] = TOOL_ROTATION
    forward = installation["forward_offset_m"] if installation else 0.
    wrist_from_tool[:3, 3] = [MOUNT_X + forward, 0., MOUNT_Z]
    objects = []
    for side in (0, 1):
        group = {}
        for name, mesh in parts[side].items():
            mesh = mesh.copy()
            mesh.apply_transform(wrist_from_tool)
            group[name] = collision_object(mesh)
        objects.append(group)
    fixed = {name: collision_object(mesh) for name, mesh in obstacles.items()}
    minima, collisions = {}, []

    def check(key, a, b, q):
        distance = float(fcl.distance(a, b).min_distance)
        minima[key] = min(minima.get(key, float("inf")), distance)
        if fcl.collide(a, b).is_collision():
            collisions.append(dict(pair=key, slide_m=float(q)))

    for q in np.linspace(0., .044, samples):
        for side in (0, 1):
            for obj in objects[side].values():
                obj.set_pose(Pose([0., (-1 if side == 0 else 1) * q, 0.], [1., 0., 0., 0.]))
        for left, a in objects[0].items():
            for right, b in objects[1].items():
                check(f"left_{left}/right_{right}", a, b, q)
        for side in (0, 1):
            for name, a in objects[side].items():
                for wrist, b in fixed.items():
                    check(f"{side}_{name}/{wrist}", a, b, q)
        if q == 0:
            for side in (0, 1):
                check(f"{side}_camera/finger", objects[side]["camera"], objects[side]["finger"], q)
                check(f"{side}_camera/adapter", objects[side]["camera"], objects[side]["adapter"], q)
    # Cross-jaw parts lie strictly on opposite sides at closure. Native slides
    # only translate outward: this proves clearance for asymmetric motion too.
    left_max = max(m.bounds[1, 0] for m in parts[0].values())
    right_min = min(m.bounds[0, 0] for m in parts[1].values())
    separating_gap = float(right_min - left_max)
    camera_containment = [bool(group["finger"].contains(np.vstack([
        group["camera"].vertices, group["camera"].centroid])).any()) for group in parts]
    # Distance to a fixed obstacle is 1-Lipschitz under slide translation.
    # Every unsampled q is <= half a sample interval from a checked q.
    wrist_margin = min(v for k, v in minima.items() if k.split("/")[1] in fixed) - .044 / (samples - 1) / 2
    forward_extent = max(m.bounds[1, 2] for group in parts for m in group.values()) + forward
    result = dict(scale=scale, installation=installation, native_slide_limits_m=[0., .044],
        symmetric_samples=samples, minimum_pair_distances_m=minima, collisions=collisions,
        continuous_cross_jaw_separating_gap_m=separating_gap,
        continuous_wrist_clearance_lower_bound_m=float(wrist_margin),
        asymmetric_slide_clearance_proven=bool(separating_gap > 0 and wrist_margin > 0),
        camera_housing_points_inside_finger_solid=camera_containment,
        forward_from_factory_slide_m=float(forward_extent),
        working_surface=surface_gaps(parts, scale),
        attachment_contact_excluded="same-side wedge top/finger base; unchanged slide interface",
        camera_proxy="12x12x8 mm times scale; independent physical hardware dimensions pending",
        collision_clearance_valid=bool(not collisions and not any(camera_containment)
                                      and separating_gap > 0 and wrist_margin > 0))
    return result, parts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Report directory; normally reports/installation of a run")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    obstacles = wrist_obstacles()
    reports, drawings = [], []
    configurations = [("parallel_legacy", .75, None)]
    configurations += [(f"tilt_{angle:g}", .75, measured_installation(.75, angle)) for angle in (0, 6, 10, 12, 14, 16)]
    configurations += [(f"measured_scale_{scale:g}", scale, measured_installation(scale)) for scale in (.65, .75, .8)]
    for label, scale, installation in configurations:
        report, parts = audit(scale, installation, obstacles)
        report["label"] = label
        reports.append(report)
        if label in ("parallel_legacy", "tilt_6", "tilt_12", "tilt_16"):
            drawings.append((label, parts, report))
        print(label, "collision_clearance_valid", report["collision_clearance_valid"], "collisions", len(report["collisions"]),
              "surface gap p95 mm", report["working_surface"]["gap_p95_m"] * 1000, flush=True)
    report = dict(scope=__doc__.strip(), results=reports,
        fixture_comparison=dict(direct_angle="Identical finger pose requires a tilted seat machined into/replacing the factory slide interface; no such CAD is available.",
            wedge="Selected: separate flat-bottom/tilted-top bridge retains factory slide frames and limits.",
            manufacturing_status="Simulation envelope only; fastening, tolerances, strength and real camera packaging unverified."),
        physical_execution=dict(status="not_part_of_this_offline_audit", reference="See separate Isaac assembly_motion result and task run results."),
        task_clearance="Table/block approach, release and retreat require scene-specific planning and physical validation.")
    (args.output / "installation.json").write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    fig, axes = plt.subplots(1, len(drawings), figsize=(16, 6), sharex=True, sharey=True)
    stock_links = link_transforms(np.zeros(8))
    tool = np.eye(4)
    tool[:3, :3], tool[:3, 3] = TOOL_ROTATION, [MOUNT_X, 0., MOUNT_Z]
    stock_meshes = []
    for name in ("link8", "link7"):
        mesh = trimesh.load(ROOT / f"third_party/RoboDojo/Assets/Robots/x5/meshes/{name}.STL", force="mesh")
        mesh.apply_transform(np.linalg.inv(tool) @ np.linalg.inv(stock_links["link6"]) @ stock_links[name])
        stock_meshes.append(mesh)
    colors = dict(finger="#f87522", adapter="#64748b", camera="#0c9baa")
    for ax, (label, parts, entry) in zip(axes, drawings):
        for mesh in stock_meshes:
            ax.add_collection(PolyCollection(mesh.triangles[:, :, [0, 2]] * 1000, facecolors="#cbd5e1", edgecolors="none", alpha=.5))
        for group in parts:
            for name, mesh in group.items():
                triangles = mesh.triangles[:, :, [0, 2]].copy()
                if entry["installation"]:
                    triangles[:, :, 1] += entry["installation"]["forward_offset_m"]
                ax.add_collection(PolyCollection(triangles * 1000, facecolors=colors[name], edgecolors="none"))
        ax.set_title(label + f"\nP95 surface gap {entry['working_surface']['gap_p95_m'] * 1000:.2f} mm")
        ax.set_xlim(-38, 38)
        ax.set_ylim(-18, 78)
        ax.set_aspect("equal")
        ax.set_xlabel("Closing axis (mm)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Forward from factory slide (mm)")
    fig.suptitle("Native slides fully closed; stock (gray), soft fingers (orange), wedges (slate), camera proxies (cyan)")
    fig.tight_layout()
    fig.savefig(args.output / "closed_comparison.png", dpi=180)
    fig.savefig(args.output / "closed_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
