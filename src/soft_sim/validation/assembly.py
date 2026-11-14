"""Audit saved full-travel FEM samples against fixture, wrist, table and blocks.

Sampled surface intersections are reported as-is. Own wedge/base attachment and
opposed soft/soft contact are intentional; this is not a force or task test.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import trimesh
from mplib.collision_detection import fcl

from soft_sim.validation.installation import collision_object, wrist_obstacles
from soft_sim.robots.x5.mount_geometry import fixture_meshes
from soft_sim.robots.x5.soft_tool import finger_in_link, link_transforms
from soft_sim.mesh import surface_faces
from soft_sim.validation.fem_projection import projected_depth


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    folder = args.run / "assembly_motion"
    motion = json.loads((folder / "result.json").read_text())
    if not motion["valid"]:
        raise ValueError("Complete and pass the native-drive FEM sweep before auditing its collisions")
    scene = json.loads((args.run / "scene.json").read_text())
    rows = json.loads((folder / "samples.json").read_text())
    installation, scale = scene["finger_installation"], scene["finger_scale"]
    reference = np.load(folder / "reference.npz")
    boundaries = {(index, side): surface_faces(reference[f"robot{index}_tets_{side}"])
                  for index in (0, 1) for side in (0, 1)}
    wrist_meshes = wrist_obstacles()
    table = trimesh.creation.box(extents=scene["table_size_m"])
    table.apply_translation(scene["table_center_m"])
    fixed = {"table": collision_object(table)}
    for label, entry in scene["task_objects"].items():
        with np.load(args.run / entry["geometry_file"]) as mesh:
            fixed[label] = collision_object(trimesh.Trimesh(mesh["vertices"], mesh["faces"], process=False).convex_hull)
    minima, intersections, expected_contacts = {}, [], Counter()
    for row in rows:
        with np.load(folder / row["state_file"]) as state:
            moving, wrists = {}, {}
            for index in (0, 1):
                mounts = state[f"robot{index}_mounts"]
                links = link_transforms(state[f"robot{index}_q"])
                world_wrist = mounts[0] @ np.linalg.inv(finger_in_link(0, installation)) @ np.linalg.inv(
                    np.linalg.inv(links["link6"]) @ links["link8"])
                for name, mesh in wrist_meshes.items():
                    mesh = mesh.copy()
                    mesh.apply_transform(world_wrist)
                    wrists[f"robot{index}_{name}"] = collision_object(mesh)
                for side in (0, 1):
                    nodes = state[f"robot{index}_nodes_{side}"]
                    moving[(index, side, "finger")] = collision_object(trimesh.Trimesh(
                        nodes, boundaries[index, side], process=False))
                    world_link = mounts[side] @ np.linalg.inv(finger_in_link(side, installation))
                    for name, mesh in fixture_meshes(scale, side, installation).items():
                        if f"robot{index}_fixtures" in state.files:
                            paths = reference[f"robot{index}_fixture_paths"].tolist()
                            parent = ("link8", "link7")[side]
                            slot = paths.index(f"/World/envs/env_0/robot{index}/{parent}/SoftFixtures/{name}")
                            pose = state[f"robot{index}_fixtures"][slot] @ np.linalg.inv(finger_in_link(side, installation))
                        else:
                            pose = world_link
                        mesh.apply_transform(pose)
                        moving[(index, side, name)] = collision_object(mesh)

            def check(name, a, b, expected=False):
                distance = float(fcl.distance(a, b).min_distance)
                minima[name] = min(minima.get(name, float("inf")), distance)
                if fcl.collide(a, b).is_collision():
                    if expected:
                        expected_contacts[name] += 1
                    else:
                        intersections.append(dict(time_s=row["time_s"], phase=row["phase"], pair=name))

            entries = list(moving.items())
            for position, (key, a) in enumerate(entries):
                index, side, component = key
                label = f"{index}_{side}_{component}"
                for name, b in {**fixed, **wrists}.items():
                    check(label + "/" + name, a, b)
                for other_key, b in entries[position+1:]:
                    other_index, other_side, other_component = other_key
                    if index == other_index and side == other_side and {component, other_component} == {"adapter", "finger"}:
                        continue
                    expected = index == other_index and side != other_side and component == other_component == "finger"
                    check(label + "/" + "_".join(map(str, other_key)), a, b, expected)
        if len(intersections):
            print("ASSEMBLY_SURFACE_INTERSECTIONS", row["state_file"], len(intersections), flush=True)
    depth_checks = []
    from soft_sim.recording.reader import first_existing
    preview = first_existing(args.run/"previews/initial/preview_depth.npz",
                             args.run/"preview_depth.npz") if (
        (args.run/"previews/initial/preview_depth.npz").exists() or (args.run/"preview_depth.npz").exists()) else None
    for path in [*([preview] if preview else []), *folder.glob("*_depth.npz")]:
        if not path.exists():
            continue
        with np.load(path) as frame:
            height, width = frame["depth"].shape[1:]
            x, y = np.meshgrid(np.arange(8, width-8, 8), np.arange(8, height-8, 8))
            pixels = np.column_stack([x.ravel(), y.ravel()])
            for side, name in enumerate(("left", "right")):
                expected = projected_depth(frame["nodes_"+name], boundaries[scene["active_arm"], side],
                    frame["cameras"][side], frame["intrinsics"][side], pixels)
                near = np.isfinite(expected) & (expected>.001) & (expected<.02)
                actual = frame["depth"][side, pixels[near,1], pixels[near,0]]
                agreement = float(np.mean(np.isfinite(actual) & (np.abs(actual-expected[near])<.002)))
                depth_checks.append(dict(file=path.name, camera=name, rays=int(near.sum()),
                    agreement=agreement, passed=bool(near.sum()>=50 and agreement>.85)))
    report = dict(scope=__doc__.strip(), samples=len(rows), sampling_period_s=.1,
        valid=not bool(intersections) and all(x["passed"] for x in depth_checks),
        camera_fem_depth_checks=depth_checks, unexpected_surface_intersections=intersections,
        expected_opposed_soft_contact_samples=dict(expected_contacts), minimum_pair_distances_m=minima,
        fixture_pose_source="Recorded native rigid-body views" if "robot0_fixture_paths" in reference.files else "Reconstructed FEM mount poses (legacy)",
        limitations="Deformed geometry checked at saved samples, not a continuous FEM collision proof; table and original task blocks included, other scene furniture excluded. No contact-force conclusion.")
    (folder / "collision_validation.json").write_text(json.dumps(report, indent=2))
    print("ASSEMBLY_COLLISION_RESULT", report["valid"], "samples", len(rows), "unexpected", len(intersections))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
