"""Export the live RoboDojo scene to the portable geometry the planner reads.

Only world-frame triangle geometry, poses and the measured installation leave
Isaac Sim; the planning environment never starts Kit.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from soft_sim.execution import layout_target_parameters
from soft_sim.robots.fingers import finger_in_link, fixture_meshes


def world_geometry(stage, path, world_from_root=None):
    """Triangulate every mesh under ``path`` in world coordinates."""
    from pxr import Usd, UsdGeom
    vertices, faces, cursor = [], [], 0
    cache = UsdGeom.XformCache()
    root = stage.GetPrimAtPath(path)
    correction = np.eye(4)
    if world_from_root is not None:
        correction = world_from_root @ np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(root)).T)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        transform = correction @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        points = points @ transform[:3, :3].T + transform[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if np.all(counts == 3):
            faces.append(indices.reshape(-1, 3) + cursor)
        else:
            triangles, start = [], 0
            for count in counts:
                polygon = indices[start:start + count]
                triangles.extend([[cursor + polygon[0], cursor + polygon[i], cursor + polygon[i + 1]]
                                  for i in range(1, count - 1)])
                start += count
            faces.append(np.asarray(triangles, dtype=np.int32))
        vertices.append(points)
        cursor += len(points)
    if not vertices:
        raise ValueError(f"No triangle geometry: {path}")
    return np.concatenate(vertices), np.concatenate(faces).astype(np.int32)


def pose_matrix(pose):
    """Column-vector SE(3) from a (position, quaternion wxyz) upstream pose."""
    from scipy.spatial.transform import Rotation
    position, quaternion = pose
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_quat(np.r_[quaternion[1:], quaternion[0]]).as_matrix()
    return result


class SceneExporter:
    """Write ``scene.json`` plus world-frame NPZ geometry for one operation."""

    def __init__(self, *, env, cfg, layout, layout_path, layout_hash, installation,
                 base, arm_index, mechanisms, installations, fem_report, fem_valid,
                 finger_scale, adapter_offset_m, texture_max_resolution, task_name):
        self.env = env
        self.cfg = cfg
        self.layout = layout
        self.layout_path = Path(layout_path)
        self.layout_hash = layout_hash
        self.installation = installation
        self.base = base
        self.arm_index = arm_index
        self.mechanisms = list(mechanisms)
        self.installations = list(installations)
        self.mechanism = self.mechanisms[arm_index]
        self.fem_report = fem_report
        self.fem_valid = fem_valid
        self.manager = env.scene_manager.layout_manager
        self.finger_scale = finger_scale
        self.adapter_offset_m = adapter_offset_m
        self.texture_max_resolution = texture_max_resolution
        self.task_name = task_name

    # -- helpers ---------------------------------------------------------
    def _numpy_pose(self, pose):
        from soft_sim.sim.robodojo import numpy_pose
        return numpy_pose(pose)

    def _object_pose(self, obj):
        return self._numpy_pose(obj.get_world_pose())

    def _export_object(self, destination, obj, filename):
        vertices, faces = world_geometry(self.env.stage, obj.prim_path,
                                         pose_matrix(self._object_pose(obj)))
        np.savez_compressed(Path(destination) / filename, vertices=vertices, faces=faces)
        return vertices, faces

    # -- main ------------------------------------------------------------
    def export(self, destination, target_label):
        import trimesh
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        env, manager = self.env, self.manager
        target_parameters = layout_target_parameters(self.layout, target_label)
        target_name = manager.get_instance_name(label=target_label, env_idx=0)
        target = manager.get_scene_object(inst_name=target_name, env_idx=0)
        target_path = target.prim_path

        def object_pose():
            return self._object_pose(target)

        vertices, faces = self._export_object(destination, target, "object_geometry.npz")
        task_objects = {}
        labels = ("block_0", "block_1", "block_2") if self.task_name == "stack_blocks" else (target_label,)
        for label in labels:
            instance = manager.get_instance_name(label=label, env_idx=0)
            obj = manager.get_scene_object(inst_name=instance, env_idx=0)
            position, quaternion = self._object_pose(obj)
            filename = f"task_object_{label}.npz"
            v, f = self._export_object(destination, obj, filename)
            task_objects[label] = dict(prim_path=obj.prim_path, position=position.tolist(),
                                       quaternion_wxyz=quaternion.tolist(), geometry_file=filename)
        print("TASK_TARGET_GEOMETRY", len(vertices), len(faces), flush=True)

        obstacles, count = {}, 0
        exported = {target_path}
        for obj in env.scene_manager.get_objects([0]).values():
            if obj.prim_path in exported:
                continue
            exported.add(obj.prim_path)
            print("TASK_EXPORT_GEOMETRY", obj.prim_path, flush=True)
            try:
                actual_pose = pose_matrix(self._object_pose(obj)) if hasattr(obj, "get_world_pose") else None
                v, f = world_geometry(env.stage, obj.prim_path, actual_pose)
            except ValueError:
                continue
            hull = trimesh.Trimesh(v, f, process=False).convex_hull
            obstacles.update({f"vertices_{count}": hull.vertices, f"faces_{count}": hull.faces})
            count += 1
            print("TASK_OBSTACLE", count, obj.prim_path, flush=True)

        other_index = 1 - self.arm_index
        other = env.robot_manager.robot_key[other_index]
        other_mechanism = self.mechanisms[other_index]
        other_installation = self.installations[other_index]
        poses = other.data.body_link_pose_w[0].cpu().numpy()
        from soft_sim.sim.robodojo import robot_link_path
        for name, pose in zip(other.body_names, poses):
            if name in other_mechanism.finger_links:
                continue
            actual_pose = pose_matrix((pose[:3], pose[3:]))
            v, f = world_geometry(env.stage, robot_link_path(env.stage, other_index, name), actual_pose)
            hull = trimesh.Trimesh(v, f, process=False).convex_hull
            obstacles.update({f"vertices_{count}": hull.vertices, f"faces_{count}": hull.faces})
            count += 1
        for nodes in env.tools[other_index].positions():
            hull = trimesh.convex.convex_hull(nodes)
            obstacles.update({f"vertices_{count}": hull.vertices, f"faces_{count}": hull.faces})
            count += 1
        for side in (0, 1):
            pose = env.tools[other_index].mounts[side] @ np.linalg.inv(finger_in_link(
                side, other_installation, other_mechanism))
            for mesh in fixture_meshes(self.finger_scale, side, other_installation,
                                       other_mechanism).values():
                mesh.apply_transform(pose)
                obstacles.update({f"vertices_{count}": mesh.vertices, f"faces_{count}": mesh.faces})
                count += 1
        np.savez_compressed(destination / "obstacle_geometry.npz", count=count, **obstacles)

        robot = env.robot_manager.robot_key[self.arm_index]
        scene = dict(
            task=self.task_name, layout=str(self.layout_path), layout_sha256=self.layout_hash,
            task_objects=task_objects, obstacle_count=count, target_parameters=target_parameters,
            target_label=target_label, target_prim=target_path,
            initial_q=robot.data.joint_pos[0].cpu().numpy().tolist(),
            initial_object_position=object_pose()[0].tolist(),
            table_z=.765, table_center_m=[0., -.05, .74], table_size_m=[1.4, 1.1, .05],
            world_from_robot_base=self.base.tolist(), hand_mount_offset_m=.075,
            adapter_offset_m=self.adapter_offset_m, finger_mesh=self.cfg.mesh.stl,
            collision_scope="task rigid and geometry objects + table; convex hulls; room excluded",
            embodiment_variant=self.mechanism.embodiment)
        scene.update(robot_name=self.mechanism.name, active_arm=self.arm_index,
                     finger_scale=self.finger_scale,
                     finger_scale_to_m=self.cfg.mesh.scale_to_m,
                     target_geometry_sha256=hashlib.sha256(
                         (destination / "object_geometry.npz").read_bytes()).hexdigest())
        scene.update(material=self.cfg.to_dict()["material"],
                     fem_initialized_after_task_reset=True,
                     removed_factory_finger_colliders=env.removed_finger_colliders,
                     initial_fem_validation=self.fem_report,
                     physics_initialization_valid=self.fem_valid,
                     texture_max_resolution=self.texture_max_resolution,
                     deformable_mesh=[tool.mesh_settings for tool in env.tools],
                     mesh_topology=[tool.mesh_topology for tool in env.tools])
        scene["collision_scope"] += f"; parked second {other_mechanism.name} links and deformed soft fingers included"
        scene.update(hand_mount_offset_m=float(self.mechanism.tool_origin_m @ self.mechanism.growth_axis_in_hand),
                     adapter_offset_m=self.installation["closed_half_spacing_m"],
                     joint_names=robot.joint_names,
                     tool_origin_in_hand_m=list(self.mechanism.tool_origin_m),
                     tool_rotation=self.mechanism.rotation.tolist())
        scene["finger_installation"] = self.installation
        scene["finger_in_slide_transforms"] = [
            finger_in_link(side, self.installation, self.mechanism).tolist() for side in (0, 1)]
        scene["adapter_offset_m"] = self.installation["closed_half_spacing_m"]
        scene["hand_mount_offset_m"] += self.installation["forward_offset_m"]
        (destination / "scene.json").write_text(json.dumps(scene, indent=2))
        return scene, target, object_pose
