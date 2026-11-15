"""Scene-aware cuRobo v2 world-collision planner wrapper.

Builds a RoboDojo-compatible ``CuroboPlanner`` populated with the exported
non-convex triangle-mesh obstacles from ``scene.json`` and
``obstacle_geometry.npz`` in the robot base frame, alongside the standard table
cuboid.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

_SCENE_PLANNER_CACHE: dict[tuple[Any, ...], Any] = {}
# cuRobo v2 compute_local_sdf clamps outside-mesh queries to 0.5 * bbox_diagonal;
# meshes smaller than the largest robot collision sphere + activation buffer (~0.053m)
# would always register false collisions, so we represent them as exact cuboids.
MIN_MESH_HALF_DIAGONAL_M = 0.06


def clear_scene_planner_cache() -> None:
    """Clear the module-level scene planner cache."""
    _SCENE_PLANNER_CACHE.clear()


def transform_world_vertices_to_base(
    vertices_world: np.ndarray,
    world_from_robot_base: np.ndarray,
) -> np.ndarray:
    """Transform ``(N, 3)`` world-frame vertices into the robot base frame.

    ``world_from_robot_base`` is the 4x4 homogeneous transform ``T`` mapping
    points in the robot base frame to world coordinates: ``p_world = R @ p_base + t``.
    Therefore ``p_base = R.T @ (p_world - t)``, which for row vectors is
    ``(V_world - t) @ R``.
    """
    transform = np.asarray(world_from_robot_base, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) world_from_robot_base, got shape {transform.shape}")
    verts = np.asarray(vertices_world, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) vertices_world, got shape {verts.shape}")
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return ((verts - trans) @ rot).astype(np.float32)


def build_collision_cache(num_meshes: int, num_cuboids: int = 4) -> dict[str, int]:
    """Return cuRobo v2 ``collision_cache`` sizing for cuboids and meshes."""
    return {
        "cuboid": max(1, int(num_cuboids)),
        "mesh": max(1, int(num_meshes)),
    }


def make_scene_planner_cache_key(
    robot_name: str,
    arm_joints: list[str] | tuple[str, ...],
    table_height: float,
    sample_dt: float,
    base_pose_hash: str,
    geometry_hash: str,
) -> tuple[Any, ...]:
    """Build deterministic cache key separating base transforms and geometries."""
    return (
        str(robot_name),
        tuple(str(j) for j in arm_joints),
        round(float(table_height), 4),
        round(float(sample_dt), 4),
        str(base_pose_hash),
        str(geometry_hash),
    )


def load_scene_world_spec(
    scene_dir: str | Path,
    table_height: float | None = None,
) -> dict[str, Any]:
    """Load ``scene.json`` and ``obstacle_geometry.npz`` into robot-base frame.

    Preserves raw non-convex triangle meshes (no convex hullification) and
    builds a JSON-serializable ``world_report`` describing scope and geometry.
    """
    scene_path = Path(scene_dir)
    scene_json_path = scene_path / "scene.json"
    obstacle_npz_path = scene_path / "obstacle_geometry.npz"

    if not scene_json_path.is_file():
        raise FileNotFoundError(f"Missing scene.json in {scene_path}")
    if not obstacle_npz_path.is_file():
        raise FileNotFoundError(f"Missing obstacle_geometry.npz in {scene_path}")

    scene = json.loads(scene_json_path.read_text(encoding="utf-8"))
    world_from_robot_base = np.asarray(scene["world_from_robot_base"], dtype=np.float64)
    if world_from_robot_base.shape != (4, 4):
        raise ValueError(
            f"Invalid world_from_robot_base shape {world_from_robot_base.shape} in {scene_json_path}"
        )

    table_z = float(scene["table_z"])
    base_z = float(world_from_robot_base[2, 3])
    resolved_table_height = (
        float(table_height) if table_height is not None else (table_z - base_z - 0.025)
    )

    meshes: dict[str, dict[str, Any]] = {}
    cuboids: dict[str, dict[str, Any]] = {}
    obstacle_summaries: list[dict[str, Any]] = []
    geom_hasher = hashlib.sha256()

    with np.load(obstacle_npz_path, allow_pickle=False) as data:
        count = int(data["count"])
        for idx in range(count):
            verts_world = np.asarray(data[f"vertices_{idx}"], dtype=np.float64)
            faces = np.asarray(data[f"faces_{idx}"], dtype=np.int32)
            if verts_world.size == 0 or faces.size == 0:
                continue
            verts_base = transform_world_vertices_to_base(verts_world, world_from_robot_base)
            v_min = verts_base.min(axis=0)
            v_max = verts_base.max(axis=0)
            dims = np.maximum(v_max - v_min, 1e-4)
            half_diag = float(0.5 * np.linalg.norm(dims))
            name = f"obstacle_{idx}"
            if half_diag < MIN_MESH_HALF_DIAGONAL_M:
                center = 0.5 * (v_min + v_max)
                cuboids[name] = {
                    "dims": [float(d) for d in dims],
                    "pose": [float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
                }
                collider_kind = "cuboid_aabb_fallback"
            else:
                meshes[name] = {
                    "vertices": verts_base.tolist(),
                    "faces": faces.tolist(),
                    "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                }
                collider_kind = "mesh"
            geom_hasher.update(name.encode("utf-8"))
            geom_hasher.update(np.round(verts_base.astype(np.float64), 6).tobytes())
            geom_hasher.update(faces.tobytes())
            obstacle_summaries.append(
                {
                    "name": name,
                    "collider_kind": collider_kind,
                    "half_diagonal_m": round(half_diag, 6),
                    "vertex_count": int(verts_base.shape[0]),
                    "face_count": int(faces.shape[0]),
                    "z_min_base_m": round(float(v_min[2]), 6),
                    "z_max_base_m": round(float(v_max[2]), 6),
                }
            )

    base_pose_hash = hashlib.sha256(
        np.round(world_from_robot_base, 7).tobytes()
    ).hexdigest()[:16]
    geometry_hash = geom_hasher.hexdigest()[:16]
    collision_cache = build_collision_cache(len(meshes), num_cuboids=max(4, 1 + len(cuboids)))

    world_report: dict[str, Any] = {
        "scene_dir": str(scene_path),
        "task": scene.get("task"),
        "robot_name": scene.get("robot_name"),
        "active_arm": scene.get("active_arm"),
        "target_label": scene.get("target_label"),
        "base_pose_hash": base_pose_hash,
        "geometry_hash": geometry_hash,
        "table_z_world_m": table_z,
        "robot_base_z_world_m": base_z,
        "planner_table_height_m": resolved_table_height,
        "planner_table_top_z_base_m": resolved_table_height,
        "planner_table_center_z_base_m": resolved_table_height - 0.025,
        "table_relationship": (
            "Planner table cuboid top sits at planner_table_height_m in robot base frame "
            "(table_z - base_z - 0.025 = -0.025m when table_z == base_z) so the fixed "
            "base_link mount at z=0 does not intersect the table cuboid, while exported "
            "obstacle meshes sit at z_world >= table_z (z_base >= 0.0m)."
        ),
        "obstacle_count": len(meshes) + len(cuboids),
        "mesh_obstacle_count": len(meshes),
        "cuboid_fallback_count": len(cuboids),
        "min_mesh_half_diagonal_m": MIN_MESH_HALF_DIAGONAL_M,
        "non_convex_triangles_preserved": True,
        "obstacles": obstacle_summaries,
        "collision_cache": collision_cache,
        "scope": {
            "included": [
                "table_cuboid",
                "static_task_obstacles_non_convex_meshes",
                "parked_second_arm_and_deformed_soft_fingers_from_obstacle_geometry_npz",
                "stock_robot_collision_spheres",
            ],
            "excluded": [
                "target_object_being_grasped",
                "room_architecture",
                "dynamic_fem_finger_deformation_during_motion",
                "carried_object_attached_geometry",
            ],
            "notes": (
                "Stock robot collision_spheres from RoboDojo curobo yml are preserved "
                "unchanged; dynamic soft-finger deformation and in-hand carried-object "
                "collision require separate attachment/sphere updates."
            ),
        },
        "collision_backends": {},
    }

    return {
        "meshes": meshes,
        "cuboids": cuboids,
        "table_height": resolved_table_height,
        "world_from_robot_base": world_from_robot_base,
        "base_pose_hash": base_pose_hash,
        "geometry_hash": geometry_hash,
        "collision_cache": collision_cache,
        "world_report": world_report,
    }


def _inspect_collision_backend(component: Any) -> dict[str, Any]:
    """Extract active SceneCollision types from a cuRobo v2 component."""
    checker = getattr(component, "scene_collision_checker", None)
    if checker is None:
        for attr in ("trajopt_solver", "ik_solver"):
            sub = getattr(component, attr, None)
            checker = getattr(sub, "scene_collision_checker", None)
            if checker is not None:
                break
    if checker is None:
        return {"present": False, "collision_types": None}
    ctypes = getattr(checker, "collision_types", None)
    if isinstance(ctypes, dict):
        ctypes = {str(k): bool(v) for k, v in ctypes.items()}
    return {
        "present": True,
        "type": type(checker).__name__,
        "collision_types": ctypes,
    }


def _create_scene_planner_instance(
    robot_name: str,
    arm_joints: list[str],
    world_spec: dict[str, Any],
    sample_dt: float,
    planner_cls: type | None = None,
    yml_path: str | None = None,
) -> Any:
    """Instantiate a ``SceneCuroboPlanner`` wrapping RoboDojo ``CuroboPlanner``."""
    if planner_cls is None or yml_path is None:
        from soft_sim.planning.curobo import _ensure_robodojo_path, _patched_curobo_yml

        _ensure_robodojo_path()
        if yml_path is None:
            yml_path = _patched_curobo_yml(robot_name)
        if planner_cls is None:
            from env.planner_manager.curobo_planner import CuroboPlanner

            planner_cls = CuroboPlanner

    meshes = deepcopy(world_spec["meshes"])
    cuboids = deepcopy(world_spec.get("cuboids", {}))
    collision_cache = dict(world_spec["collision_cache"])
    world_report = deepcopy(world_spec["world_report"])

    class SceneCuroboPlanner(planner_cls):  # type: ignore[misc,valid-type]
        """``CuroboPlanner`` subclass injecting exported scene meshes into cuRobo v2."""

        def __init__(self) -> None:
            self._scene_meshes = meshes
            self._scene_cuboids = cuboids
            self._collision_cache = collision_cache
            self.world_report = world_report
            super().__init__(
                robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                active_joints_name=list(arm_joints),
                all_joints=list(arm_joints),
                dt=float(sample_dt),
                yml_path=yml_path,
                table_height=float(world_spec["table_height"]),
            )
            self.world_report["collision_backends"] = {
                "motion_planner": _inspect_collision_backend(getattr(self, "motion_planner", None)),
                "motion_planner_batch": _inspect_collision_backend(
                    getattr(self, "motion_planner_batch", None)
                ),
                "ik_solver": _inspect_collision_backend(getattr(self, "ik_solver", None)),
            }

        def _build_robot_and_scene_cfg(self, yml_data: dict[str, Any], table_height: float):
            robot_cfg_v2, scene_model = super()._build_robot_and_scene_cfg(yml_data, table_height)
            scene_model = dict(scene_model)
            if self._scene_cuboids:
                merged_cuboids = dict(scene_model.get("cuboid", {}))
                merged_cuboids.update(deepcopy(self._scene_cuboids))
                scene_model["cuboid"] = merged_cuboids
            if self._scene_meshes:
                scene_model["mesh"] = deepcopy(self._scene_meshes)
            return robot_cfg_v2, scene_model

        def _build_motion_planner_cfg(self, max_batch_size: int):
            from curobo.motion_planner import MotionPlannerCfg

            cfg = MotionPlannerCfg.create(
                robot=self.robot_cfg,
                scene_model=self.scene_model,
                collision_cache=self._collision_cache,
                device_cfg=self.device_cfg,
                num_ik_seeds=32,
                num_trajopt_seeds=4,
                self_collision_check=True,
                use_cuda_graph=self.use_cuda_graph,
                max_batch_size=max_batch_size,
                multi_env=False,
                max_goalset=1,
            )
            if self.dt is not None:
                cfg.trajopt_solver_config.interpolation_dt = float(self.dt)
            if not self.use_graph_planner:
                cfg.graph_planner_config = None
            return cfg

        def _build_ik_cfg(self):
            from curobo.inverse_kinematics import InverseKinematicsCfg

            return InverseKinematicsCfg.create(
                robot=self.robot_cfg,
                scene_model=self.scene_model,
                collision_cache=self._collision_cache,
                device_cfg=self.device_cfg,
                num_seeds=32,
                seed_solver_num_seeds=32,
                position_tolerance=0.001,
                orientation_tolerance=0.02,
                self_collision_check=True,
                use_cuda_graph=self.use_cuda_graph,
                max_batch_size=1,
                multi_env=False,
                max_goalset=1,
            )

    return SceneCuroboPlanner()


def get_scene_planner(
    scene_dir: str | Path,
    robot_name: str,
    arm_joints: list[str] | tuple[str, ...],
    table_height: float | None = None,
    sample_dt: float = 0.02,
    *,
    planner_cls: type | None = None,
    yml_path: str | None = None,
) -> Any:
    """Return a cached RoboDojo-compatible planner with scene mesh collision."""
    world_spec = load_scene_world_spec(scene_dir, table_height=table_height)
    key = make_scene_planner_cache_key(
        robot_name=robot_name,
        arm_joints=arm_joints,
        table_height=world_spec["table_height"],
        sample_dt=sample_dt,
        base_pose_hash=world_spec["base_pose_hash"],
        geometry_hash=world_spec["geometry_hash"],
    )
    planner = _SCENE_PLANNER_CACHE.get(key)
    if planner is None:
        planner = _create_scene_planner_instance(
            robot_name=robot_name,
            arm_joints=list(arm_joints),
            world_spec=world_spec,
            sample_dt=sample_dt,
            planner_cls=planner_cls,
            yml_path=yml_path,
        )
        _SCENE_PLANNER_CACHE[key] = planner
    return planner
