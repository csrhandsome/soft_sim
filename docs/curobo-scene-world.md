# cuRobo Scene World Collision (`world_collision.py`)

`src/soft_sim/planning/world_collision.py` provides a standalone scene-aware wrapper around RoboDojo's cuRobo v2 `CuroboPlanner` without modifying `curobo.py`, `candidates.py`, `robodojo.py`, or upstream `third_party/`.

## API

```python
from soft_sim.planning.world_collision import (
    clear_scene_planner_cache,
    get_scene_planner,
    load_scene_world_spec,
)

planner = get_scene_planner(
    scene_dir="artifacts/runs/<run>/plans/operation_000",
    robot_name="x5",
    arm_joints=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    table_height=None,  # defaults to scene["table_z"] - base_z - 0.025
    sample_dt=0.02,
)
```

`get_scene_planner(...)` returns a `SceneCuroboPlanner` instance with the standard RoboDojo `CuroboPlanner` interface (`plan_path`, `solve_ik_to_joint`, `plan_batch`, `plan_joint`) plus a JSON-serializable `planner.world_report`.

## Coordinate Transform

`scene.json` stores `world_from_robot_base` as a 4×4 homogeneous matrix `T = [R, t; 0, 1]` mapping robot-base points to world coordinates (`p_world = R @ p_base + t`).
Because `CuroboPlanner` plans in the robot base frame (`robot_origin_pose = [0, 0, 0, 1, 0, 0, 0]`), `world_collision.py` transforms every exported obstacle vertex array `V_world` from `obstacle_geometry.npz` into robot-base coordinates:

```python
V_base = (V_world - t) @ R
```

Raw triangle soups (`faces_i`) are passed directly into `scene_model["mesh"]` with **no convex hullification**, preserving non-convex features such as the holes in `test_tube_slot`.

### Small-Mesh Cuboid Fallback (`MIN_MESH_HALF_DIAGONAL_M = 0.06`)

cuRobo v2's `MeshDataWarp` (`compute_local_sdf`) clamps outside-mesh distance queries to `max_distance = 0.5 * bbox_diagonal`. For small meshes whose half-diagonal is smaller than the robot's largest collision sphere radius + activation buffer (`~0.053 m` on X5), `compute_local_sdf` returns `max_distance < sphere_radius` everywhere in space, causing false collisions on every pose.
`world_collision.py` therefore keeps all obstacles with `half_diagonal >= 0.06 m` (`tube1`, `tube2`, `slot`, `camera_stand`, large arm links) as exact non-convex triangle meshes in `scene_model["mesh"]`, and represents small obstacles (`half_diagonal < 0.06 m`, such as parked deformed soft fingers and small camera/fixture boxes) as conservative axis-aligned cuboids in `scene_model["cuboid"]`. Every obstacle's `collider_kind` (`"mesh"` vs `"cuboid_aabb_fallback"`) is recorded in `world_report["obstacles"]`.

## Table Cuboid vs. Real Surface Relationship

Following `src/soft_sim/planning/curobo.py`:
- When `table_height` is omitted (`None`), `table_height = scene["table_z"] - base_z - 0.025`.
- `CuroboPlanner` places a `3.0 × 3.0 × 0.05 m` cuboid centered at `z_base = table_height - 0.025`, so its top surface sits at `table_height` (`-0.025 m` below `base_z` when `table_z == base_z`).
- This prevents the fixed `base_link` mount at `z_base = 0.0` from intersecting the table cuboid, while exported obstacle meshes sit at `z_world >= table_z` (`z_base >= 0.0 m`).

## Cache Keying

`get_scene_planner(...)` caches instances by:

```python
(robot_name, tuple(arm_joints), round(table_height, 4), round(sample_dt, 4), base_pose_hash, geometry_hash)
```

Switching between left and right arms (different `world_from_robot_base`) or changing obstacle geometry produces a distinct cache key so stale obstacle worlds never leak across operations.

## Collision Scope & Boundaries

Included in `world_report["scope"]["included"]`:
- `table_cuboid`: 5 cm table cuboid from `CuroboPlanner`.
- `static_task_obstacles_non_convex_meshes`: non-convex triangle meshes from `obstacle_geometry.npz`.
- `parked_second_arm_and_deformed_soft_fingers_from_obstacle_geometry_npz`: exported links and deformed soft fingers of the non-active arm.
- `stock_robot_collision_spheres`: stock spheres from RoboDojo's cuRobo YAML.

Excluded (`world_report["scope"]["excluded"]`):
- Target object currently being grasped.
- Room architecture.
- Dynamic FEM finger deformation during motion.
- Attached/carried object geometry during transfer.

`SceneCuroboPlanner` passes `collision_cache={"cuboid": 4, "mesh": max(1, len(meshes))}` to both `_build_motion_planner_cfg` and `_build_ik_cfg`, so `SceneCollision` enables `cuboid=True` and `mesh=True` across `motion_planner`, `motion_planner_batch`, and `ik_solver`.
