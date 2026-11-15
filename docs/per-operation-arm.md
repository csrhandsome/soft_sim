# Per-Operation Automatic Arm Selection

## 1. Behavior & Scope

- **Automatic mode (`args.active_arm is None`)**:
  - At `operation_000` (after `apply_saved_poses`) and before each subsequent `operation_index > 0`, [`select_arm_for_position`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/tasks/collect.py) reads the current world position of `operation.target` and selects `0` (left arm) if `x < 0.0`, else `1` (right arm).
  - Switching only occurs **between** operations, after the previous operation has finished its full `release -> retreat -> return_home` sequence. No mid-operation handoff, object teleportation, or skipped phase is performed; if a single operation requires cross-workspace motion that is geometrically unreachable, planning fails honestly (`planning_failed`).
- **Fixed mode (`--active-arm 0|1`)**:
  - Preserves the explicit arm across all operations in the episode.
- **Synchronous switching**:
  - On each operation boundary (`operation_index > 0`), [`collect.py`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/tasks/collect.py) updates `arm_index`, `robot`, `env.tool` (wrist-mounted soft-finger cameras, FEM state, and `fixtures`), `arm` (`mechanism`), `installation`, the freshly measured `base` (`measure_arm_base` from `robot.data.root_link_pose_w[0]`), and calls [`SceneExporter.set_active_arm`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/sim/scene.py).
  - `exporter.export(operation_dir, target_label)` then writes an up-to-date `scene.json` and `obstacle_geometry.npz` (with the **other** arm `1 - arm_index` exported as the parked obstacle).
- **Per-arm independent state & initialization**:
  - Jaw width continuity is tracked per arm in `last_widths[arm_index]` (initialized from `mechanisms[i].max_open_width_m`, with `last_widths[initial_arm]` set to `plan["candidate"]["opened"]`).
  - `parked_commands` maintains independent 8-DOF targets per arm while recording the full 16-DOF `joint_positions`, `joint_velocities`, and `commanded_joint_positions` in `[robot0, robot1]` order.
  - When `args.active_arm is None`, both arms' cameras are enabled (`tool_config.simulation.render = True`) and `reports/initial/mount_validation.json` validates FK vs. measured mounts for both arms (`per_arm`).
  - `capture/finger_reference.npz` preserves top-level `finger_reference_*` / `finger_pin_*` / `finger_tets_*` keys for backward compatibility and adds explicit `robot{0,1}_finger_reference_*`, `robot{0,1}_finger_pin_*`, and `robot{0,1}_finger_tets_*` keys.

## 2. Data Consumer `operation_index -> active_arm` Mapping

The official LeRobot v3 Parquet feature schema is **unchanged**. Downstream consumers determine which arm owns `left_rgb`, `right_rgb`, `finger_depth_m`, `finger_mounts`, `finger_camera_to_world`, `finger_nodes_left`, `finger_nodes_right`, and `fixture_to_world` on any frame by:

1. Reading the frame's `operation_index`.
2. Looking up `executed_plans[operation_index]["active_arm"]` (in `plans/executed.json` or `episode["result"]["executed_plans"]`), or `plans/operation_{operation_index:03d}/scene.json["active_arm"]`.
3. Falling back to `episode["calibration"]["active_arm"]` for legacy single-arm recordings that lack `executed_plans[op]["active_arm"]`.

`episode["calibration"]` also records `all_fixture_paths`, `all_finger_installations`, `all_robot_names`, `all_robot_types`, `all_finger_projections`, `all_render_settings`, `all_base_poses`, and `per_operation_arm_selection` (`"auto"` or `"fixed"`), while retaining all legacy single-arm keys (`fixture_paths`, `finger_installation`, `active_arm`, etc.).

Offline validators ([`collisions.py`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/validation/collisions.py), [`fem_projection.py`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/validation/fem_projection.py), [`finger_motion.py`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/validation/finger_motion.py), [`clamp.py`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/validation/clamp.py)) resolve the active arm per frame via `resolve_operation_arm` and `finger_reference_for_arm`, with full fallback for legacy data.

## 3. Known Limitations & Unverified GPU Scope

- **Not yet GPU-verified**: This implementation was developed and tested on CPU only (unit, planning, and dataset suites). Isaac Sim + cuRobo GPU execution on `insert_tubes` has **not** been run in this round.
- **cuRobo collision world scope**: `src/soft_sim/planning/curobo.py` currently only passes `table_height` to cuRobo and does **not** import arbitrary obstacle meshes from `obstacle_geometry.npz` (such as the parked arm or other tubes) into cuRobo's collision world. Parked-arm avoidance relies on the previous operation completing `return_home` and workspace separation. Returning both arms home does **not** guarantee collision-free trajectories; offline FCL checks in [`collisions.py`](file:///data/codes/grade_two/my_projects/robot/soft_sim/src/soft_sim/validation/collisions.py) must still be run on the recorded episode.

## 4. Incremental Sync & GPU Verification Command

### Incremental Sync File List (to 4090)

- `src/soft_sim/sim/scene.py`
- `src/soft_sim/tasks/collect.py`
- `src/soft_sim/validation/collisions.py`
- `src/soft_sim/validation/fem_projection.py`
- `src/soft_sim/validation/finger_motion.py`
- `src/soft_sim/validation/clamp.py`
- `tests/unit/test_per_operation_arm.py`
- `tests/integration/planning/test_per_operation_arm.py`
- `docs/per-operation-arm.md`

### GPU Acceptance Command

```bash
.venv/bin/python scripts/run.py collect --task insert_tubes --seed 0 --planner curobo --indent 0.010 --run-id verify_insert_tubes_s0_per_op_arm_v1
```

Acceptance requires all three tubes (`tube0`, `tube1`, `tube2`) to complete and pass the official task reward (`success == true`, `official_reward_success == true`).
