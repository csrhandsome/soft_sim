# `insert_tubes` Insertion V2 Adaptation

## Overview

This adaptation fixes the root software defects uncovered during the `verify_insert_tubes_s0_per_op_arm_v1_20261115` GPU run:
1. `insert_tubes` previously fell back to `container` 2x2 bounding-box grid placement, misaligning tube endpoints relative to the real `slot` pockets and failing to account for `slot` rotation and off-center `tube` origins.
2. `curobo.py` planned with a table-only cuRobo world instead of `world_collision.get_scene_planner`.
3. `release` and `retreat` opened immediately to `max_open_width_m`, driving fingers into `slot` walls inside confined holes.
4. Grasp loss during `lift` / `transfer` was not caught before `place`, causing empty-jaw descent, and in-hand slip was not corrected before placement.

## Changes

- **Hole-Aligned Placement (`src/soft_sim/tasks/insert_tubes_geometry.py`, `src/soft_sim/tasks/adapters.py`)**
  - `extract_slot_holes` ray-casts downward on a local XY grid (`grid_step_m=0.002`, `min_depth_m=0.015`) over the `slot` triangle mesh and labels connected pockets with `scipy.ndimage.label`.
  - `measure_tube_profile` transforms `tube` vertices into the upright goal frame and measures `shaft_radius_m`, `origin_to_bottom_m`, and `upright_center_xy_offset_m`.
  - `select_insert_tubes_holes` selects `op.count` qualifying holes (`equiv_radius_m >= 0.92 * shaft_radius_m`, `depth_m >= 0.045`) and raises `ValueError` on insufficient or ambiguous holes (no 2x2 bbox fallback).
  - `TaskAdapter.placement` stores `self.last_placement_report` (algorithm parameters, selected holes, geometry SHA-256) and `collect.py` persists it in `placement.json`.

- **Long-Axis Grasp Sampling (`src/soft_sim/planning/candidates.py`)**
  - For `task == "insert_tubes"`, `generate_candidates` computes the object's principal horizontal axis and samples axial offsets `(0.0, -0.025, 0.025, -0.040, 0.040)` clamped to `0.4 * span` while keeping the full `contact_mesh` intact.

- **World Collision & Bounded Release (`src/soft_sim/planning/curobo.py`, `src/soft_sim/execution.py`)**
  - `plan_scene_curobo` calls `world_collision.get_scene_planner(scene_dir, arm.name, list(arm.arm_joints), table_height, sample_dt=sample_dt)` and records `world_collision` in `plan.json` (no table-only fallback).
  - For `insert_tubes` placements, `curobo.py` computes a bounded `release_width` from `candidate["width"]`, `finger_inner_radius_m`, `adapter_offset_m`, and `+0.012 m` clearance, clamped to `[closed + 0.004, opened]`.
  - `pickup_phases` uses `release_width` during `release` and `retreat`, then opens to `opened` in `return_home` and `verify_hold`. Tasks without `release_width` (including `general_pickup`) keep their exact original behavior.

- **Grasp Envelope Check & 1-Time Pose-Corrected Placement (`src/soft_sim/execution.py`, `src/soft_sim/tasks/collect.py`, `src/soft_sim/planning/curobo.py`)**
  - `hand_from_object_transform` and `grasp_envelope_status` verify after `lift` and `transfer` that the object remains within `max_mount_distance_m=0.16` and `max_slip_m=0.08`. On loss, `collect.py` writes `diagnostics/grasp_loss.json`, appends a `grasp_lost` checkpoint, sets `rejected=True`, and aborts before `place`.
  - If an `insert_tubes` object remains gripped after `lift` with `slip_m > 0.005` or `slip_deg > 3.0`, `collect.py` invokes `replan_placement_curobo` once and records `plan_revision` in `plans/executed.json`.

## Verification

- CPU unit tests (`PYTHONPATH=.venv/lib/python3.12/site-packages:src pytest tests/unit -v`): **153 passed** (including 6 new tests in `tests/unit/test_insert_tubes_v2.py`).
- `tests/integration/planning/test_per_operation_arm.py` requires `mplib` (present in the Isaac Sim runtime environment).
