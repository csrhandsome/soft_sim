from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

from soft_sim.execution import (
    grasp_envelope_status,
    hand_from_object_transform,
    pickup_phases,
)
from soft_sim.planning.candidates import generate_candidates
from soft_sim.planning.curobo import replan_placement_curobo
from soft_sim.tasks.adapters import TaskAdapter
from soft_sim.tasks.insert_tubes_geometry import (
    extract_slot_holes,
    insert_tubes_placement,
    measure_tube_profile,
)


def _make_rack_with_holes(hole_centers_xy, hole_radii, floor_z=-0.02, top_z=0.03):
    """Build a synthetic 3D rack mesh with cylindrical through-pockets from top_z down to floor_z."""
    parts = []
    # Solid base plate below floor_z (matching outer wall footprint so no perimeter ledge exists)
    base = trimesh.creation.box(extents=[0.22, 0.12, 0.02], transform=trimesh.transformations.translation_matrix([0.0, 0.0, floor_z - 0.01]))
    parts.append(base)
    # Build walls around each hole using a fine 2D grid extruded? Simpler: a box per grid cell outside all holes!
    xs = np.linspace(-0.11, 0.11, 56)
    ys = np.linspace(-0.06, 0.06, 31)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    h = top_z - floor_z
    zc = 0.5 * (top_z + floor_z)
    for x in xs:
        for y in ys:
            inside_any = any((x - cx) ** 2 + (y - cy) ** 2 <= r**2 for (cx, cy), r in zip(hole_centers_xy, hole_radii))
            if not inside_any:
                cell = trimesh.creation.box(
                    extents=[dx * 1.02, dy * 1.02, h],
                    transform=trimesh.transformations.translation_matrix([x, y, zc]),
                )
                parts.append(cell)
    return trimesh.util.concatenate(parts)


def _dummy_layout():
    return {
        "Rigid": {
            "default": [
                {"label": "slot"},
                {"label": "tube0"},
                {"label": "tube1"},
                {"label": "tube2"},
            ]
        }
    }


class TestInsertTubesV2(unittest.TestCase):
    def test_insert_tubes_rotated_slot_and_offset_tube_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Three non-2x2 qualifying holes in a line + one small distractor hole
            hole_centers = [(-0.06, 0.0), (0.0, 0.0), (0.06, 0.0), (0.09, 0.03)]
            hole_radii = [0.017, 0.017, 0.017, 0.007]
            rack = _make_rack_with_holes(hole_centers, hole_radii, floor_z=-0.02, top_z=0.03)
            slot_path = tmp_path / "slot.npz"
            np.savez(slot_path, vertices=np.asarray(rack.vertices), faces=np.asarray(rack.faces))

            # Tube mesh lying along Y in local frame (so op.orientation_wxyz=(2**-.5, 2**-.5, 0, 0) stands it upright)
            # with local origin offset at a corner (0.02, 0.0, -0.01) instead of shaft center
            cyl = trimesh.creation.cylinder(
                radius=0.014,
                height=0.11,
                sections=24,
                transform=trimesh.transformations.rotation_matrix(-np.pi / 2, [1.0, 0.0, 0.0]),
            )
            tube_pos = [-0.20, 0.15, 0.77]
            cyl.apply_translation([0.02, 0.055, -0.01])
            cyl_world = cyl.copy()
            cyl_world.apply_translation(tube_pos)
            tube_path = tmp_path / "tube0.npz"
            np.savez(tube_path, vertices=np.asarray(cyl_world.vertices), faces=np.asarray(cyl_world.faces))

            # Rotate slot by 30 deg around Z
            yaw = np.deg2rad(30.0)
            q_slot = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
            slot_pos = [0.10, -0.05, 0.79]
            scene = {
                "task": "insert_tubes",
                "task_objects": {
                    "slot": {
                        "geometry_file": "slot.npz",
                        "position": slot_pos,
                        "quaternion_wxyz": q_slot,
                    },
                    "tube0": {
                        "geometry_file": "tube0.npz",
                        "position": [-0.20, 0.15, 0.77],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
            }

            adapter = TaskAdapter("insert_tubes", _dummy_layout())
            goal = adapter.placement(scene, tmp_path, gap_m=0.002)
            assert goal is not None
            assert adapter.last_placement_report is not None
            report = adapter.last_placement_report
            assert report["algorithm"] == "raycast_hole_extraction_v1"
            assert len(report["slot_geometry_sha256"]) == 64
            assert len(report["tube_geometry_sha256"]) == 64
            assert len(report["selected_holes"]) == 3

            # Verify the tube's shaft bottom center in world frame actually lands on the selected hole floor + clearance
            chosen_floor = np.asarray(report["chosen_hole"]["world_floor_center_m"], dtype=float)
            axis_world = np.asarray(report["chosen_hole"]["axis_world"], dtype=float)
            expected_bottom = chosen_floor + axis_world * (0.002 + 0.002)

            # Transform tube vertices by `goal` and measure actual shaft bottom center
            placed_verts = (goal[:3, :3] @ np.asarray(cyl.vertices).T).T + goal[:3, 3]
            z_min = float(placed_verts[:, 2].min())
            bottom_ring = placed_verts[placed_verts[:, 2] <= z_min + 0.01]
            actual_bottom_xy = 0.5 * (bottom_ring[:, :2].min(axis=0) + bottom_ring[:, :2].max(axis=0))
            assert np.allclose(actual_bottom_xy, expected_bottom[:2], atol=0.004)
            assert abs(z_min - expected_bottom[2]) < 0.002

    def test_insert_tubes_rejects_ambiguous_or_insufficient_holes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Only 2 large holes when op.count == 3 -> must fail loudly, not fall back to 2x2 bbox
            rack = _make_rack_with_holes([(-0.04, 0.0), (0.04, 0.0)], [0.017, 0.017])
            np.savez(tmp_path / "slot.npz", vertices=np.asarray(rack.vertices), faces=np.asarray(rack.faces))
            cyl = trimesh.creation.cylinder(
                radius=0.014,
                height=0.11,
                sections=24,
                transform=trimesh.transformations.rotation_matrix(-np.pi / 2, [1.0, 0.0, 0.0]),
            )
            cyl.apply_translation([0.2, 0.0, 0.8])
            np.savez(tmp_path / "tube0.npz", vertices=np.asarray(cyl.vertices), faces=np.asarray(cyl.faces))

            scene = {
                "task": "insert_tubes",
                "task_objects": {
                    "slot": {"geometry_file": "slot.npz", "position": [0.0, 0.0, 0.8], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                    "tube0": {"geometry_file": "tube0.npz", "position": [0.2, 0.0, 0.8], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
                },
            }
            adapter = TaskAdapter("insert_tubes", _dummy_layout())
            with self.assertRaisesRegex(ValueError, "qualifying slot holes"):
                adapter.placement(scene, tmp_path)

    def test_candidates_insert_tubes_samples_long_axis_with_intact_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Lying cylinder along X axis on table
            cyl = trimesh.creation.cylinder(
                radius=0.014,
                height=0.12,
                sections=24,
                transform=trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0]),
            )
            cyl.apply_translation([0.2, 0.0, 0.764])
            obj_path = tmp_path / "object_geometry.npz"
            np.savez(obj_path, vertices=np.asarray(cyl.vertices), faces=np.asarray(cyl.faces))

            from soft_sim.robots.fingers import measured_installation
            from soft_sim.robots.mechanism import mechanism

            arm = mechanism("x5")
            scene = {
                "task": "insert_tubes",
                "robot_name": arm.name,
                "table_z": 0.73,
                "hand_mount_offset_m": 0.03,
                "adapter_offset_m": 0.005,
                "finger_scale": 0.5,
                "finger_mesh": "assets/meshes/soft_finger.stl",
                "finger_scale_to_m": 0.001,
                "finger_installation": measured_installation(0.5, mechanism=arm),
                "object_geometry_file": "object_geometry.npz",
            }
            # Use real scene.json from an existing unit fixture if finger_mesh is relative, or skip if asset missing
            from soft_sim.config import ROOT

            if not (ROOT / scene["finger_mesh"]).exists() or arm.missing_assets():
                self.skipTest("Robot or finger assets not available")

            (tmp_path / "scene.json").write_text(json.dumps(scene))
            generate_candidates(tmp_path, tmp_path / "out", contact_height=0.028)
            candidates = json.loads((tmp_path / "out" / "candidates.json").read_text())
            assert len(candidates) > 0
            axial_offsets = {c["axial_offset_m"] for c in candidates}
            assert any(abs(val) > 0.01 for val in axial_offsets)
            # Verify intact object.npz was not modified
            with np.load(obj_path) as data:
                assert len(data["vertices"]) == len(cyl.vertices)

    def test_pickup_phases_release_width_and_legacy_fallback(self):
        paths = {
            k: np.zeros((4, 7), dtype=float)
            for k in ("pregrasp", "approach", "probe", "lift", "transfer", "place", "release", "retreat", "return_home")
        }
        plan_v2 = {
            "initial_q": np.zeros(9, dtype=float).tolist(),
            "candidate": {"opened": 0.08, "closed": 0.02, "release_width": 0.046},
            "placement": {"release_width_m": 0.046},
        }
        phases_v2 = {name: width for name, _, _, width in pickup_phases(plan_v2, paths)}
        self.assertAlmostEqual(phases_v2["place"], 0.02)
        self.assertAlmostEqual(phases_v2["release"], 0.046)
        self.assertAlmostEqual(phases_v2["retreat"], 0.046)
        self.assertAlmostEqual(phases_v2["return_home"], 0.08)
        self.assertAlmostEqual(phases_v2["verify_hold"], 0.08)

        # Legacy plan without release_width preserves opened everywhere
        plan_legacy = {
            "initial_q": np.zeros(9, dtype=float).tolist(),
            "candidate": {"opened": 0.08, "closed": 0.02},
            "placement": {},
        }
        phases_legacy = {name: width for name, _, _, width in pickup_phases(plan_legacy, paths)}
        self.assertAlmostEqual(phases_legacy["release"], 0.08)
        self.assertAlmostEqual(phases_legacy["retreat"], 0.08)
        self.assertAlmostEqual(phases_legacy["return_home"], 0.08)

        # general_pickup without placement stops at verify_hold with closed
        plan_pickup = {
            "initial_q": np.zeros(9, dtype=float).tolist(),
            "candidate": {"opened": 0.08, "closed": 0.02},
        }
        phases_pickup = pickup_phases(plan_pickup, paths)
        assert phases_pickup[-1][0] == "verify_hold"
        self.assertAlmostEqual(phases_pickup[-1][3], 0.02)

    def test_grasp_envelope_status_detects_drop_and_slip(self):
        h_init = hand_from_object_transform([0.0, 0.0, 0.90], np.eye(3), [0.0, 0.0, 0.85], [1.0, 0.0, 0.0, 0.0])
        h_ok = hand_from_object_transform([0.1, 0.0, 0.95], np.eye(3), [0.1, 0.0, 0.90], [1.0, 0.0, 0.0, 0.0])
        ok = grasp_envelope_status([0.1, 0.0, 0.90], [0.1, 0.0, 0.95], initial_hand_from_object=h_init, current_hand_from_object=h_ok)
        assert ok["in_envelope"] is True
        self.assertAlmostEqual(ok["slip_m"], 0.0)

        # Object dropped onto table while hand stayed elevated
        h_drop = hand_from_object_transform([0.1, 0.0, 0.95], np.eye(3), [0.1, 0.0, 0.77], [1.0, 0.0, 0.0, 0.0])
        lost = grasp_envelope_status([0.1, 0.0, 0.77], [0.1, 0.0, 0.95], initial_hand_from_object=h_init, current_hand_from_object=h_drop)
        assert lost["in_envelope"] is False
        assert lost["mount_distance_m"] > 0.16

    def test_replan_placement_curobo_pose_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            class FakePlanner:
                world_report = {"objects": ["slot"]}

                def solve_ik_to_joint(self, q, pose, real_robot_pose=None):
                    return {"status": "Success", "joint_value": np.asarray(q, dtype=float) + 0.01}

                def plan_path(self, q0, goal_pose, real_robot_pose=None):
                    return {"status": "Success", "position": np.vstack([q0, np.asarray(q0) + 0.05])}

                def plan_joint(self, q0, q1):
                    return {"status": "Success", "position": np.vstack([q0, q1])}

            plan = {
                "initial_q": np.zeros(9, dtype=float).tolist(),
                "world_from_robot_base": np.eye(4, dtype=float).tolist(),
                "scene": {"robot_name": "x5", "table_z": 0.75},
            }
            world_from_goal_object = np.eye(4, dtype=float)
            world_from_goal_object[:3, 3] = [0.1, -0.05, 0.80]
            hand_from_object = np.eye(4, dtype=float)
            hand_from_object[:3, 3] = [0.0, 0.0, 0.05]

            paths, revision = replan_placement_curobo(
                tmp_path,
                plan,
                np.zeros(9, dtype=float),
                world_from_goal_object,
                hand_from_object,
                planner=FakePlanner(),
            )
            assert set(paths.keys()) == {"transfer", "place", "release", "retreat", "return_home"}
            goal_hand = np.asarray(revision["world_from_goal_hand"], dtype=float)
            # goal_hand @ hand_from_object must equal world_from_goal_object
            assert np.allclose(goal_hand @ hand_from_object, world_from_goal_object)
            assert revision["world_collision"] == {"objects": ["slot"]}
