"""Task recipes preserve ordering, initial memory and live placement semantics."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from soft_sim.tasks.adapters import TaskAdapter, rotation
from soft_sim.execution import placement_transform_from_pose


def layout(labels):
    return {"Rigid": {"test": [dict(label=label, category_idx=index)
                               for label, index in labels]}}


class TaskAdapters(unittest.TestCase):
    def test_align_blocks_uses_official_set_square_label(self):
        # Labels from official align_blocks_0.json; no downloaded models needed.
        adapter = TaskAdapter("align_blocks", layout([
            ("target", 0), *[(f"cube{i}", 0) for i in range(3)]]))
        self.assertEqual([op.target for op in adapter.operations], ["target"] * 3)
        self.assertEqual([op.references for op in adapter.operations],
                         [(f"cube{i}",) for i in range(3)])

    def test_digits_use_actual_category_value_and_all_pads(self):
        adapter = TaskAdapter("arrange_largest_number", layout([
            ("digit_0", 21), ("digit_1", 19), ("digit_2", 3), ("digit_3", 15),
            *[(f"mat_{i}", 0) for i in range(4)]]))
        self.assertEqual([op.target for op in adapter.operations],
                         ["digit_1", "digit_3", "digit_2", "digit_0"])
        self.assertEqual([op.references for op in adapter.operations], [(f"mat_{i}",) for i in range(4)])
        self.assertTrue(all(op.orientation_wxyz == (1., 0., 0., 0.) for op in adapter.operations))

    def test_dolls_match_original_reward_modulo_order(self):
        adapter = TaskAdapter("sort_nesting_dolls_by_size", layout([(f"doll{i}", n)
                                             for i, n in enumerate((7, 5, 9, 6, 8))]))
        self.assertEqual([op.target for op in adapter.operations], ["doll2", "doll4", "doll0", "doll3", "doll1"])
        self.assertEqual([op.xy for op in adapter.operations], [(-.24+.12*i, -.12) for i in range(5)])

    def test_partial_target_override_and_missing_steps_rejected(self):
        with self.assertRaisesRegex(ValueError, "all steps"):
            TaskAdapter("stack_blocks", {}, target_label="block_1")
        with self.assertRaisesRegex(ValueError, "required labels"):
            TaskAdapter("stack_blocks", layout([("block_1", 0)]))
        with self.assertRaisesRegex(ValueError, "No soft-finger"):
            TaskAdapter("unknown_task", {})

    def test_variable_counts_keep_stable_labels_with_missing_slots(self):
        def digits(count):
            return TaskAdapter("arrange_largest_number", layout(
                [(f"digit_{i}", i) for i in range(count)] + [(f"mat_{i}", 0) for i in range(count)]))
        self.assertEqual(digits(4).recording_labels, digits(5).recording_labels)
        self.assertEqual(len(digits(4).recording_labels), 10)

    def test_formal_task_root_does_not_nest_dataset_id_or_storage_root(self):
        from soft_sim.config import ExperimentConfig, ROOT
        from soft_sim.storage import Storage
        from soft_sim.tasks.collect import dataset_root_for
        cfg = ExperimentConfig()
        cfg.recording.dataset_id = "a_logical_label"
        with tempfile.TemporaryDirectory() as directory:
            storage = Storage.create(directory)
            root, label = dataset_root_for(cfg, storage)
            self.assertEqual(root, ROOT / "data/datasets/general_pickup")
            self.assertEqual(label, "a_logical_label")
            cfg.recording.success_only = False
            diagnostic_root, _ = dataset_root_for(cfg, storage)
            self.assertEqual(diagnostic_root, storage.dataset_dir(label))

    def test_swap_uses_initial_pose_after_target_has_moved(self):
        adapter = TaskAdapter("swap_T", layout([("t0", 0), ("t1", 1)]))
        scene = {"task_objects": {
            "t0": dict(position=[-.06, -.2, .79], quaternion_wxyz=[1., 0., 0., 0.]),
            "t1": dict(position=[.06, -.2, .79], quaternion_wxyz=[2**-.5, 0., 0., 2**-.5])}}
        adapter.observe_initial(scene)
        adapter.advance()
        scene["task_objects"]["t0"]["position"] = [0., .04, .79]
        goal = adapter.placement(scene, ".")
        np.testing.assert_allclose(goal[:3, 3], [-.06, -.2, .79])
        adapter.advance()
        goal = adapter.placement(scene, ".")
        np.testing.assert_allclose(goal[:3, 3], [.06, -.2, .79])
        np.testing.assert_allclose(goal[:3, :3], rotation([2**-.5, 0., 0., 2**-.5]))
        adapter.advance()
        self.assertIsNone(adapter.current)

    def test_support_is_recomputed_from_current_geometry_and_rotated_bottom(self):
        adapter = TaskAdapter("stack_bowls", layout([(f"bowl{i}", 0) for i in range(3)]))
        q = [2**-.5, 0., 2**-.5, 0.]
        scene = dict(table_z=.765, task_objects={
            "bowl1": dict(position=[.2, -.2, .8], quaternion_wxyz=q, geometry_file="target.npz"),
            "bowl0": dict(position=[-.1, -.1, .8], quaternion_wxyz=[1, 0, 0, 0], geometry_file="base.npz")})
        local = np.array([[-.01, -.02, -.03], [.01, .02, .03]])
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory)/"target.npz", vertices=local @ rotation(q).T + [.2, -.2, .8])
            for z in (.81, .92):
                np.savez(Path(directory)/"base.npz", vertices=[[-.12, -.12, .765], [-.08, -.08, z]])
                goal = adapter.placement(scene, directory)
                transformed = local @ goal[:3, :3].T + goal[:3, 3]
                np.testing.assert_allclose(transformed.mean(0)[:2], [-.1, -.1])
                self.assertAlmostEqual(transformed[:, 2].min(), z+.002)

    def test_pose_goal_moves_tcp_and_payload_as_one_rigid_body(self):
        initial = np.eye(4)
        initial[:3, :3] = Rotation.from_euler("xyz", [.3, -.2, 1.]).as_matrix()
        initial[:3, 3] = [.2, -.1, .8]
        q = Rotation.from_matrix(initial[:3, :3]).as_quat()
        scene = dict(initial_object_position=initial[:3, 3], initial_object_orientation_wxyz=np.r_[q[3], q[:3]])
        goal = np.eye(4)
        goal[:3, :3] = Rotation.from_euler("xyz", [1., .4, -.5]).as_matrix()
        goal[:3, 3] = [-.2, .1, .95]
        displacement = placement_transform_from_pose(scene, goal)
        np.testing.assert_allclose(displacement @ initial, goal, atol=1e-14)
        local_tcp = np.array([.01, .03, -.02, 1.])
        np.testing.assert_allclose(displacement @ initial @ local_tcp, goal @ local_tcp)
        for invalid in (np.zeros((4, 4)), np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1])):
            with self.assertRaises(ValueError):
                placement_transform_from_pose(scene, invalid)

    def test_language_variants_and_new_task_recipes(self):
        stack_lang = TaskAdapter("stack_blocks_by_language", layout([(f"block_{i}", i) for i in range(3)]))
        self.assertEqual([op.target for op in stack_lang.operations], ["block_1", "block_2"])
        classify_lang = TaskAdapter("classify_objects_by_language", layout(
            [(f"cat{c}_0", 0) for c in range(3)] + [(f"basket{c}", 0) for c in range(3)]))
        self.assertEqual([op.target for op in classify_lang.operations], ["cat0_0", "cat1_0", "cat2_0"])
        self.assertEqual(len(classify_lang.recording_labels), 12)
        bottles = TaskAdapter("put_bottles_into_dustbin", layout(
            [(f"bottle{i}", 0) for i in range(4)] + [("dustbin", 0)]))
        self.assertEqual(len(bottles.operations), 4)
        tubes = TaskAdapter("insert_tubes", layout([(f"tube{i}", 0) for i in range(3)] + [("slot", 0)]))
        self.assertEqual(len(tubes.operations), 3)
        coin = TaskAdapter("deposit_coin", layout([("coin0", 0), ("piggy_bank", 0)]))
        self.assertEqual(coin.operations[0].target_frame, "passive/center")
        pens = TaskAdapter("fill_pen_holder", layout([(f"target{i}", 0) for i in range(4)] + [("pen_holder", 0)]))
        self.assertEqual(pens.operations[0].mode, "container_tip_down")
        table = TaskAdapter("organize_table", layout(
            [(l, 0) for l in ("mouse", "mousemat", "keyboard", "garage", "cube_cushion", "alarm", "drawer")]))
        self.assertEqual([op.target for op in table.operations], ["mouse", "keyboard", "garage", "alarm"])
        toy = TaskAdapter("play_stacking_toy", layout([(f"block{i}", 0) for i in range(10)] + [("stack_base", 0)]))
        self.assertEqual(len(toy.operations), 10)
        mugs = TaskAdapter("hang_mugs", layout([(f"mug{i}", 0) for i in range(3)] + [("cup_holder", 0)]))
        self.assertEqual(mugs.operations[0].active_frame, "active/handle")

    def test_cover_blocks_sorts_by_x_and_uncovers_red_green_blue(self):
        adapter = TaskAdapter("cover_blocks", layout(
            [(l, 0) for l in ("red", "green", "blue", "cup0", "cup1", "cup2")]))
        with tempfile.TemporaryDirectory() as directory:
            for name, pos in (("cup0", [-.2, -.2, .8]), ("cup1", [0., -.2, .8]), ("cup2", [.2, -.2, .8])):
                v = np.array([[-.02, -.02, 0.], [.02, .02, .05]]) @ rotation([0, 1, 0, 0]).T + pos
                np.savez(Path(directory) / f"{name}.npz", vertices=v)
            scene = dict(table_z=.765, task_objects={
                "blue": dict(position=[-.2, 0., .78], quaternion_wxyz=[1, 0, 0, 0]),
                "red": dict(position=[.2, 0., .78], quaternion_wxyz=[1, 0, 0, 0]),
                "green": dict(position=[0., 0., .78], quaternion_wxyz=[1, 0, 0, 0]),
                "cup0": dict(position=[-.2, -.2, .8], quaternion_wxyz=[0, 1, 0, 0], geometry_file="cup0.npz"),
                "cup1": dict(position=[0., -.2, .8], quaternion_wxyz=[0, 1, 0, 0], geometry_file="cup1.npz"),
                "cup2": dict(position=[.2, -.2, .8], quaternion_wxyz=[0, 1, 0, 0], geometry_file="cup2.npz"),
            })
            goal0 = adapter.placement(scene, directory)
            np.testing.assert_allclose(goal0[:2, 3], [-.2, 0.])  # cup0 covers leftmost (blue)
            # After observe_initial, uncover order is red (cup2), green (cup1), blue (cup0).
            self.assertEqual([op.target for op in adapter.operations[3:]], ["cup2", "cup1", "cup0"])

    def test_solve_equation_selects_missing_digit_or_operator(self):
        eq_layout = {"Rigid": {
            "number": [dict(label="t0", category_idx=3), dict(label="t2", category_idx=4),
                       dict(label="t4", category_idx=7), dict(label="num_7", category_idx=7),
                       dict(label="num_1", category_idx=1)],
            "plus": [dict(label="op_plus", category_idx=0)],
            "minus": [dict(label="op_minus", category_idx=0)],
            "mat": [dict(label=f"mat{i}", category_idx=0) for i in range(5)],
        }}
        adapter = TaskAdapter("solve_equation", eq_layout)
        self.assertEqual(adapter.operations[0].target, "op_plus")
        self.assertEqual(adapter.operations[0].references, ("mat1",))

    def test_fill_pen_holder_rotates_horizontal_pen_vertically_tip_down(self):
        adapter = TaskAdapter("fill_pen_holder", layout(
            [(f"target{i}", 0) for i in range(4)] + [("pen_holder", 0)]))
        with tempfile.TemporaryDirectory() as directory:
            # Horizontal pen along +X (14 cm long, 1 cm wide) and 8 cm wide pen holder.
            np.savez(Path(directory) / "pen.npz",
                     vertices=[[-.07, -.005, -.005], [.07, .005, .005]])
            np.savez(Path(directory) / "holder.npz",
                     vertices=[[-.04, -.04, 0.], [.04, .04, .10]])
            check_frame = np.eye(4)
            check_frame[:3, 3] = [.06, 0., .77]
            scene = dict(table_z=.765, task_objects={
                "target0": dict(position=[0., 0., .77], quaternion_wxyz=[1, 0, 0, 0],
                                geometry_file="pen.npz",
                                functional_frames={"passive/check": [check_frame.tolist()]}),
                "pen_holder": dict(position=[0., 0., .765], quaternion_wxyz=[1, 0, 0, 0],
                                   geometry_file="holder.npz"),
            })
            goal = adapter.placement(scene, directory)
            # Check tip (+X in local frame) now points straight down (-Z in world).
            np.testing.assert_allclose(goal[:3, :3] @ [1., 0., 0.], [0., 0., -1.], atol=1e-6)

    def test_all_42_base_and_12_random_tasks_registered(self):
        from soft_sim.tasks.adapters import BASE_TASKS, IMPLEMENTED_TASKS, RANDOM_TASKS
        self.assertEqual(len(BASE_TASKS), 42)
        self.assertEqual(len(RANDOM_TASKS), 12)
        self.assertEqual(len(IMPLEMENTED_TASKS), 54)


if __name__ == "__main__":
    unittest.main()
