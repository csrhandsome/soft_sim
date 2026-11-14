"""Planning/execution contracts for real drives and repeated pick/place phases."""
import unittest

import numpy as np

from soft_sim.execution import (
    layout_target_parameters, pickup_phases, settled_phase_pairs, support_placement_translation, trajectory_command,
)


class MotionExecution(unittest.TestCase):
    def test_layout_can_be_enriched_by_original_scene_manager(self):
        record = dict(label="block_1", category="block", physics=dict(mass=.25))
        layout = dict(Rigid=dict(block=[record]), version=1)
        result = layout_target_parameters(layout, "block_1")
        self.assertEqual(result, record)
        self.assertIsNot(result, record)
        self.assertEqual(layout_target_parameters(layout, "missing"), {})

    def test_joint_velocity_matches_time_scaled_path_and_jaw_derivative(self):
        path = np.array([[0., 1.], [.4, .8], [.8, .6]])
        t, epsilon = .8, 1e-5
        position, velocity, width = trajectory_command(path, 2., t, .088, .02)
        before = trajectory_command(path, 2., t-epsilon, .088, .02)[0]
        after = trajectory_command(path, 2., t+epsilon, .088, .02)[0]
        np.testing.assert_allclose(velocity, (after-before)/(2*epsilon), atol=1e-9)
        self.assertAlmostEqual(position[-2:].sum(), width)
        np.testing.assert_allclose(position[:2], [.32, .84])

    def test_single_waypoint_hold_has_no_arm_motion(self):
        positions, velocities, width = trajectory_command([[.5, -.2]], 1., 1., .02, .088)
        np.testing.assert_allclose(positions, [.5, -.2, .044, .044])
        np.testing.assert_allclose(velocities, 0.)
        self.assertEqual(width, .088)

    def test_repeated_holds_do_not_measure_the_transfer_between_them(self):
        rows = [dict(phase=phase, time_s=i*.1) for i, phase in enumerate(
            ["hold"]*8+["transfer"]*3+["hold"]*8)]
        pairs = settled_phase_pairs(rows, "hold")
        self.assertEqual(pairs, [(3, 4), (4, 5), (5, 6), (6, 7),
                                 (14, 15), (15, 16), (16, 17), (17, 18)])
        self.assertNotIn((7, 11), pairs)

    def test_place_preserves_size_and_aligns_bottom_with_support_top(self):
        target = np.array([[.2, -.2, .765], [.235, -.165, .8]])
        support = np.array([[-.1, -.1, .765], [-.065, -.065, .8]])
        shifted = target+support_placement_translation(target, support)
        np.testing.assert_allclose(shifted.mean(0)[:2], support.mean(0)[:2])
        self.assertAlmostEqual(shifted[:, 2].min(), support[:, 2].max()+.002)
        np.testing.assert_allclose(np.ptp(shifted, axis=0), np.ptp(target, axis=0))

    def test_stack_execution_releases_retracts_and_returns_home(self):
        plan = dict(initial_q=[0., 0., .044, .044], candidate=dict(opened=.088, closed=.02), placement={})
        paths = {key: np.array([[0., 0.], [.1, .2]]) for key in
                 ["pregrasp", "approach", "probe", "lift", "transfer", "place", "release", "retreat", "return_home"]}
        phases = pickup_phases(plan, paths)
        self.assertEqual([phase[0] for phase in phases][-6:],
                         ["transfer", "place", "release", "retreat", "return_home", "verify_hold"])
        self.assertEqual(next(p[3] for p in phases if p[0] == "transfer"), .02)
        self.assertTrue(all(p[3] == .088 for p in phases[-4:]))


if __name__ == "__main__":
    unittest.main()
