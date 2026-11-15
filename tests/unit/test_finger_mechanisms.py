"""One soft finger, two robot wrists: the mount tables must stay consistent.

These are data/geometry checks only; they never start Isaac Sim or a planner.
"""

import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from soft_sim.config import ExperimentConfig
from soft_sim.robots import fingers
from soft_sim.robots.mechanism import FRANKA, MECHANISM_NAMES, X5, mechanism

MECHANISMS = (X5, FRANKA)


def joint_axes(arm):
    """Finger slide axes as authored in the upstream URDF, per side."""
    robot = ET.parse(arm.urdf).getroot()
    axes = {}
    for joint in robot.findall("joint"):
        if joint.get("name") in arm.finger_joints:
            axes[joint.get("name")] = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
    return axes


class MechanismTables(unittest.TestCase):
    def test_slide_links_and_joints_match_the_upstream_urdf(self):
        for arm in MECHANISMS:
            with self.subTest(robot=arm.name):
                robot = ET.parse(arm.urdf).getroot()
                joints = {joint.get("name"): joint for joint in robot.findall("joint")}
                for side, slide in enumerate(arm.slides):
                    self.assertIn(slide.joint, joints)
                    joint = joints[slide.joint]
                    self.assertEqual(joint.get("type"), "prismatic")
                    self.assertEqual(joint.find("child").get("link"), slide.link)
                    self.assertEqual(joint.find("parent").get("link"), arm.hand_link)
                    origin = np.fromstring(joint.find("origin").get("xyz", "0 0 0"), sep=" ")
                    np.testing.assert_allclose(origin, slide.origin_m, atol=1e-9)
                self.assertEqual(arm.joint_names[-2:], tuple(
                    sorted(arm.finger_joints, key=arm.joint_names.index)))
                self.assertEqual(set(arm.arm_joints) | set(arm.finger_joints), set(arm.joint_names))

    def test_closing_axis_agrees_with_the_urdf_joint_axes(self):
        for arm in MECHANISMS:
            with self.subTest(robot=arm.name):
                axes = joint_axes(arm)
                for side, slide in enumerate(arm.slides):
                    axis = axes[slide.joint]
                    cosine = float(axis @ arm.closing_axis_in_hand)
                    # Side 0 travels toward -Y, side 1 toward +Y.
                    self.assertAlmostEqual(cosine, -1. if side == 0 else 1., places=9)

    def test_tool_rotation_is_a_proper_rotation_that_points_along_growth(self):
        for arm in MECHANISMS:
            with self.subTest(robot=arm.name):
                rotation = arm.rotation
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(float(np.linalg.det(rotation)), 1., places=12)
                self.assertAlmostEqual(arm.max_open_width_m, 2 * arm.jaw_travel_m)
                np.testing.assert_allclose(arm.closed_joint_q()[:-2], 0.)

    def test_finger_links_stay_parallel_to_the_hand_frame(self):
        for arm in MECHANISMS:
            with self.subTest(robot=arm.name):
                links = fingers.link_transforms(arm.closed_joint_q(), mechanism=arm)
                hand = links[arm.hand_link]
                for side, slide in enumerate(arm.slides):
                    np.testing.assert_allclose(links[slide.link][:3, :3], hand[:3, :3], atol=1e-12)
                    offset = np.linalg.inv(hand) @ links[slide.link]
                    # The URDF prismatic axis owns the travel direction: side 0
                    # travels toward -Y, side 1 toward +Y.
                    axis = arm.closing_axis_in_hand * (-1. if side == 0 else 1.)
                    np.testing.assert_allclose(offset[:3, 3], np.asarray(slide.origin_m) +
                                               axis * arm.jaw_travel_m, atol=1e-12)

    def test_unknown_mechanism_is_rejected(self):
        with self.assertRaises(ValueError):
            mechanism("kuka")
        self.assertEqual(MECHANISM_NAMES, ("x5", "franka"))


class InstallationGeometry(unittest.TestCase):
    def test_pads_nearly_touch_at_rest_on_both_wrists(self):
        for arm in MECHANISMS:
            with self.subTest(robot=arm.name):
                installation = fingers.measured_installation(.75, mechanism=arm)
                mesh = fingers.source_mesh().copy()
                mesh.apply_scale(.75)
                # Zero slide travel: the pads must leave exactly the rest gap.
                links = fingers.link_transforms(np.zeros(len(arm.joint_names)), mechanism=arm)
                hand = links[arm.hand_link]
                lateral, mounted = [], []
                for side, slide in enumerate(arm.slides):
                    pose = (np.linalg.inv(hand) @ links[slide.link]
                            @ fingers.finger_in_link(side, installation, arm))
                    mounted.append(mesh.vertices @ pose[:3, :3].T + pose[:3, 3])
                    lateral.append(mounted[-1] @ arm.closing_axis_in_hand)
                gap = lateral[1].min() - lateral[0].max()
                self.assertAlmostEqual(gap, installation["minimum_rest_gap_m"], places=6)
                # Both fingers sit on opposite sides of the closing midplane.
                self.assertGreater(lateral[1].min(), 0.)
                self.assertLess(lateral[0].max(), 0.)
                # Growth: the mounted fingers extend along the slide growth axis.
                growth = mounted[0] @ arm.growth_axis_in_hand
                self.assertGreater(growth.max() - growth.min(), .05)

    def test_x5_installation_numbers_are_pinned(self):
        """Replay and saved plans depend on these exact X5 values."""
        installation = fingers.measured_installation(.75)
        self.assertEqual(installation["profile"], "measured_wedge_v3")
        self.assertAlmostEqual(installation["closed_half_spacing_m"], .015707816907238048, places=15)
        self.assertAlmostEqual(installation["forward_offset_m"], .005118675399008811, places=15)
        self.assertAlmostEqual(installation["seat_width_m"], .024)
        fingers.validate_installation(installation, .75)

    def test_franka_installation_is_derived_from_the_same_shared_mesh(self):
        installation = fingers.measured_installation(.75, mechanism=FRANKA)
        self.assertEqual(installation["profile"], "derived_wedge_v1")
        # Geometry of the finger itself is robot independent.
        self.assertAlmostEqual(installation["closed_half_spacing_m"],
                               fingers.measured_installation(.75)["closed_half_spacing_m"], places=15)
        self.assertAlmostEqual(installation["forward_offset_m"],
                               fingers.measured_installation(.75)["forward_offset_m"], places=15)
        self.assertAlmostEqual(installation["seat_width_m"], .021)
        fingers.validate_installation(installation, .75, FRANKA)
        with self.assertRaises(ValueError):
            fingers.validate_installation(installation, .75)


class SceneJointOrder(unittest.TestCase):
    def test_articulation_order_is_reordered_into_planning_order(self):
        scene = dict(initial_q=[10, 11, 12, 13, 14, 15, 16, 17, 18],
                     joint_names=list(FRANKA.joint_names))
        np.testing.assert_allclose(fingers.scene_joint_vector(scene, FRANKA),
                                   [10, 11, 12, 13, 14, 15, 16, 17, 18])
        swapped = dict(scene, joint_names=list(FRANKA.joint_names[:-2]) +
                       list(reversed(FRANKA.joint_names[-2:])))
        np.testing.assert_allclose(fingers.scene_joint_vector(swapped, FRANKA),
                                   [10, 11, 12, 13, 14, 15, 16, 18, 17])
        with self.assertRaises(ValueError):
            fingers.scene_joint_vector(dict(initial_q=[0] * 8, joint_names=list(X5.joint_names)), FRANKA)


class TaskFileSelection(unittest.TestCase):
    def test_franka_is_a_task_file_option_and_x5_stays_the_default(self):
        from soft_sim.config import ROOT
        default = ExperimentConfig.load(ROOT / "configs/general_pickup.json")
        self.assertEqual(default.task.robot, "x5")
        self.assertIsNone(default.recording.dataset_id)
        franka = ExperimentConfig.load(ROOT / "configs/general_pickup_franka.json")
        self.assertEqual(franka.task.robot, "franka")
        self.assertEqual(franka.recording.dataset_id, "panda_pickup_v1")
        # Only the mechanism and collection name differ from the default task.
        self.assertEqual(franka.task.name, default.task.name)
        self.assertEqual(franka.planning.friction, default.planning.friction)
        self.assertEqual(franka.task.finger_scale, default.task.finger_scale)


class RobotConfigurationRules(unittest.TestCase):
    def test_robot_choice_and_jaw_limit(self):
        config = ExperimentConfig()
        self.assertEqual(config.task.robot, "x5")
        config.gripper.profile = "native_franka"
        config.gripper.open_jaw_width_m = .08
        config.task.robot = "franka"
        config.validate()
        config.gripper.open_jaw_width_m = .088
        with self.assertRaises(ValueError):
            config.validate()
        config.gripper.open_jaw_width_m = .08
        config.task.robot = "ur5"
        with self.assertRaises(ValueError):
            config.validate()

    def test_finger_scale_limits_hold_for_both_mechanisms(self):
        for arm in MECHANISMS:
            for scale in (.25, .5, .75, 1.):
                installation = fingers.measured_installation(scale, mechanism=arm)
                self.assertTrue(math.isfinite(installation["closed_half_spacing_m"]))
                self.assertGreater(installation["closed_half_spacing_m"], 0.)


if __name__ == "__main__":
    unittest.main()
