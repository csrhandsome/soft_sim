"""Cross-process file contracts: versions, capture index, dataset signature."""

import unittest

from soft_sim.contracts import (
    CAPTURE_PROTOCOL_VERSION, ContractError, check_dataset_signature,
    dataset_signature, validate_capture_metadata,
)


def capture(**overrides):
    metadata = dict(
        storage_format="lerobot_v3", protocol_version=CAPTURE_PROTOCOL_VERSION,
        status="staged", fps=20., camera_order=["left_rgb", "right_rgb", "external_rgb"],
        color_order="RGB", frame_count=2, state_keys=["joint_positions"],
        frames=[dict(index=0, time_s=.05, phase="hold"),
                dict(index=1, time_s=.1, phase="hold")])
    metadata.update(overrides)
    return metadata


class CaptureContract(unittest.TestCase):
    def test_valid_index_passes(self):
        self.assertEqual(validate_capture_metadata(capture())["status"], "staged")

    def test_protocol_and_layout_are_enforced(self):
        with self.assertRaises(ContractError):
            validate_capture_metadata(capture(protocol_version=CAPTURE_PROTOCOL_VERSION + 1))
        with self.assertRaises(ContractError):
            validate_capture_metadata(capture(camera_order=["left_rgb"]))
        with self.assertRaises(ContractError):
            validate_capture_metadata(capture(color_order="BGR"))

    def test_frames_must_be_contiguous_and_counted(self):
        with self.assertRaises(ContractError):
            validate_capture_metadata(capture(frame_count=3))
        with self.assertRaises(ContractError):
            validate_capture_metadata(capture(frames=[dict(index=1, time_s=.05, phase="hold")]))

    def test_unknown_status_is_rejected(self):
        with self.assertRaises(ContractError):
            validate_capture_metadata(capture(status="finished"))


class DatasetSignature(unittest.TestCase):
    calibration = dict(robot_type="dual_x5_softfinger", joint_names=["a"], active_arm=1,
                       finger_scale=.75, finger_resolution=[256, 256],
                       external_resolution=[640, 480], finger_intrinsics=[[1., 0., 0.]])

    def test_signature_is_stable_and_named_on_mismatch(self):
        first = dataset_signature(self.calibration, 20)
        self.assertEqual(first, dataset_signature(self.calibration, 20))
        changed = dict(self.calibration, finger_scale=.8)
        with self.assertRaises(ContractError) as raised:
            check_dataset_signature(first, dataset_signature(changed, 20))
        self.assertIn("finger_scale", str(raised.exception))

    def test_fps_change_requires_a_new_dataset(self):
        with self.assertRaises(ContractError):
            check_dataset_signature(dataset_signature(self.calibration, 20),
                                    dataset_signature(self.calibration, 30))


if __name__ == "__main__":
    unittest.main()
