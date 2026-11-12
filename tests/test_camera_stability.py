"""Known image translations test the RGB check independently of FEM metadata."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

if cv2 is not None:
    from benchmarks.check_camera_stability import check, feature_motion
    from benchmarks.episode_recording import EpisodeRecorder
    from benchmarks.validate_rgb_episode import validate


@unittest.skipUnless(cv2 is not None, "RGB stability tests use the RoboDojo environment's OpenCV")
class CameraStabilityMeasurements(unittest.TestCase):
    def texture(self):
        random = np.random.default_rng(31)
        gray = random.integers(0, 256, size=(128, 128), dtype=np.uint8)
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_LINEAR)
        return np.repeat(gray[..., None], 3, axis=2)

    def test_known_translation_and_stationary_frame(self):
        image = self.texture()
        translated = cv2.warpAffine(image, np.float32([[1, 0, 2], [0, 1, -1]]), (256, 256))
        self.assertLess(feature_motion(image, image)["p95_motion_px"], .001)
        self.assertAlmostEqual(feature_motion(image, translated)["p95_motion_px"], np.sqrt(5), delta=.03)

    def test_unmeasurable_images_do_not_pass(self):
        self.assertIsNone(feature_motion(np.zeros((256, 256, 3), np.uint8),
                                         np.zeros((256, 256, 3), np.uint8)))

    def test_rgb_jitter_is_rejected_without_camera_or_fem_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            image = self.texture()
            rows = []
            for i in range(12):
                frame = cv2.warpAffine(image, np.float32([[1, 0, 2 * (i % 2)], [0, 1, 0]]), (256, 256))
                name = f"{i}.npz"
                np.savez(folder / name, left_rgb=image, right_rgb=frame, external_rgb=image)
                rows.append(dict(file=name, phase="hold", time_s=.04 * (i + 1)))
            (folder / "episode.json").write_text(json.dumps(dict(status="complete", fps=25., frames=rows)))
            report = check(folder)
        self.assertFalse(report["valid"])
        self.assertTrue(report["phases"]["hold"]["left_rgb"]["passed"])
        self.assertFalse(report["phases"]["hold"]["right_rgb"]["passed"])

    def test_episode_validation_rejects_jitter_even_when_mount_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder) / "recording"
            calibration = dict(finger_resolution=[256, 256], external_resolution=[256, 256])
            recorder = EpisodeRecorder(folder, 25., calibration)
            image = self.texture()
            for i in range(12):
                shifted = cv2.warpAffine(image, np.float32([[1, 0, 2 * (i % 2)], [0, 1, 0]]), (256, 256))
                recorder.append(image, shifted, image, time_s=.04 * (i + 1), phase="hold",
                    joint_positions=np.zeros(9), joint_velocities=np.zeros(9), commanded_joint_positions=np.zeros(9),
                    object_position=np.zeros(3), object_orientation_wxyz=np.array([1., 0., 0., 0.]),
                    finger_camera_to_world=np.repeat(np.eye(4)[None], 2, axis=0))
            recorder.close(complete=True, result=dict(success=None))
            (folder / "finger_motion_validation.json").write_text(json.dumps(dict(valid=True,
                camera_mount_valid=True, physics_initialization_valid=True, hold_projected_motion_valid=True)))
            check(folder)
            report = validate(folder)
        self.assertTrue(report["data_integrity_valid"])
        self.assertTrue(report["camera_mount_valid"])
        self.assertTrue(report["hold_projected_motion_valid"])
        self.assertFalse(report["camera_rgb_hold_stability_valid"])
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()
