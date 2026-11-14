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
    from soft_sim.validation.camera_stability import check, feature_motion
    from soft_sim.validation.episode import validate


def write_episode(folder, frames, fps, calibration, result):
    """Write one complete plain-MP4 episode without an external recorder dependency."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "frames").mkdir()
    names = ("left_rgb", "right_rgb", "external_rgb")
    rows, writers = [], {}

    def writer(name, shape):
        path = folder / (name + ".partial.mp4")
        handle = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (shape[1], shape[0]))
        if not handle.isOpened():
            raise RuntimeError(f"Cannot open MP4 encoder: {path}")
        writers[name] = handle

    for index, frame in enumerate(frames):
        images = [np.asarray(frame[name]) for name in names]
        state = {key: value for key, value in frame.items()
                 if key not in names + ("time_s", "phase")}
        name = f"frames/frame_{index:06d}.npz"
        np.savez_compressed(folder / name, **dict(zip(names, images)),
                            time_s=frame["time_s"], **state)
        rows.append(dict(index=index, time_s=float(frame["time_s"]),
                         phase=frame["phase"], file=name))
        if not writers:
            for camera, image in zip(names, images):
                writer(camera, image.shape)
            writer("episode", (480, 1280, 3))
        for camera, image in zip(names, images):
            writers[camera].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        canvas = np.full((480, 1280, 3), 20, dtype=np.uint8)
        canvas[:, :640] = cv2.resize(images[2], (640, 480))
        for image, x in ((images[0], 640), (images[1], 960)):
            canvas[80:400, x:x + 320] = cv2.resize(image, (320, 320))
        writers["episode"].write(canvas)
    for handle in writers.values():
        handle.release()
    for camera in writers:
        (folder / (camera + ".partial.mp4")).rename(folder / (camera + ".mp4"))
    data = dict(status="complete", fps=float(fps), frame_count=len(rows),
                duration_s=len(rows) / fps, camera_order=list(names), color_order="RGB",
                camera_coordinates="optical: +X right, +Y down, +Z forward",
                calibration=calibration, frames=rows, result=result)
    (folder / "episode.json").write_text(json.dumps(data, indent=2))


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
            image = self.texture()
            frames = []
            for i in range(12):
                shifted = cv2.warpAffine(image, np.float32([[1, 0, 2 * (i % 2)], [0, 1, 0]]), (256, 256))
                frames.append(dict(left_rgb=image, right_rgb=shifted, external_rgb=image,
                    time_s=.04 * (i + 1), phase="hold",
                    joint_positions=np.zeros(9), joint_velocities=np.zeros(9), commanded_joint_positions=np.zeros(9),
                    object_position=np.zeros(3), object_orientation_wxyz=np.array([1., 0., 0., 0.]),
                    finger_camera_to_world=np.repeat(np.eye(4)[None], 2, axis=0)))
            write_episode(folder, frames, 25., calibration, dict(success=None))
            (folder / "finger_motion_validation.json").write_text(json.dumps(dict(valid=True,
                camera_mount_valid=True, physics_initialization_valid=True, hold_projected_motion_valid=True)))
            check(folder)
            report = validate(folder)
        self.assertTrue(report["data_integrity_valid"])
        self.assertTrue(report["camera_mount_valid"])
        self.assertTrue(report["hold_projected_motion_valid"])
        self.assertFalse(report["camera_rgb_hold_stability_valid"])
        self.assertFalse(report["valid"])

    def test_data_integrity_alone_does_not_pass_episode_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "recording"
            calibration = dict(finger_resolution=[256, 256], external_resolution=[256, 256])
            image = self.texture()
            frames = [dict(left_rgb=image, right_rgb=image, external_rgb=image,
                time_s=.04 * (index + 1), phase="hold",
                joint_positions=np.zeros(9), joint_velocities=np.zeros(9), commanded_joint_positions=np.zeros(9),
                object_position=np.zeros(3), object_orientation_wxyz=np.array([1., 0., 0., 0.]),
                finger_camera_to_world=np.repeat(np.eye(4)[None], 2, axis=0)) for index in range(2)]
            write_episode(folder, frames, 25., calibration, dict(success=True))
            report = validate(folder)
            self.assertTrue(report["data_integrity_valid"])
            self.assertTrue(report["grasp_success"])
            self.assertFalse(report["quality_checks_complete"])
            self.assertIn("camera_fem_projection_valid", report["missing_quality_checks"])
            self.assertFalse(report["valid"])
            (folder / "camera_projection_validation.json").write_text(json.dumps(dict(
                valid=True, initial_projection_valid=True, motion_sync_valid=True)))
            (folder / "fem_camera_projection_validation.json").write_text(json.dumps(dict(valid=True)))
            (folder / "finger_motion_validation.json").write_text(json.dumps(dict(valid=True,
                camera_mount_valid=True, clamp_valid=True, physics_initialization_valid=True,
                hold_projected_motion_valid=True)))
            (folder / "camera_stability_validation.json").write_text(json.dumps(dict(valid=True)))
            self.assertTrue(validate(folder)["valid"])
            (folder / "camera_projection_validation.json").write_text(json.dumps(dict(
                valid=True, initial_projection_valid=True, motion_sync_valid=None)))
            report = validate(folder)
            self.assertFalse(report["valid"])
            self.assertEqual(report["missing_quality_checks"], ["camera_motion_sync_valid"])


if __name__ == "__main__":
    unittest.main()
