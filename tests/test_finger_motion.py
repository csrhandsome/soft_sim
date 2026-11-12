"""Independent fixtures separate arm travel, elastic motion and camera slip."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.check_finger_motion import check
from benchmarks.isaac51_tool import base_pin_mask
from shared.geometry import rotation_y


class FingerMotionMeasurements(unittest.TestCase):
    def test_support_layer_is_preserved_across_cooking_quantization(self):
        # Actual cooking moved the 4 mm layer to about 4.008 mm. The
        # support should retain three layers without clamping the 6 mm layer.
        for spacing in (.001998, .002, .0020042):
            nodes = np.zeros((5, 3))
            nodes[:, 2] = .0000286 + np.arange(5) * spacing
            pin, _ = base_pin_mask(nodes, .004, .002)
            np.testing.assert_array_equal(pin, [True, True, True, False, False])

    def make_recording(self, folder, camera_slip_m=0., vibration_m=0.):
        folder = Path(folder)
        (folder / "frames").mkdir()
        rest = np.array([[0., 0., 0.], [.002, 0., 0.], [0., 0., .02], [.002, 0., .02]])
        pin = np.array([True, True, False, False])
        np.savez(folder / "finger_reference.npz", finger_reference_left=rest, finger_reference_right=rest,
                 finger_pin_left=pin, finger_pin_right=pin)
        rows = []
        for i in range(11):
            mounts = np.repeat(np.eye(4)[None], 2, axis=0)
            mounts[:, :3, :3] = rotation_y(i * .1)
            mounts[:, :3, 3] = [[.3 + i * .01, -.03, .2], [.3 + i * .01, .03, .2]]
            local = rest.copy()
            if i:
                local[2, 0] += .005
                local[2:, 0] += (-1)**i * vibration_m
            nodes = [local @ m[:3, :3].T + m[:3, 3] for m in mounts]
            camera_local = np.eye(4)
            camera_local[:3, 3] = [camera_slip_m if i else 0., 0., .006]
            filename = f"frames/frame_{i:06d}.npz"
            np.savez(folder / filename, finger_mounts=mounts, finger_camera_to_world=mounts @ camera_local,
                     finger_nodes_left=nodes[0], finger_nodes_right=nodes[1])
            rows.append(dict(file=filename, time_s=(i + 1) * .04, phase="approach" if i == 0 else "hold"))
        calibration = dict(finger_intrinsics=[[[150., 0., 127.5], [0., 150., 127.5], [0., 0., 1.]]] * 2,
                           finger_resolution=[256, 256])
        (folder / "episode.json").write_text(json.dumps(dict(status="complete", frames=rows, calibration=calibration)))

    def test_rigid_motion_is_removed_and_elastic_change_remains(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_recording(folder)
            report = check(folder)
        self.assertTrue(report["camera_mount_valid"])
        self.assertTrue(report["hold_projected_motion_valid"])
        self.assertLess(report["maximum_pinned_node_error_mm"], 1e-9)
        np.testing.assert_allclose(report["phases"]["approach"]["maximum_displacement_mm"], 0., atol=1e-9)
        np.testing.assert_allclose(report["phases"]["hold"]["maximum_change_from_precontact_mm"], 5., atol=1e-9)
        np.testing.assert_allclose(report["phases"]["hold"]["settled_max_frame_node_delta_mm"], 0., atol=1e-9)
        np.testing.assert_allclose(report["phases"]["hold"]["settled_max_frame_p95_projected_node_delta_px"], 0., atol=1e-9)

    def test_submillimeter_vibration_near_lens_fails_pixel_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_recording(folder, vibration_m=.0001)
            report = check(folder)
        self.assertTrue(report["camera_mount_valid"])
        self.assertFalse(report["hold_projected_motion_valid"])
        self.assertFalse(report["valid"])
        # 0.2 mm of motion, at 14 mm optical depth and 150 px focal length.
        np.testing.assert_allclose(report["phases"]["hold"]["settled_max_frame_p95_projected_node_delta_px"],
                                   150. * .0002 / .014, atol=1e-9)

    def test_missing_intrinsics_does_not_claim_projection_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_recording(folder)
            path = Path(folder) / "episode.json"
            metadata = json.loads(path.read_text())
            del metadata["calibration"]
            path.write_text(json.dumps(metadata))
            report = check(folder)
        self.assertIsNone(report["hold_projected_motion_valid"])

    def test_invalid_limits_are_rejected(self):
        for value in (float("nan"), float("inf"), 0., -1.):
            with self.assertRaises(ValueError):
                check("unused", maximum_hold_p95_px=value)

    def test_camera_slip_is_reported_independently(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_recording(folder, camera_slip_m=.001)
            report = check(folder)
        self.assertFalse(report["camera_mount_valid"])
        self.assertAlmostEqual(report["maximum_camera_mount_translation_error_mm"], 1.)
        np.testing.assert_allclose(report["phases"]["hold"]["maximum_displacement_mm"], 5., atol=1e-9)

    def test_stable_camera_cannot_hide_clamp_slip(self):
        with tempfile.TemporaryDirectory() as folder:
            self.make_recording(folder)
            path = Path(folder) / 'frames/frame_000010.npz'
            with np.load(path) as f:
                values = {key: f[key].copy() for key in f.files}
            values['finger_nodes_left'][0, 0] += .0001
            np.savez(path, **values)
            report = check(folder)
        self.assertTrue(report['camera_mount_valid'])
        self.assertFalse(report['clamp_valid'])
        self.assertFalse(report['valid'])


if __name__ == "__main__":
    unittest.main()
