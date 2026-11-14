"""Finalize must be idempotent and must never duplicate an episode."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from soft_sim.recording.capture import EpisodeRecorder
from soft_sim.recording.finalize import finalize


class FinalizeRetry(unittest.TestCase):
    def build(self, root):
        recorder = EpisodeRecorder(root / "run/capture", 20,
            dict(robot_type="dual_x5_softfinger", joint_names=["j"], active_arm=1,
                 finger_scale=.75, finger_resolution=[256, 256],
                 external_resolution=[640, 480], finger_intrinsics=[[1., 0., 0.]]),
            dataset_root=root / "datasets/lerobot/x5_pickup_v1",
            repo_id="local/softfinger75a", task="pick up")
        for index in range(3):
            rgb = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            recorder.append(rgb, rgb, rgb, time_s=(index + 1) / 20, phase="hold",
                            joint_positions=np.full(6, index))
        recorder.close(complete=True, result=dict(success=True))
        return root / "run/capture"

    def test_repeated_finalize_returns_the_same_episode(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            capture = self.build(root)
            first = json.loads((capture / "episode.json").read_text())
            self.assertEqual(first["status"], "committed")
            again = finalize(capture)
            self.assertEqual(again["episode_index"], first["episode_index"])
            info = json.loads((root / "datasets/lerobot/x5_pickup_v1/meta/info.json").read_text())
            self.assertEqual(info["total_episodes"], 1)
            self.assertFalse((capture / "capture.partial.npy").exists())

    def test_interrupted_capture_keeps_the_staged_stream(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            recorder = EpisodeRecorder(root / "run/capture", 20, {},
                dataset_root=root / "datasets/lerobot/x5_pickup_v1")
            rgb = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            recorder.append(rgb, rgb, rgb, time_s=.05, phase="hold", joint_positions=np.zeros(6))
            recorder.close()
            metadata = json.loads((recorder.directory / "episode.json").read_text())
            self.assertEqual(metadata["status"], "incomplete")
            self.assertTrue((recorder.directory / "capture.partial.npy").is_file())
            with self.assertRaises(Exception):
                finalize(recorder.directory)


if __name__ == "__main__":
    unittest.main()
