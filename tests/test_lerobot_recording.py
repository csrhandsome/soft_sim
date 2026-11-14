"""Run with envs/dataset/.venv/bin/python -m unittest tests.test_lerobot_recording."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.episode_recording import EpisodeRecorder
from benchmarks.episode_reader import open_frame, _dataset


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "Run in envs/dataset")
class LeRobotRoundTrip(unittest.TestCase):
    def test_two_episodes_append_and_preserve_state_and_depth(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            for episode in range(2):
                calibration = dict(robot_type="dual_x5_softfinger",
                    joint_names=[f"robot{arm}/joint{joint}" for arm in range(2) for joint in range(1,9)])
                recorder = EpisodeRecorder(root / str(episode), 20, calibration, dataset_root=dataset)
                for index in range(4):
                    rgb = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
                    depth = np.full((2,32,32), index+.125)
                    depth[0,0,0] = np.nan
                    recorder.append(rgb, rgb, rgb, time_s=(index+1)/20, phase="hold",
                        joint_positions=np.full(16, index+episode),
                        commanded_joint_positions=np.full(16, index+1),
                        operation_index=episode,
                        task_object_positions=np.full((3, 3), index+.5),
                        finger_depth_m=depth,
                        finger_nodes_left=np.full((12,3), index+.25))
                recorder.close(complete=True, result=dict(success=bool(episode)))
                meta = json.loads((recorder.directory / "episode.json").read_text())
                self.assertEqual(meta["episode_index"], episode)
                self.assertEqual(meta["status"], "complete")
                self.assertFalse((recorder.directory / "capture.partial.npy").exists())
                with open_frame(recorder.directory, meta["frames"][2]) as frame:
                    np.testing.assert_array_equal(frame["joint_positions"], np.full(16, 2+episode))
                    expected_depth = np.full((2,32,32),2.125)
                    expected_depth[0,0,0] = np.nan
                    np.testing.assert_array_equal(frame["finger_depth_m"], expected_depth)
                    self.assertEqual(frame["left_rgb"].shape, (32,32,3))
                    np.testing.assert_array_equal(frame["task_object_positions"], np.full((3, 3), 2.5))
                    np.testing.assert_array_equal(frame["operation_index"], [episode])
                    with open_frame(recorder.directory, meta["frames"][2], include_images=False) as state:
                        self.assertNotIn("left_rgb", state)
                        self.assertEqual(set(state), set(frame)-{"left_rgb", "right_rgb", "external_rgb"})
                        for key in state:
                            np.testing.assert_equal(state[key], frame[key])
                _dataset.cache_clear()
            info = json.loads((dataset / "meta/info.json").read_text())
            self.assertEqual(info["codebase_version"], "v3.0")
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 8)
            self.assertEqual(info["robot_type"], "dual_x5_softfinger")
            summary = json.loads((dataset / "meta/simulation/collection_summary.json").read_text())
            self.assertEqual(summary["successful_episode_indices"], [1])
            self.assertEqual(summary["failed_episode_indices"], [0])

    def test_interrupted_capture_is_not_a_training_episode(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
            recorder = EpisodeRecorder(Path(temporary)/"recording",20,{})
            rgb = np.random.default_rng(3).integers(0,255,(32,32,3),dtype=np.uint8)
            recorder.append(rgb,rgb,rgb,time_s=.05,phase="hold",joint_positions=np.zeros(9))
            checkpoint = json.loads((recorder.directory/"episode.json").read_text())
            self.assertEqual(checkpoint["frame_count"],1)
            self.assertEqual(checkpoint["state_keys"],["joint_positions"])
            self.assertEqual(checkpoint["status"],"incomplete")
            recorder.close()
            meta = json.loads((recorder.directory/"episode.json").read_text())
            self.assertEqual(meta["status"],"incomplete")
            self.assertFalse((recorder.root/"meta/info.json").exists())
