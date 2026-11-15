"""Failed task attempts must not reach the dataset writer."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from soft_sim.recording.capture import EpisodeRecorder


class SuccessOnlyCapture(unittest.TestCase):
    def build(self, root):
        recorder = EpisodeRecorder(root / "capture", 20, {}, dataset_root=root / "dataset")
        rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        recorder.append(rgb, rgb, rgb, time_s=.05, phase="hold", joint_positions=np.zeros(6))
        (recorder.directory / "finger_reference.npz").touch()
        return recorder

    def test_failed_or_unknown_result_discards_frames_without_invoking_writer(self):
        for result in ({"success": False}, {}, None):
            with self.subTest(result=result), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                recorder = self.build(root)
                with patch.object(recorder, "finalize") as writer:
                    recorder.close(complete=True, result=result)
                    recorder.close()
                writer.assert_not_called()
                metadata = json.loads((recorder.directory / "episode.json").read_text())
                self.assertEqual(metadata["status"], "discarded")
                self.assertTrue(metadata["success_only"])
                self.assertFalse((root / "dataset").exists())
                self.assertFalse((recorder.directory / "capture.partial.npy").exists())
                self.assertFalse((recorder.directory / "finger_reference.npz").exists())

    def test_success_reaches_writer_and_keeps_stream_until_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.build(Path(directory))
            with patch.object(recorder, "finalize") as writer:
                recorder.close(complete=True, result={"success": True})
            writer.assert_called_once_with()
            self.assertTrue((recorder.directory / "capture.partial.npy").is_file())
            self.assertEqual(recorder._read_metadata()["status"], "staged")

    def test_interrupted_success_is_not_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = self.build(Path(directory))
            with patch.object(recorder, "finalize") as writer:
                recorder.close(result={"success": True})
            writer.assert_not_called()
            self.assertEqual(recorder._read_metadata()["status"], "incomplete")
            self.assertTrue((recorder.directory / "capture.partial.npy").is_file())
