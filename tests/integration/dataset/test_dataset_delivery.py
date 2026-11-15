"""Integration tests for LeRobot v3 dataset delivery, validation, and safe finalize."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from soft_sim.contracts import ContractError
from soft_sim.recording.capture import EpisodeRecorder
import soft_sim.recording.finalize as finalize_mod
from soft_sim.recording.finalize import finalize
from soft_sim.recording.lerobot_io import LeRobotRecorder
from soft_sim.recording.reader import _dataset, open_frame


def _make_rgb(seed, shape=(32, 32, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def _record_episode(capture_dir, dataset_root, *, frames=4, fps=20, success=True,
                    success_only=True, run_id="run_0", start_seed=0):
    calibration = dict(
        robot_type="dual_x5_softfinger",
        joint_names=[f"robot{arm}/joint{j}" for arm in range(2) for j in range(1, 9)],
        active_arm=1,
        finger_scale=0.75,
        finger_resolution=[32, 32],
        external_resolution=[32, 32],
        finger_intrinsics=[[1.0, 0.0, 0.0]],
    )
    recorder = EpisodeRecorder(
        capture_dir,
        fps,
        calibration,
        dataset_root=dataset_root,
        repo_id="local/softfinger",
        task="pick up object",
        run_id=run_id,
        success_only=success_only,
    )
    for idx in range(frames):
        left = _make_rgb(start_seed + idx * 3 + 1)
        right = _make_rgb(start_seed + idx * 3 + 2)
        ext = _make_rgb(start_seed + idx * 3 + 3)
        recorder.append(
            left,
            right,
            ext,
            time_s=(idx + 1) / fps,
            phase="grasp",
            joint_positions=np.full(16, idx + 0.1, dtype=np.float32),
            commanded_joint_positions=np.full(16, idx + 0.2, dtype=np.float32),
            object_position=np.array([0.1, 0.2, 0.3 + idx * 0.01], dtype=np.float32),
            finger_nodes_left=np.full((12, 3), idx + 0.5, dtype=np.float32),
            finger_nodes_right=np.full((12, 3), idx + 0.6, dtype=np.float32),
        )
    recorder.close(complete=True, result=dict(success=success, joint_control="position"))
    return recorder


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "Run in envs/dataset")
class DatasetDeliveryIntegration(unittest.TestCase):
    def tearDown(self):
        _dataset.cache_clear()

    def test_success_episode_commits_to_direct_task_root_with_both_mp4s_and_validates(self):
        from soft_sim.validation.dataset import validate_dataset

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            recorder = _record_episode(root / "artifacts/runs/r1/capture", task_root, frames=5, success=True)
            self.assertTrue(recorder.committed)
            self.assertFalse((recorder.directory / "capture.partial.npy").exists())
            meta = json.loads((recorder.directory / "episode.json").read_text())
            self.assertEqual(meta["status"], "committed")
            self.assertEqual(meta["episode_index"], 0)

            report = validate_dataset(task_root)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["total_episodes"], 1)
            self.assertEqual(report["total_frames"], 5)
            self.assertEqual(report["episodes"][0]["task_success"], True)
            self.assertIsNone(report["episodes"][0]["quality_valid"])
            self.assertIn("left_rgb", report["episodes"][0]["videos"])
            self.assertIn("right_rgb", report["episodes"][0]["videos"])
            self.assertIn("external_rgb", report["episodes"][0]["videos"])
            self.assertEqual(report["episodes"][0]["videos"]["left_rgb"]["decoded_frames"], 5)
            self.assertEqual(report["episodes"][0]["videos"]["right_rgb"]["decoded_frames"], 5)

            with open_frame(recorder.directory, meta["frames"][2]) as frame:
                self.assertEqual(frame["left_rgb"].shape, (32, 32, 3))
                self.assertEqual(frame["right_rgb"].shape, (32, 32, 3))
                np.testing.assert_allclose(frame["joint_positions"], np.full(16, 2.1, dtype=np.float32))
                np.testing.assert_allclose(frame["commanded_joint_positions"], np.full(16, 2.2, dtype=np.float32))

    def test_failed_episode_does_not_enter_formal_dataset(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/stack_blocks"
            recorder = _record_episode(
                root / "artifacts/runs/r_fail/capture", task_root, frames=4, success=False, success_only=True
            )
            self.assertFalse(recorder.committed)
            self.assertFalse(task_root.exists())
            meta = json.loads((recorder.directory / "episode.json").read_text())
            self.assertEqual(meta["status"], "discarded")
            self.assertFalse((recorder.directory / "capture.partial.npy").exists())

    def test_missing_or_corrupt_mp4_is_rejected_and_preserves_staged_stream(self):
        from soft_sim.validation.dataset import validate_dataset

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            _record_episode(root / "artifacts/runs/r1/capture", task_root, frames=4, success=True)

            # Corrupt right_rgb MP4 -> validate_dataset must reject with specific error
            right_mp4 = task_root / "videos/observation.images.right_rgb/chunk-000/file-000.mp4"
            right_mp4.write_bytes(b"corrupt-mp4-bytes")
            report = validate_dataset(task_root)
            self.assertFalse(report["valid"])
            self.assertTrue(any("right_rgb" in err for err in report["errors"]), report["errors"])

            # Delete left_rgb MP4 -> validate_dataset must reject with specific error
            left_mp4 = task_root / "videos/observation.images.left_rgb/chunk-000/file-000.mp4"
            left_mp4.unlink()
            report2 = validate_dataset(task_root)
            self.assertFalse(report2["valid"])
            self.assertTrue(any("left_rgb" in err for err in report2["errors"]), report2["errors"])

            # If finalize verification fails, capture.partial.npy must NOT be deleted
            task_root2 = root / "data/datasets/stack_blocks"
            cap2 = root / "artifacts/runs/r2/capture"
            recorder2 = EpisodeRecorder(
                cap2, 20, dict(robot_type="dual_x5_softfinger"),
                dataset_root=task_root2, run_id="r2", success_only=True,
            )
            for i in range(3):
                rgb = _make_rgb(100 + i)
                recorder2.append(rgb, rgb, rgb, time_s=(i + 1) / 20, phase="hold", joint_positions=np.zeros(16))
            # Stage without finalizing yet, then simulate verification failure in finalize
            recorder2.closed = True
            recorder2.stream.close()
            recorder2._metadata("staged", dict(success=True))
            with patch("soft_sim.recording.finalize._verify_written_episode", side_effect=ContractError("Simulated MP4 failure")):
                with self.assertRaisesRegex(ContractError, "Simulated MP4 failure"):
                    finalize(cap2)
            self.assertTrue((cap2 / "capture.partial.npy").is_file())
            failed_meta = json.loads((cap2 / "episode.json").read_text())
            self.assertEqual(failed_meta["status"], "failed")
            self.assertIn("Simulated MP4 failure", failed_meta["finalize_error"])

    def test_append_two_episodes_and_finalize_retry_no_duplicate(self):
        from soft_sim.validation.dataset import validate_dataset

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            rec0 = _record_episode(
                root / "artifacts/runs/r0/capture", task_root, frames=4, success=True, run_id="r0", start_seed=0
            )
            rec1 = _record_episode(
                root / "artifacts/runs/r1/capture", task_root, frames=6, success=True, run_id="r1", start_seed=50
            )
            report = validate_dataset(task_root)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["total_episodes"], 2)
            self.assertEqual(report["total_frames"], 10)
            self.assertEqual(report["episodes"][0]["videos"]["left_rgb"]["decoded_frames"], 4)
            self.assertEqual(report["episodes"][1]["videos"]["left_rgb"]["decoded_frames"], 6)

            # Retry finalize on already committed episode -> no duplicate
            finalize(rec0.directory)
            # Simulate crash after save_episode() wrote episode 1 but before committed/unlink
            ep1_meta = json.loads((rec1.directory / "episode.json").read_text())
            ep1_meta["status"] = "converting"
            (rec1.directory / "episode.json").write_text(json.dumps(ep1_meta, indent=2))
            (rec1.directory / "capture.partial.npy").write_bytes(b"stale-stream")
            recovered = finalize(rec1.directory)
            self.assertEqual(recovered["status"], "committed")
            self.assertEqual(recovered["episode_index"], 1)
            self.assertFalse((rec1.directory / "capture.partial.npy").exists())
            info = json.loads((task_root / "meta/info.json").read_text())
            self.assertEqual(info["total_episodes"], 2)

    def test_quality_valid_false_does_not_block_dataset_validity(self):
        from soft_sim.validation.dataset import validate_dataset

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            _record_episode(root / "artifacts/runs/r0/capture", task_root, frames=3, success=True)
            prov_path = task_root / "meta/simulation/episode_000000.json"
            prov = json.loads(prov_path.read_text())
            prov["quality_valid"] = False
            prov_path.write_text(json.dumps(prov, indent=2))

            report = validate_dataset(task_root)
            self.assertTrue(report["valid"], report["errors"])
            self.assertTrue(report["episodes"][0]["task_success"])
            self.assertIs(report["episodes"][0]["quality_valid"], False)

    def test_crash_right_after_save_episode_recovers_without_duplicate(self):
        from soft_sim.validation.dataset import validate_dataset

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            cap = root / "artifacts/runs/run_crash/capture"
            recorder = EpisodeRecorder(
                cap, 20,
                dict(robot_type="dual_x5_softfinger", joint_names=[f"j{i}" for i in range(16)]),
                dataset_root=task_root, run_id="run_crash", success_only=True,
            )
            for idx in range(4):
                rgb = _make_rgb(200 + idx)
                recorder.append(
                    rgb, rgb, rgb, time_s=(idx + 1) / 20, phase="hold",
                    joint_positions=np.full(16, idx, dtype=np.float32),
                    commanded_joint_positions=np.full(16, idx, dtype=np.float32),
                )
            recorder.closed = True
            recorder.stream.close()
            recorder._metadata("staged", dict(success=True))

            orig_close = LeRobotRecorder.close

            def close_then_crash(self, *, complete=False, result=None):
                res = orig_close(self, complete=complete, result=result)
                if complete:
                    raise RuntimeError("Simulated crash right after save_episode")
                return res

            with patch.object(LeRobotRecorder, "close", close_then_crash):
                with self.assertRaisesRegex(RuntimeError, "Simulated crash right after save_episode"):
                    finalize(cap)

            # Capture stream and pending journal must survive; total_episodes is already 1
            self.assertTrue((cap / "capture.partial.npy").is_file())
            failed_meta = json.loads((cap / "episode.json").read_text())
            self.assertEqual(failed_meta["status"], "failed")
            self.assertEqual(failed_meta["episode_index"], 0)
            pending_file = task_root / "meta/simulation/.pending_episode_000000.json"
            self.assertTrue(pending_file.is_file())
            info = json.loads((task_root / "meta/info.json").read_text())
            self.assertEqual(info["total_episodes"], 1)

            # Retry finalize -> must NOT append a second episode
            recovered = finalize(cap)
            self.assertEqual(recovered["status"], "committed")
            self.assertEqual(recovered["episode_index"], 0)
            self.assertFalse((cap / "capture.partial.npy").exists())
            self.assertFalse(pending_file.exists())
            info_after = json.loads((task_root / "meta/info.json").read_text())
            self.assertEqual(info_after["total_episodes"], 1)
            prov = json.loads((task_root / "meta/simulation/episode_000000.json").read_text())
            self.assertEqual(prov["run_id"], "run_crash")
            self.assertTrue(validate_dataset(task_root)["valid"])

    def test_recovery_refuses_to_claim_another_runs_episode(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            _record_episode(root / "artifacts/runs/run_a/capture", task_root, frames=4, success=True, run_id="run_a")
            cap_b = root / "artifacts/runs/run_b/capture"
            rec_b = EpisodeRecorder(
                cap_b, 20, dict(robot_type="dual_x5_softfinger"),
                dataset_root=task_root, run_id="run_b", success_only=True,
            )
            for idx in range(4):
                rgb = _make_rgb(300 + idx)
                rec_b.append(rgb, rgb, rgb, time_s=(idx + 1) / 20, phase="hold",
                             joint_positions=np.zeros(16), commanded_joint_positions=np.zeros(16))
            rec_b.closed = True
            rec_b.stream.close()
            meta_b = rec_b._metadata("failed", dict(success=True))
            meta_b["episode_index"] = 0
            (cap_b / "episode.json").write_text(json.dumps(meta_b, indent=2))
            with self.assertRaisesRegex(ContractError, "belongs to run_id='run_a'"):
                finalize(cap_b)

    def test_abnormal_timestamps_and_pts_rejected(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from soft_sim.validation.dataset import validate_dataset

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            _record_episode(root / "artifacts/runs/r_ts/capture", task_root, frames=4, success=True, run_id="r_ts")

            # 1. Corrupt video PTS (uneven gap) -> rejected
            bad_shard = dict(
                ok=True, path="fake.mp4", frame_count=4, fps=20.0,
                resolution=[32, 32], min_rgb_std=10.0, timestamps=[0.0, 0.05, 0.12, 0.15],
            )
            with patch("soft_sim.validation.dataset._decode_video_shard", return_value=bad_shard):
                report = validate_dataset(task_root)
            self.assertFalse(report["valid"])
            self.assertTrue(any("PTS drift/gap" in err for err in report["errors"]), report["errors"])

            # 2. Corrupt provenance.frames.time_s -> sim_time_s mismatch rejected
            prov_path = task_root / "meta/simulation/episode_000000.json"
            prov = json.loads(prov_path.read_text())
            orig_prov = json.loads(json.dumps(prov))
            prov["frames"][2]["time_s"] = 9.99
            prov_path.write_text(json.dumps(prov, indent=2))
            report_sim = validate_dataset(task_root)
            self.assertFalse(report_sim["valid"])
            self.assertTrue(any("observation.sim_time_s disagrees" in err for err in report_sim["errors"]), report_sim["errors"])
            prov_path.write_text(json.dumps(orig_prov, indent=2))

            # 3. Corrupt parquet timestamp (not frame_index / fps) -> rejected
            pq_path = task_root / "data/chunk-000/file-000.parquet"
            table = pq.read_table(pq_path)
            col_idx = table.schema.get_field_index("timestamp")
            new_ts = pa.array([0.5, 0.55, 0.60, 0.65], type=table.schema.field(col_idx).type)
            pq.write_table(table.set_column(col_idx, "timestamp", new_ts), pq_path)
            report_pq = validate_dataset(task_root)
            self.assertFalse(report_pq["valid"])
            self.assertTrue(any("frame_index / fps" in err for err in report_pq["errors"]), report_pq["errors"])

    def test_formal_capture_without_action_is_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            cap = root / "artifacts/runs/r_no_act/capture"
            rec = EpisodeRecorder(cap, 20, dict(robot_type="dual_x5_softfinger"),
                                  dataset_root=task_root, run_id="r_no_act", success_only=True)
            for idx in range(3):
                rgb = _make_rgb(400 + idx)
                rec.append(rgb, rgb, rgb, time_s=(idx + 1) / 20, phase="hold", joint_positions=np.zeros(16))
            rec.closed = True
            rec.stream.close()
            rec._metadata("staged", dict(success=True))
            with self.assertRaisesRegex(ContractError, "action"):
                finalize(cap)
            self.assertTrue((cap / "capture.partial.npy").is_file())

    def test_finalize_non_formal_target_succeeds_when_formal_storage_unmounted(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            diag_target = root / "artifacts/datasets/lerobot/diag_ep"
            with patch("soft_sim.recording.finalize.ensure_formal_datasets_ready",
                       side_effect=RuntimeError("Should not guard non-formal target")):
                rec = _record_episode(
                    root / "artifacts/runs/r_diag/capture", diag_target,
                    frames=3, success=False, success_only=False, run_id="r_diag",
                )
            self.assertTrue(rec.committed)

    def test_resolve_layout_handles_double_robodojo_path_and_rejects_missing(self):
        import hashlib
        from soft_sim.config import ROOT
        from soft_sim.recording.finalize import _resolve_layout

        real_layout = ROOT / "third_party/RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0/general_pickup_22.json"
        if not real_layout.is_file():
            self.skipTest(f"Missing official layout {real_layout}")
        expected_sha = hashlib.sha256(real_layout.read_bytes()).hexdigest()
        scene = {
            "layout": "/home/sustechdl/code/threecheng/soft_sim/third_party/RoboDojo/Assets/Eval_Layout/RoboDojo/arx_x5/0/general_pickup_22.json",
            "layout_sha256": expected_sha,
        }
        _resolve_layout(scene)
        self.assertIn("layout_snapshot", scene)
        with self.assertRaisesRegex(ContractError, "missing"):
            _resolve_layout({"layout": "/nonexistent/path/missing_layout.json", "layout_sha256": "abc"})

    def test_cross_machine_pending_journal_recovery_uses_run_id(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            cap = root / "artifacts/runs/run_cross/capture"
            rec = _record_episode(cap, task_root, frames=4, success=True, run_id="run_cross")
            # Simulate pending journal left from 4090 with a remote absolute capture_dir
            meta = json.loads((cap / "episode.json").read_text())
            meta["status"] = "converting"
            (cap / "episode.json").write_text(json.dumps(meta, indent=2))
            (task_root / "meta/simulation/episode_000000.json").unlink()
            (task_root / "meta/simulation/.pending_episode_000000.json").write_text(json.dumps({
                "episode_index": 0,
                "run_id": "run_cross",
                "capture_dir": "/home/sustechdl/code/threecheng/soft_sim/artifacts/runs/run_cross/capture",
                "frame_count": 4,
            }))
            recovered = finalize(cap)
            self.assertEqual(recovered["status"], "committed")
            self.assertEqual(recovered["episode_index"], 0)
            info = json.loads((task_root / "meta/info.json").read_text())
            self.assertEqual(info["total_episodes"], 1)

    def test_provenance_frames_length_mismatch_is_rejected(self):
        from soft_sim.validation.dataset import validate_dataset

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            task_root = root / "data/datasets/general_pickup"
            _record_episode(root / "artifacts/runs/r_prov/capture", task_root, frames=4, success=True, run_id="r_prov")
            prov_path = task_root / "meta/simulation/episode_000000.json"
            prov = json.loads(prov_path.read_text())
            prov["frames"] = prov["frames"][:2]
            prov_path.write_text(json.dumps(prov, indent=2))
            report = validate_dataset(task_root)
            self.assertFalse(report["valid"])
            self.assertTrue(any("provenance.frames length" in err for err in report["errors"]), report["errors"])


if __name__ == "__main__":
    unittest.main()
