"""Unit tests for formal dataset delivery routing and CLI registration."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from soft_sim.commands import command, resolve
from soft_sim.config import ROOT
from soft_sim.storage import Storage, ensure_formal_datasets_ready


class DatasetDeliveryUnit(unittest.TestCase):
    def test_validate_dataset_command_registered(self):
        self.assertEqual(resolve("validate_dataset"), "validate-dataset")
        spec = command("validate-dataset")
        self.assertEqual(spec["env"], "dataset")
        self.assertEqual(spec["module"], "soft_sim.cli.validate_dataset")

    def test_validate_dataset_cli_parses_task_or_path(self):
        from soft_sim.cli.validate_dataset import main
        self.assertTrue(callable(main))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Requires numpy")
    def test_dataset_root_for_routes_formal_success_directly_under_data_datasets_task(self):
        from soft_sim.tasks.collect import dataset_root_for

        storage = Storage.create(ROOT / "artifacts")
        cfg = SimpleNamespace(
            task=SimpleNamespace(name="general_pickup", robot="x5", joint_control="position"),
            recording=SimpleNamespace(dataset_id="x5_90a_general_pickup_success_v1", success_only=True),
        )
        dataset_root, dataset_id = dataset_root_for(cfg, storage)
        self.assertEqual(dataset_root, ROOT / "data/datasets/general_pickup")
        self.assertEqual(dataset_id, "x5_90a_general_pickup_success_v1")

        diag_cfg = SimpleNamespace(
            task=SimpleNamespace(name="general_pickup", robot="x5", joint_control="position"),
            recording=SimpleNamespace(dataset_id="diag_v1", success_only=False),
        )
        diag_root, _ = dataset_root_for(diag_cfg, storage)
        self.assertEqual(diag_root, storage.dataset_dir("diag_v1"))

    def test_ensure_formal_datasets_ready_protects_against_dangling_symlink_and_unmounted_target(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            proj = Path(tmp)
            (proj / "data").mkdir()

            # Normal directory works without external mount
            (proj / "data/datasets").mkdir()
            ensure_formal_datasets_ready(project_root=proj)
            (proj / "data/datasets").rmdir()

            # Dangling symlink must be rejected
            (proj / "data/datasets").symlink_to(proj / "missing_external_drive")
            with self.assertRaisesRegex(RuntimeError, "dangling"):
                ensure_formal_datasets_ready(project_root=proj)
            (proj / "data/datasets").unlink()

            # Valid symlink with storage marker checks mountpoint
            ext_target = proj / "ext_mount/soft_sim/datasets"
            ext_target.mkdir(parents=True)
            (proj / "data/datasets").symlink_to(ext_target)
            (proj / "data/.datasets_storage_target.json").write_text(
                json.dumps({"mountpoint": str(proj / "ext_mount"), "target": str(ext_target)})
            )
            with patch("soft_sim.storage.os.path.ismount", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "not mounted"):
                    ensure_formal_datasets_ready(project_root=proj)
            with patch("soft_sim.storage.os.path.ismount", return_value=True):
                ensure_formal_datasets_ready(project_root=proj)

            # Regular directory when storage marker exists must be rejected (no system-disk fallback)
            (proj / "data/datasets").unlink()
            (proj / "data/datasets").mkdir()
            with self.assertRaisesRegex(RuntimeError, "regular directory"):
                ensure_formal_datasets_ready(project_root=proj)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "Requires numpy")
    def test_episode_recorder_preserves_logical_dataset_root_across_symlink(self):
        import numpy as np
        from soft_sim.recording.capture import EpisodeRecorder

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            proj = Path(tmp)
            ext_target = proj / "hik_mount/soft_sim/datasets"
            ext_target.mkdir(parents=True)
            (proj / "data").mkdir()
            (proj / "data/datasets").symlink_to(ext_target)
            cap = proj / "artifacts/runs/r1/capture"
            rec = EpisodeRecorder(cap, 20, {}, dataset_root=proj / "data/datasets/general_pickup")
            rec.close()
            meta = json.loads((cap / "episode.json").read_text())
            self.assertNotIn("hik_mount", meta["dataset_root"])
            self.assertTrue(meta["dataset_root"].endswith("data/datasets/general_pickup"))


if __name__ == "__main__":
    unittest.main()
