"""Storage ownership: one root, immutable runs, one dataset location."""

import json
from pathlib import Path
import tempfile
import unittest


from soft_sim.config import ExperimentConfig
from soft_sim.storage import RunStore, Storage, make_run_id


class StorageOwnership(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Storage.create(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_run_ids_encode_utc_task_seed_and_are_unique(self):
        first, second = make_run_id("general_pickup", 22), make_run_id("general_pickup", 22)
        self.assertNotEqual(first, second)
        self.assertIn("general-pickup", first)
        self.assertIn("_s22_", first)
        self.assertRegex(first, r"^\d{8}T\d{6}Z_")

    def test_run_directory_is_never_reused(self):
        RunStore.create(self.storage, command="collect:collect", task="general_pickup", seed=1,
                        run_id="fixed")
        with self.assertRaises(FileExistsError):
            RunStore.create(self.storage, command="collect:collect", task="general_pickup", seed=1,
                            run_id="fixed")

    def test_manifest_records_states_and_storage_relative_artifacts(self):
        store = RunStore.create(self.storage, command="collect:collect", task="general_pickup",
                                seed=2, run_id="run-a")
        path = store.write_json("reports/initial/mount_validation.json", {"valid": True},
                                record="mount_validation")
        store.update(execution="running")
        store.update(execution="completed", task_success=True, capture="committed")
        manifest = json.loads((store.root / "manifest.json").read_text())
        self.assertEqual(manifest["execution"], "completed")
        self.assertEqual(manifest["capture"], "committed")
        self.assertIs(manifest["task_success"], True)
        self.assertEqual(manifest["artifacts"]["mount_validation"],
                         "runs/run-a/reports/initial/mount_validation.json")
        self.assertTrue(Path(self.storage.root / manifest["artifacts"]["mount_validation"]).is_file())
        self.assertEqual(path, store.root / "reports/initial/mount_validation.json")

    def test_manifest_updates_are_atomic_and_typed(self):
        store = RunStore.create(self.storage, command="validate", run_id="run-b")
        config = ExperimentConfig().validate()
        store.write_config_snapshot(config, {"task.seed": 5})
        snapshot = json.loads((store.root / "config.resolved.json").read_text())
        self.assertEqual(snapshot["_cli_overrides"], {"task.seed": 5})
        self.assertEqual(set(snapshot), set(config.to_dict()) | {"_format_version", "_cli_overrides"})
        self.assertFalse(list(store.root.glob("*.tmp")))

    def test_buckets_keep_datasets_and_cache_out_of_runs(self):
        self.assertEqual(self.storage.dataset_dir("x5_pickup_v1"),
                         self.storage.root / "datasets/lerobot/x5_pickup_v1")
        self.assertEqual(self.storage.cache_dir("meshes"),
                         self.storage.root / "cache/meshes")
        self.assertFalse(str(self.storage.dataset_dir("a")).startswith(
            str(self.storage.run_dir("b")).rsplit("/runs/", 1)[0] + "/runs/"))

    def test_reopened_run_reports_the_same_identity(self):
        created = RunStore.create(self.storage, command="validate", task="stack_blocks", seed=4,
                                  run_id="run-c")
        reopened = RunStore.open(self.storage, "run-c")
        self.assertEqual(reopened.command, created.command)
        self.assertEqual(reopened.manifest["task"], "stack_blocks")
        self.assertEqual(reopened.manifest["seed"], 4)


if __name__ == "__main__":
    unittest.main()
