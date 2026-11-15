"""Tests for the YAML collection manifest parser and batch runner."""

import json
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from soft_sim.config import ROOT
from soft_sim.tasks.collection import load_collection_manifest, parse_collection_yaml, run_batch


class CollectionBatchTests(unittest.TestCase):
    def test_material_and_background_dependencies_checked_before_start(self):
        from soft_sim.sim.robodojo import layout_materials, validate_layout_assets
        layout = {"Table": {"default": "material_0261"},
                  "Ground": {"materials": {"default": "material_0042"}},
                  "Background": {"category_name": "scene.hdr"}}
        self.assertEqual(layout_materials(layout), ["material_0042", "material_0261"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("soft_sim.sim.robodojo.SNAPSHOT", root):
                with self.assertRaisesRegex(FileNotFoundError, "material_0261"):
                    validate_layout_assets(layout)
                for material in layout_materials(layout):
                    target = root / "Assets/Material" / material
                    target.mkdir(parents=True)
                    (target / "surface.mdl").write_text("mdl 1.7;\n")
                with self.assertRaisesRegex(FileNotFoundError, "scene.hdr"):
                    validate_layout_assets(layout)
                target = root / "Assets/Background/scene.hdr"
                target.parent.mkdir(parents=True)
                target.write_bytes(b"#?RADIANCE\n")
                validate_layout_assets(layout)

    def test_room_dependency_checked_before_simulator_start(self):
        from soft_sim.sim.robodojo import validate_layout_assets
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            room = root / "Assets/Room/Simple_Room_purple"
            with patch("soft_sim.sim.robodojo.SNAPSHOT", root):
                with self.assertRaisesRegex(FileNotFoundError, "Simple_Room_purple"):
                    validate_layout_assets({"Room": {"default": "Simple_Room_purple"}})
                room.mkdir(parents=True)
                (room / "room.usd").write_text("#usda 1.0\n")
                validate_layout_assets({"Room": {"default": "Simple_Room_purple"}})

    def test_yaml_manifest_expands_defaults_and_skips_disabled(self):
        yaml_text = """
defaults:
  material: bambu_tpu_90a
  success_only: true
  attempts_per_seed: 2

tasks:
  - task: general_pickup
    enabled: true
    seeds: [0, 22]
    dataset_id: ds_pickup
    max_saved_episodes: 5
  - task: fold_clothes
    enabled: false
    seeds: [0]
"""
        parsed = parse_collection_yaml(yaml_text)
        self.assertEqual(parsed["defaults"]["material"], "bambu_tpu_90a")
        self.assertEqual(len(parsed["tasks"]), 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.yaml"
            path.write_text(yaml_text)
            manifest = load_collection_manifest(path)
        self.assertEqual(len(manifest["entries"]), 4)
        self.assertEqual(manifest["entries"][0].seed, 0)
        self.assertEqual(manifest["entries"][1].attempt, 1)
        self.assertEqual(manifest["entries"][2].seed, 22)

    def test_dry_run_flags_unsupported_and_missing_inputs_explicitly(self):
        yaml_text = """
defaults:
  material: bambu_tpu_90a
  success_only: true
  attempts_per_seed: 1

tasks:
  - task: general_pickup
    enabled: true
    seeds: [22]
    max_saved_episodes: 100
  - task: nonexistent_task
    enabled: true
    seeds: [0]
    max_saved_episodes: 100
  - task: general_pickup
    enabled: true
    seeds: [99999]
    max_saved_episodes: 100
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.yaml"
            path.write_text(yaml_text)
            try:
                summary = run_batch(path, batch_id="unit_test_batch", dry_run=True, resume=False)
            finally:
                shutil.rmtree(ROOT / "artifacts/batches/unit_test_batch", ignore_errors=True)
        statuses = [run["status"] for run in summary["runs"]]
        self.assertEqual(statuses, ["ready", "unsupported", "input_missing"])


if __name__ == "__main__":
    unittest.main()
