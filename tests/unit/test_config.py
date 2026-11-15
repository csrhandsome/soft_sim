"""Configuration rules: one file, typed defaults, explicit CLI overrides."""

import json
from pathlib import Path
import tempfile
import unittest

from soft_sim.config import CONFIG_VERSION, MATERIAL_PRESETS, ExperimentConfig, MaterialConfig, ROOT


class ConfigurationRules(unittest.TestCase):
    def write(self, directory, payload):
        path = Path(directory) / "config.json"
        path.write_text(json.dumps(payload))
        return path

    def test_defaults_round_trip_and_snapshot_version(self):
        config = ExperimentConfig().validate()
        self.assertEqual(config.task.name, "general_pickup")
        self.assertEqual(config.task.finger_scale, .75)
        self.assertEqual(config.recording.fps, 20)
        self.assertEqual(config.simulation.solver_iterations,80)
        self.assertEqual(config.simulation.deformable_vertex_velocity_damping,20.)
        self.assertTrue(config.mesh.eliminate_fixed_cells)
        self.assertIsInstance(CONFIG_VERSION, int)
        self.assertTrue((ROOT / "assets/meshes/soft_finger.stl").is_file())
        self.assertTrue(config.to_dict()["planning"]["contact_height_candidates_m"])

    def test_single_task_file_overrides_only_named_fields(self):
        base = ExperimentConfig().to_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, {"task": {"name": "stack_blocks", "seed": 3}})
            config = ExperimentConfig.load(path)
        self.assertEqual(config.task.name, "stack_blocks")
        self.assertEqual(config.task.seed, 3)
        self.assertEqual(config.task.finger_scale, base["task"]["finger_scale"])
        self.assertEqual(config.planning.friction, base["planning"]["friction"])

    def test_material_presets_have_matching_sources_and_valid_parameters(self):
        for preset in MATERIAL_PRESETS:
            with self.subTest(preset=preset):
                config = ExperimentConfig.load(material_preset=preset)
                self.assertEqual(config.material.shore_a, float(preset.removeprefix("bambu_tpu_")[:-1]))
                self.assertTrue(config.material.source.startswith("https://"))
                for suffix in ("json", "md"):
                    self.assertTrue((ROOT / "assets/materials" / f"{preset}.{suffix}").is_file())
                self.assertTrue(all(0 <= c <= 1 for c in config.material.color_linear))
        with self.assertRaises(ValueError):
            MaterialConfig.load("unknown")

    def test_material_file_selection_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, {"material": {"preset": "bambu_tpu_85a", "friction": .4}})
            config = ExperimentConfig.load(path)
            self.assertEqual(config.material.young_modulus_pa, 6.8e6)
            self.assertEqual(config.material.friction, .4)
            selected = ExperimentConfig.load(path, material_preset="bambu_tpu_90a")
            self.assertEqual(selected.material.young_modulus_pa, 5.3e6)
            self.assertEqual(selected.material.friction, .7)
            explicit = ExperimentConfig.load(path, {"material.friction": .6},
                                              material_preset="bambu_tpu_90a")
            self.assertEqual(explicit.material.friction, .6)

    def test_unknown_section_and_field_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ExperimentConfig.load(self.write(directory, {"unknown": {}}))
            with self.assertRaises(ValueError):
                ExperimentConfig.load(self.write(directory, {"task": {"nope": 1}}))

    def test_cli_overrides_win_over_the_file_and_none_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, {"task": {"seed": 3}})
            config = ExperimentConfig.load(path, {"task.seed": 7, "task.name": "stack_blocks",
                                                  "planning.indent_m": None})
        self.assertEqual(config.task.seed, 7)
        self.assertEqual(config.task.name, "stack_blocks")
        self.assertEqual(config.planning.indent_m, ExperimentConfig().planning.indent_m)

    def test_invalid_values_fail_before_starting_the_simulator(self):
        config = ExperimentConfig()
        config.mesh.eliminate_fixed_cells = 'false'
        with self.assertRaisesRegex(ValueError,'boolean'):
            config.validate()
        config = ExperimentConfig()
        config.task.finger_scale = 2.
        with self.assertRaises(ValueError):
            config.validate()
        config = ExperimentConfig()
        config.recording.fps = 7          # does not divide 1/0.001 s
        with self.assertRaises(ValueError):
            config.validate()
        config = ExperimentConfig()
        config.recording.dataset_id = "a/b"
        with self.assertRaises(ValueError):
            config.validate()

    def test_collect_defaults_and_explicit_full_mesh_override(self):
        from soft_sim.cli.collect import build_parser
        from soft_sim.tasks.collect import build_config
        with tempfile.TemporaryDirectory() as directory:
            mesh = Path(directory)/'full.npz'
            mesh.touch()  # Config resolution checks existence; simulation loads it later.
            args = build_parser().parse_args(['--fem-mesh',str(mesh)])
            config = build_config(args)
            self.assertEqual(config.simulation.solver_iterations,80)
            self.assertTrue(config.mesh.eliminate_fixed_cells)
            args = build_parser().parse_args(['--fem-mesh',str(mesh),'--no-fixed-cell-elimination'])
            self.assertFalse(build_config(args).mesh.eliminate_fixed_cells)

    def test_native_x5_profile_is_not_the_legacy_panda_profile(self):
        config = ExperimentConfig()
        config.gripper.profile = "native_x5"
        config.gripper.open_jaw_width_m = .088
        config.validate()
        self.assertNotEqual(config.gripper.profile, "panda")


if __name__ == "__main__":
    unittest.main()
