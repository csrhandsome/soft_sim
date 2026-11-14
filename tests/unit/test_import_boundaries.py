"""Importing the package must stay cheap and side-effect free."""

import json
import subprocess
import sys
import unittest

from soft_sim.config import ROOT

HEAVY = ("torch", "mplib", "sapien", "isaacsim", "isaaclab", "lerobot", "pxr", "omni")

PROBE = f"""
import json, sys, os
import soft_sim
import soft_sim.cli.collect, soft_sim.cli.plan, soft_sim.cli.finalize
import soft_sim.validation.finger_motion, soft_sim.validation.task
import soft_sim.storage, soft_sim.runtime, soft_sim.contracts, soft_sim.migration
print(json.dumps({{name: name in sys.modules for name in {HEAVY!r}}}))
"""


class ImportBoundaries(unittest.TestCase):
    def test_light_modules_do_not_load_simulation_or_dataset_stacks(self):
        result = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual([name for name, present in loaded.items() if present], [])

    def test_import_does_not_create_directories(self):
        before = sorted(path.name for path in ROOT.iterdir())
        result = subprocess.run([sys.executable, "-c", "import soft_sim"], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(path.name for path in ROOT.iterdir()), before)

    def test_command_table_covers_every_cli_module(self):
        from soft_sim.commands import COMMANDS
        for name, spec in COMMANDS.items():
            module = ROOT / "src" / (spec["module"].replace(".", "/") + ".py")
            self.assertTrue(module.is_file(), f"{name} -> missing {module}")
            self.assertIn(spec["env"], {"core", "planning", "robodojo", "dataset", "meshing"})

    def test_entry_point_lists_commands_without_an_interpreter(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run.py"), "--help"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("collect", result.stdout)
        self.assertIn("validate", result.stdout)


if __name__ == "__main__":
    unittest.main()
