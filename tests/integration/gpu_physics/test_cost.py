"""Screen cache updates and scene/soft-body iteration settings separately."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import unittest

import numpy as np

from soft_sim.config import ROOT
from soft_sim.runtime import run
from soft_sim.storage import Storage


class PhysicsCost(unittest.TestCase):
    def test_local_physics_cost_with_independent_fem_and_scene_iterations(self):
        existing = os.environ.get("SOFT_SIM_PHYSICS_REPORT")
        if existing:
            path = Path(existing).resolve()
            self.summarize(json.loads(path.read_text()), path)
            return
        source = os.environ.get("SOFT_SIM_PHYSICS_SOURCE_RUN")
        if not source:
            self.skipTest("Set SOFT_SIM_PHYSICS_SOURCE_RUN to a validated native-task run")
        storage = Storage.create()
        cfg = json.loads((storage.run_dir(source)/"config.resolved.json").read_text())
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = storage.cache_dir("benchmarks") / f"physics_cost_{stamp}"
        shared = destination / "shared/src"
        shutil.copytree(ROOT/"src/soft_sim", shared/"soft_sim",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        configuration = destination/"config.json"
        configuration.write_text(json.dumps(cfg, indent=2))
        candidate = int(os.environ.get("SOFT_SIM_PHYSICS_CANDIDATE_ITERATIONS", "40"))
        self.assertTrue(1 <= candidate < 80, "Compare 80 against one lower iteration count")
        variants = [dict(name="fem80_scene8_single", fem=80, scene=8, single=True),
                    dict(name=f"fem{candidate}_scene8_single", fem=candidate, scene=8, single=True)]
        # The broader diagnostic is opt-in. Normal iteration compares one
        # candidate against one reference, rather than starting a large sweep.
        if os.environ.get("SOFT_SIM_PHYSICS_EXTENDED") == "1":
            variants = [dict(name="fem80_scene80_double", fem=80, scene=80, single=False),
                        dict(name="fem80_scene80_single", fem=80, scene=80, single=True),
                        variants[0],
                        dict(name=f"fem{candidate}_scene80_single", fem=candidate, scene=80, single=True),
                        variants[1]]
        report = dict(source_run=source,
                      scope="Screening: full native scene, held joint targets, no timed rendering; not grasp fidelity validation",
                      source_sha256={str(p.relative_to(shared)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(shared.rglob("*.py"))}, variants=[])
        path = destination/"comparison.json"
        print(f"PHYSICS_COST_REPORT {path}", flush=True)
        for variant in variants:
            name = variant["name"]
            output = destination/f"{name}.json"
            arguments = ["--output", output, "--scene-min-iterations", variant["scene"],
                         "--steps", os.environ.get("SOFT_SIM_PHYSICS_PROBE_STEPS", "300"),
                         "--mode", "export", "--config", configuration,
                         "--solver-iterations", variant["fem"], "--run-id", f"{stamp}_{name}"]
            if variant["single"]:
                arguments.append("--single-update")
            log = destination/f"{name}.log"
            print(f"PHYSICS_COST_START {name} {log}", flush=True)
            process = run("robodojo", "tests.integration.gpu_follow.physics_probe_worker", *arguments,
                          log_path=log, extra_env={"SOFT_SIM_ROOT": str(ROOT),
                          "SOFT_SIM_FOLLOW_BACKEND": "gpu",
                          "PYTHONPATH": os.pathsep.join((str(shared), str(ROOT)))})
            entry = dict(**variant, returncode=process.returncode, log=str(log), output=str(output))
            report["variants"].append(entry)
            path.write_text(json.dumps(report, indent=2))
            self.assertTrue(process.ok, f"Probe failed: {log}")
            entry["report"] = json.loads(output.read_text())
            path.write_text(json.dumps(report, indent=2))
            self.assertEqual(entry["report"]["fem_iterations"], variant["fem"])
            self.assertEqual(entry["report"]["cache_pose_error_m"], 0.)
            for finger in entry["report"]["fingers"]:
                self.assertEqual(finger["authored_iterations"], variant["fem"])
                self.assertGreater(finger["minimum_jacobian"], 0.)
                self.assertLess(finger["maximum_pin_error_m"], 1e-5)
            print(f"PHYSICS_COST_DONE {name} loop_ms={entry['report']['timing_ms']['loop']['median']:.3f}", flush=True)
        self.summarize(report, path)

    def summarize(self, report, path):
        for entry in report["variants"]:
            self.assertEqual(entry["returncode"], 0)
            self.assertEqual(entry["report"]["cache_pose_error_m"], 0.)
            for finger in entry["report"]["fingers"]:
                self.assertGreater(finger["minimum_jacobian"], 0.)
                self.assertLess(finger["maximum_pin_error_m"], 1e-5)
        reference = next(e for e in report["variants"] if e["name"] == "fem80_scene8_single")
        candidates = [e for e in report["variants"] if e["scene"] == 8 and e["fem"] != 80]
        comparisons = []
        for entry in candidates:
            before = reference["report"]["timing_ms"]["loop"]["median"]
            after = entry["report"]["timing_ms"]["loop"]["median"]
            changes, drifts = [], []
            with np.load(Path(reference["output"]).with_suffix(".npz")) as old, \
                 np.load(Path(entry["output"]).with_suffix(".npz")) as new:
                for a, b in zip(reference["report"]["fingers"], entry["report"]["fingers"]):
                    arm, side = a["arm"], a["side"]
                    om, nm = old[f"mounts_{arm}"][side], new[f"mounts_{arm}"][side]
                    first = (old[f"nodes_{arm}_{side}"]-om[:3, 3]) @ om[:3, :3]
                    second = (new[f"nodes_{arm}_{side}"]-nm[:3, 3]) @ nm[:3, :3]
                    changes.append(b["maximum_displacement_m"]/a["maximum_displacement_m"]-1)
                    drifts.append(float(np.linalg.norm(first-second, axis=1).max()))
            # Conservative screening only: failing sends no candidate into an
            # expensive full replay. Passing still requires a short contact test.
            displacement_limit = .05
            comparisons.append(dict(reference=reference["name"], candidate=entry["name"],
                speedup=before/after, time_reduction_percent=100*(1-after/before),
                per_finger_relative_displacement_change=changes, per_finger_maximum_node_drift_m=drifts,
                relative_displacement_screen_limit=displacement_limit,
                passes_static_displacement_screen=bool(max(map(abs, changes)) <= displacement_limit),
                qualifies_for_full_replay=False,
                scope="Single held-pose terminal snapshots; contact/lift fidelity has not been tested"))
        report["comparisons"] = comparisons
        path.write_text(json.dumps(report, indent=2))
        print("PHYSICS_COST_COMPARISONS " + json.dumps(comparisons), flush=True)
