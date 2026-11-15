"""Full CPU/GPU follow A/B with 80 iterations and the same saved grasp plan.

SOFT_SIM_FOLLOW_SOURCE_RUN opts into three complete captures. Numerical tests run
without it. SOFT_SIM_FOLLOW_REPORT revalidates retained evidence without reruns.
SOFT_SIM_FOLLOW_RESUME continues missing variants from a frozen report.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

from soft_sim.config import ROOT
from soft_sim.runtime import run
from soft_sim.storage import Storage
from tests.integration.gpu_performance.test_replay_speed import normalized_configuration


def maximum_state_difference(key, first, second):
    difference = first - second
    if key in ("finger_nodes_left", "finger_nodes_right", "object_position"):
        return float(np.linalg.norm(difference, axis=-1).max())
    if key == "object_orientation_wxyz":
        # Quaternion sign/coordinates are not a rotation error. Report the
        # shortest angular distance in radians, in float64 to resolve small angles.
        a, b = first.astype(float), second.astype(float)
        a /= np.linalg.norm(a, axis=-1, keepdims=True)
        b /= np.linalg.norm(b, axis=-1, keepdims=True)
        return float((2*np.arccos(np.clip(np.abs(np.sum(a*b, axis=-1)), 0., 1.))).max())
    return float(np.abs(difference).max())


class FollowReplay(unittest.TestCase):
    def test_same_grasp_is_faster_with_valid_physics_and_bounded_state_differences(self):
        existing = os.environ.get("SOFT_SIM_FOLLOW_REPORT")
        if existing:
            path = Path(existing).resolve()
            return self.validate_report(json.loads(path.read_text()), path)
        resume = os.environ.get("SOFT_SIM_FOLLOW_RESUME")
        if resume:
            path = Path(resume).resolve()
            report = json.loads(path.read_text())
            self.run_variants(report, path)
            return self.validate_report(report, path)
        source = os.environ.get("SOFT_SIM_FOLLOW_SOURCE_RUN")
        if not source:
            self.skipTest("Set SOFT_SIM_FOLLOW_SOURCE_RUN for full 80-iteration grasp replays")
        storage = Storage.create()
        original = storage.run_dir(source)
        manifest = json.loads((original / "manifest.json").read_text())
        self.assertIs(manifest["task_success"], True)
        cfg = json.loads((original / "config.resolved.json").read_text())
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        self.assertEqual(cfg["simulation"]["solver_iterations"], 80)
        self.assertEqual(cfg["simulation"]["timestep_s"], .001)
        self.assertEqual(cfg["task"]["name"], "general_pickup")
        executed = json.loads((original / "plans/executed.json").read_text())[0]
        plan = (original / executed["plan_file"]).resolve()
        self.assertEqual(hashlib.sha256(plan.read_bytes()).hexdigest(), executed["sha256"])
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = storage.cache_dir("benchmarks") / f"follow_replay_{stamp}"
        shared = destination / "shared/src"
        shutil.copytree(ROOT / "src/soft_sim", shared / "soft_sim",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        cfg["recording"]["dataset_id"] = f"follow_replay_{stamp}"
        config_path = destination / "config.json"
        config_path.write_text(json.dumps(cfg, indent=2))
        report = dict(source_run=source, plan_sha256=executed["sha256"],
                      gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                                   "--format=csv,noheader"], text=True).strip(),
                      source_sha256={str(p.relative_to(shared)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(shared.rglob("*.py"))}, runs={})
        path = destination / "comparison.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"FOLLOW_REPLAY_REPORT {path}", flush=True)
        self.run_variants(report, path)
        self.validate_report(report, path)

    def run_variants(self, report, path):
        storage = Storage.create()
        original = storage.run_dir(report["source_run"])
        executed = json.loads((original / "plans/executed.json").read_text())[0]
        plan = (original / executed["plan_file"]).resolve()
        self.assertEqual(hashlib.sha256(plan.read_bytes()).hexdigest(), report["plan_sha256"])
        destination = path.parent
        shared = destination / "shared/src"
        hashes = {str(p.relative_to(shared)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(shared.rglob("*.py"))}
        self.assertEqual(hashes, report["source_sha256"], "Frozen source changed")
        config_path = destination / "config.json"
        stamp = destination.name.removeprefix("follow_replay_")
        # GPU first so an integration failure is caught before spending a full
        # baseline episode. Repeat the identical CPU path to measure solver
        # variability instead of treating bitwise FEM determinism as a given.
        for variant in ("gpu", "cpu", "cpu_repeat"):
            if variant in report["runs"] and "timing" in report["runs"][variant]:
                continue
            backend = "gpu" if variant == "gpu" else "cpu"
            run_id = f"{stamp}_follow_{variant}"
            states = destination / f"{variant}_states.npz"
            log = destination / f"{variant}.log"
            print(f"FOLLOW_REPLAY_START {variant} {log}", flush=True)
            process = run("robodojo", "tests.integration.gpu_follow.replay_worker",
                          "--states", states, "--mode", "replay", "--config", config_path,
                          "--grasp-plan", plan, "--run-id", run_id, log_path=log,
                          extra_env={"SOFT_SIM_ROOT": str(ROOT),
                                     "SOFT_SIM_FOLLOW_BACKEND": backend,
                                     "PYTHONPATH": os.pathsep.join((str(shared), str(ROOT)))})
            entry = dict(run_id=run_id, log=str(log), states=str(states), returncode=process.returncode)
            report["runs"][variant] = entry
            path.write_text(json.dumps(report, indent=2))
            self.assertTrue(process.ok, f"{backend} replay failed: {log}")
            folder = storage.run_dir(run_id)
            entry.update(timing=json.loads((folder / "diagnostics/collection_timing.json").read_text()),
                         result=json.loads((folder / "results/task.json").read_text()),
                         capture=json.loads((folder / "capture/episode.json").read_text()),
                         configuration=normalized_configuration(json.loads((folder / "config.resolved.json").read_text())))
            path.write_text(json.dumps(report, indent=2))
            print(f"FOLLOW_REPLAY_DONE {variant} {entry['timing']['wall_s']:.3f}s", flush=True)

    def validate_report(self, report, path):
        cpu, gpu, repeat = [report["runs"][key] for key in ("cpu", "gpu", "cpu_repeat")]
        self.assertEqual(cpu["configuration"]["simulation"]["solver_iterations"], 80)
        self.assertEqual(cpu["configuration"]["simulation"]["timestep_s"], .001)
        for backend, entry in (("cpu", cpu), ("gpu", gpu), ("cpu", repeat)):
            self.assertEqual(cpu["configuration"], entry["configuration"])
            for key in ("physics_steps", "simulation_s", "captured_frames", "timestep_s", "fps", "rendering"):
                self.assertEqual(cpu["timing"][key], entry["timing"][key], key)
            self.assertEqual(cpu["capture"]["frames"], entry["capture"]["frames"])
            self.assertEqual([(p["phase"], p["physics_steps"]) for p in cpu["timing"]["phases"]],
                             [(p["phase"], p["physics_steps"]) for p in entry["timing"]["phases"]])
            self.assertEqual(entry["returncode"], 0)
            self.assertIs(entry["result"]["success"], True)
            self.assertEqual(entry["result"]["executed_plans"][0]["sha256"], report["plan_sha256"])
            self.assertEqual(entry["capture"]["status"], "committed")
            self.assertEqual(entry["capture"]["camera_order"], ["left_rgb", "right_rgb", "external_rgb"])
            self.assertEqual(entry["timing"]["follow_backend"], backend)
        differences, repeat_differences, tolerances = {}, {}, {}
        floors = {"finger_nodes_left": 1e-4, "finger_nodes_right": 1e-4,
                      "object_position": 1e-4, "object_orientation_wxyz": np.deg2rad(.1),
                      "finger_mounts": 1e-6, "joint_positions": 1e-5,
                      "all_finger_minimum_jacobians": 1e-2,
                      "all_finger_maximum_pin_errors_m": 1e-6}
        # Controller inputs/mounts remain strict. Only solved states get an
        # allowance based on a separately measured identical CPU repetition.
        solved = {"finger_nodes_left", "finger_nodes_right", "object_position",
                  "object_orientation_wxyz", "all_finger_minimum_jacobians",
                  "all_finger_maximum_pin_errors_m"}
        # Never let a noisy baseline silently make the comparison unbounded.
        # Half the actual scaled mesh pitch is an independent hard ceiling.
        node_cap = .5 * cpu["result"]["config"]["mesh"]["voxel_pitch_m"]
        caps = {"finger_nodes_left": node_cap, "finger_nodes_right": node_cap,
                "object_position": .001, "object_orientation_wxyz": np.deg2rad(2.),
                "all_finger_minimum_jacobians": .03,
                "all_finger_maximum_pin_errors_m": 1e-5}
        with np.load(cpu["states"]) as old, np.load(gpu["states"]) as new, np.load(repeat["states"]) as again:
            for key, floor in floors.items():
                self.assertEqual(old[key].shape, new[key].shape, key)
                self.assertEqual(old[key].shape, again[key].shape, key)
                self.assertTrue(all(np.isfinite(values[key]).all() for values in (old, new, again)), key)
                differences[key] = maximum_state_difference(key, old[key], new[key])
                repeat_differences[key] = maximum_state_difference(key, old[key], again[key])
                tolerances[key] = min(caps[key], max(floor, 2*repeat_differences[key])) if key in solved else floor
            command_differences = {}
            for name, values in (("gpu", new), ("cpu_repeat", again)):
                command_differences[name] = float(np.abs(old["commanded_joint_positions"]
                                                       - values["commanded_joint_positions"]).max())
                # Parked-arm hold targets are measured after settling; identical
                # CPU repeats can differ by ~1e-18 rad around numerical zero.
                np.testing.assert_allclose(old["commanded_joint_positions"],
                                           values["commanded_joint_positions"], atol=1e-12, rtol=0)
            for values in (old, new, again):
                self.assertGreater(float(values["all_finger_minimum_jacobians"].min()), 0.)
                self.assertLess(float(values["all_finger_maximum_pin_errors_m"].max()), 1e-5)
        step_times = {key: 1000*sum(p["control_and_physics_s"] for p in entry["timing"]["phases"])
                     /entry["timing"]["physics_steps"] for key, entry in report["runs"].items()}
        before = (step_times["cpu"] + step_times["cpu_repeat"]) / 2
        after = step_times["gpu"]
        lift_difference = abs(cpu["result"]["object_lift_m"] - gpu["result"]["object_lift_m"])
        lift_repeat_difference = abs(cpu["result"]["object_lift_m"] - repeat["result"]["object_lift_m"])
        lift_tolerance = min(.0005, max(1e-4, 2*lift_repeat_difference))
        report.update(maximum_state_differences=differences, numerical_tolerances=tolerances,
                      maximum_command_differences_rad=command_differences,
                      command_absolute_tolerance_rad=1e-12,
                      cpu_repeat_maximum_state_differences=repeat_differences,
                      comparison_rule="Solved states: min(cap, max(absolute floor, 2 * CPU repeat max difference)); mounts/commands stay strict",
                      difference_measure="Node/object positions: maximum Euclidean distance (m); object orientation: shortest quaternion angular distance (rad); other arrays: maximum absolute element difference",
                      numerical_caps=caps,
                      object_lift_difference_m=lift_difference,
                      cpu_repeat_object_lift_difference_m=lift_repeat_difference,
                      object_lift_tolerance_m=lift_tolerance,
                      control_and_physics_ms_per_step=step_times,
                      control_and_physics_speedup=before/after,
                      collection_speedup=(cpu["timing"]["wall_s"]+repeat["timing"]["wall_s"])
                                         /(2*gpu["timing"]["wall_s"]))
        path.write_text(json.dumps(report, indent=2))
        print("FOLLOW_REPLAY_RESULT " + json.dumps({k: v for k, v in report.items()
              if k not in ("runs", "source_sha256")}), flush=True)
        for key, difference in differences.items():
            self.assertLessEqual(difference, tolerances[key], key)
            if key in caps:
                self.assertLessEqual(repeat_differences[key], caps[key], f"CPU baseline too variable: {key}")
        self.assertLessEqual(lift_difference, lift_tolerance)
        self.assertLess(step_times["gpu"], min(step_times["cpu"], step_times["cpu_repeat"]))
        self.assertLess(gpu["timing"]["wall_s"], min(cpu["timing"]["wall_s"], repeat["timing"]["wall_s"]))
