"""Replay the same complete episode with baseline and optimized collection code.

SOFT_SIM_PERFORMANCE_SOURCE_RUN names an existing successful general_pickup run.
SOFT_SIM_PERFORMANCE_BASELINE_REV selects the commit before this optimization.
Only the three optimized modules come from that revision; current camera/task
adapters and third-party sources are shared so unrelated fixes stay in effect.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest

from soft_sim.config import ROOT
from soft_sim.runtime import run
from soft_sim.storage import Storage


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Baseline does not match the pre-optimization timing anchor: " + before)
    return source.replace(before, after, 1)


def baseline_with_timing(source):
    """Add timers only, leaving the original collection work schedule intact."""
    source = replace_once(source,
        "        stride = round(1 / (cfg.recording.fps * cfg.simulation.timestep_s))\n",
        "        stride = round(1 / (cfg.recording.fps * cfg.simulation.timestep_s))\n"
        "        collection_started = time.perf_counter()\n"
        "        phase_timings = []\n")
    source = replace_once(source,
        '                print("TASK_PHASE", operation_index, target_label, phase, flush=True)\n',
        '                print("TASK_PHASE", operation_index, target_label, phase, flush=True)\n'
        "                phase_started = time.perf_counter()\n"
        "                phase_capture_s = 0.\n")
    source = replace_once(source,
        "                    if sim_step % stride == 0:\n",
        "                    if sim_step % stride == 0:\n"
        "                        capture_started = time.perf_counter()\n")
    source = replace_once(source,
        "                        env.reward_manager.step(env_idx_list=[0])\n",
        "                        env.reward_manager.step(env_idx_list=[0])\n"
        "                        phase_capture_s += time.perf_counter() - capture_started\n")
    source = replace_once(source,
        "                lift = float(object_pose()[0][2]) - operation_initial_z\n",
        "                phase_timings.append(dict(operation_index=operation_index, phase=phase,\n"
        "                    physics_steps=steps, simulation_s=steps * cfg.simulation.timestep_s,\n"
        "                    wall_s=time.perf_counter() - phase_started,\n"
        "                    capture_and_record_s=phase_capture_s))\n"
        "                lift = float(object_pose()[0][2]) - operation_initial_z\n")
    return replace_once(source,
        "        rewards = env.reward_manager.get_reward()\n",
        '        store.write_json("diagnostics/collection_timing.json", dict(\n'
        '            scope="Baseline collection; excludes setup and dataset finalization",\n'
        "            wall_s=time.perf_counter() - collection_started, physics_steps=sim_step,\n"
        "            simulation_s=sim_step * cfg.simulation.timestep_s,\n"
        "            captured_frames=len(recorder.rows), timestep_s=cfg.simulation.timestep_s,\n"
        "            fps=cfg.recording.fps, rendering=env.tool.render_settings, phases=phase_timings))\n"
        "        rewards = env.reward_manager.get_reward()\n")


def normalized_configuration(resolved):
    resolved = json.loads(json.dumps(resolved))
    resolved.pop("_cli_overrides", None)
    resolved["mesh"].pop("tetra_mesh", None)
    # Older snapshots predate the explicit mechanism selector. They only
    # supported X5, so omission and the new default "x5" are equivalent.
    resolved["task"].setdefault("robot", "x5")
    return resolved


class ReplaySpeed(unittest.TestCase):
    def test_complete_episode_is_faster_with_matching_workload(self):
        existing = os.environ.get("SOFT_SIM_PERFORMANCE_REPORT")
        if existing:
            report_path = Path(existing).resolve()
            report = json.loads(report_path.read_text())
            for entry in report["runs"].values():
                folder = Storage.create().run_dir(entry["run_id"])
                entry["configuration"] = normalized_configuration(
                    json.loads((folder / "config.resolved.json").read_text()))
                entry["timing"] = json.loads((folder / "diagnostics/collection_timing.json").read_text())
                entry["result"] = json.loads((folder / "results/task.json").read_text())
                entry["capture"] = json.loads((folder / "capture/episode.json").read_text())
            print(f"GPU_PERFORMANCE_RECHECK {report_path} (existing completed replays)", flush=True)
            self.validate_report(report, report_path)
            return
        source_run = os.environ.get("SOFT_SIM_PERFORMANCE_SOURCE_RUN")
        self.assertTrue(source_run,
            "Set SOFT_SIM_PERFORMANCE_SOURCE_RUN to a successful general_pickup run id")
        storage = Storage.create()
        original = storage.run_dir(source_run)
        manifest = json.loads((original / "manifest.json").read_text())
        self.assertIs(manifest["task_success"], True)
        config = json.loads((original / "config.resolved.json").read_text())
        config = {key: value for key, value in config.items() if not key.startswith("_")}
        self.assertEqual(config["task"]["name"], "general_pickup")
        executed = json.loads((original / "plans/executed.json").read_text())
        plan = (original / executed[0]["plan_file"]).resolve()
        self.assertEqual(hashlib.sha256(plan.read_bytes()).hexdigest(), executed[0]["sha256"])
        reference = os.environ.get("SOFT_SIM_PERFORMANCE_BASELINE_REV",
                                   "1a51a2b3a67e47d8e9467e2e71a50480edb873ac")
        revision = subprocess.check_output(["git", "rev-parse", reference], cwd=ROOT, text=True).strip()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = storage.cache_dir("benchmarks") / f"gpu_{stamp}"
        baseline = destination / "baseline"
        # Freeze common code before either process starts. Concurrent work in
        # the shared repository must not change the second half of an A/B run.
        shared = destination / "shared" / "src"
        shutil.copytree(ROOT / "src/soft_sim", shared / "soft_sim",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shared_hashes = {str(path.relative_to(shared)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(shared.rglob("*.py"))}
        hashes = {}
        for relative in ("execution.py", "sim/soft_tool.py", "tasks/collect.py"):
            source = subprocess.check_output(
                ["git", "show", f"{revision}:src/soft_sim/{relative}"], cwd=ROOT, text=True)
            hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
            if relative == "tasks/collect.py":
                source = baseline_with_timing(source)
            path = baseline / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        config["recording"]["dataset_id"] = f"gpu_performance_{stamp}"
        configuration = destination / "config.json"
        configuration.write_text(json.dumps(config, indent=2) + "\n")
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                      "--format=csv,noheader"], text=True).strip()
        report = dict(source_run=source_run, source_plan=str(plan), plan_sha256=executed[0]["sha256"],
                      baseline_revision=revision, baseline_source_sha256=hashes, gpu=gpu,
                      shared_source_sha256=shared_hashes,
                      scope="Complete replay collection, excluding scene setup and dataset finalization",
                      runs={})
        report_path = destination / "comparison.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"GPU_PERFORMANCE_REPORT {report_path}", flush=True)
        for variant in ("baseline", "optimized"):
            run_id = f"{stamp}_gpu-performance_{variant}"
            arguments = (["--baseline", baseline] if variant == "baseline" else [])
            arguments += ["--mode", "replay", "--config", configuration,
                          "--grasp-plan", plan, "--run-id", run_id]
            log_path = destination / f"{variant}.log"
            print(f"GPU_PERFORMANCE_START {variant} run={run_id} log={log_path}", flush=True)
            process = run("robodojo", "tests.integration.gpu_performance.replay_worker",
                          *arguments, log_path=log_path, extra_env={
                              "SOFT_SIM_ROOT": str(ROOT),
                              "PYTHONPATH": os.pathsep.join((str(shared), str(ROOT))),
                          })
            entry = dict(run_id=run_id, log=str(log_path), returncode=process.returncode)
            report["runs"][variant] = entry
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            self.assertTrue(process.ok, f"{variant} replay failed: {log_path}")
            folder = storage.run_dir(run_id)
            entry["timing"] = json.loads((folder / "diagnostics/collection_timing.json").read_text())
            entry["result"] = json.loads((folder / "results/task.json").read_text())
            entry["capture"] = json.loads((folder / "capture/episode.json").read_text())
            # Compare configurations after removing the per-run mesh cache path.
            entry["configuration"] = normalized_configuration(
                json.loads((folder / "config.resolved.json").read_text()))
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"GPU_PERFORMANCE_DONE {variant} wall_s={entry['timing']['wall_s']:.3f}", flush=True)
        self.validate_report(report, report_path)

    def validate_report(self, report, report_path):
        old, new = [report["runs"][key] for key in ("baseline", "optimized")]
        self.assertEqual(old["configuration"], new["configuration"])
        for key in ("physics_steps", "simulation_s", "captured_frames", "timestep_s", "fps", "rendering"):
            self.assertEqual(old["timing"][key], new["timing"][key], key)
        self.assertEqual([(p["phase"], p["physics_steps"]) for p in old["timing"]["phases"]],
                         [(p["phase"], p["physics_steps"]) for p in new["timing"]["phases"]])
        for entry in (old, new):
            self.assertEqual(entry["returncode"], 0)
            self.assertIs(entry["result"]["success"], True)
            self.assertEqual(entry["result"]["executed_plans"][0]["sha256"], report["plan_sha256"])
            self.assertEqual(entry["capture"]["status"], "committed")
            self.assertEqual(entry["capture"]["camera_order"], ["left_rgb", "right_rgb", "external_rgb"])
            self.assertEqual(entry["capture"]["calibration"]["task_scene"]["robot_name"], "x5")
        report["speedup"] = old["timing"]["wall_s"] / new["timing"]["wall_s"]
        report["reduction_percent"] = (1 - 1 / report["speedup"]) * 100
        report["configuration_normalization"] = "Per-run mesh path excluded; missing task.robot means legacy X5"
        report["validation_passed"] = report["speedup"] > 1.
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"GPU_PERFORMANCE_RESULT speedup={report['speedup']:.4f} "
              f"reduction_percent={report['reduction_percent']:.2f}", flush=True)
        self.assertGreater(report["speedup"], 1., f"No full-episode speedup measured: {report_path}")


if __name__ == "__main__":
    unittest.main()
