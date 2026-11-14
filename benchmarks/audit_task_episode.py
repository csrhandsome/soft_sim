"""Finalize and audit a task capture, keeping task outcome and quality separate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def acceptance(result, validation, clamp, collisions, diagnostics):
    task_success = result.get("success") is True
    checks = dict(sensor_quality=validation.get("valid") is True,
        native_drives=result.get("joint_control") == "position",
        original_task_manager=result.get("official_task_manager") is True,
        clamp=clamp.get("clamp_valid") is True,
        positive_elements=clamp.get("positive_element_volumes") is True,
        sampled_task_collisions=collisions.get("valid") is True,
        rest_mesh_positive=diagnostics.get("rest_mesh_positive") is True,
        mass_diagnostics_explained=diagnostics.get("small_positive_threshold_explains_mass_diagnostics") is True,
        cooking_setup=diagnostics.get("cooking_setup_error_count") == 0)
    quality_valid = all(checks.values())
    return dict(accepted=task_success and quality_valid, task_success=task_success,
                quality_valid=quality_valid, checks=checks,
                failed_checks=[name for name, valid in checks.items() if not valid])


def audit(run, log_path, workers=2):
    run = run.resolve()
    recording, output = run / "recording", run / "audit"
    output.mkdir(exist_ok=True)
    dataset_python = ROOT / "envs/dataset/.venv/bin/python"
    planning_python = ROOT / "envs/planning/.venv/bin/python"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLCONFIGDIR="/tmp/soft-sim-mpl",
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def execute(name, interpreter, script, *arguments):
        command = [str(interpreter), str(ROOT / "benchmarks" / script), *map(str, arguments)]
        with (output / f"{name}.log").open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        print("AUDIT", name, process.returncode, flush=True)
        return dict(name=name, returncode=process.returncode, command=command)

    episode = json.loads((recording / "episode.json").read_text())
    commands = []
    if episode["status"] == "staged":
        commands.append(execute("finalize", dataset_python, "finalize_lerobot.py", recording))
    episode = json.loads((recording / "episode.json").read_text())
    if episode["status"] != "complete":
        raise ValueError("Capture is incomplete; refusing task acceptance")
    jobs = [("camera_projection", dataset_python, "check_finger_camera.py", recording),
            ("finger_motion", dataset_python, "check_finger_motion.py", recording),
            ("fem_projection", dataset_python, "check_fem_camera_projection.py", recording),
            ("rgb_stability", dataset_python, "check_camera_stability.py", recording),
            ("clamp", dataset_python, "check_clamp_deformation.py", run, "--output", output / "clamp"),
            ("physx", planning_python, "check_physx_mesh_diagnostics.py", run, "--log", log_path)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        commands.extend(pool.map(lambda job: execute(*job), jobs))
    states = output / "collision_states"
    export = execute("collision_export", dataset_python, "check_task_collisions.py",
                     run, "--states", states, "--export-states")
    commands.append(export)
    if export["returncode"] == 0:
        commands.append(execute("collisions", planning_python, "check_task_collisions.py", run, "--states", states))
    commands.append(execute("data_integrity", dataset_python, "validate_rgb_episode.py", recording))

    def read(path):
        return json.loads(path.read_text()) if path.is_file() else {}

    report = acceptance(read(run / "result.json"), read(recording / "validation.json"),
        read(output / "clamp/clamp_deformation.json"), read(recording / "task_collision_validation.json"),
        read(run / "physx_mesh_diagnostics.json"))
    report.update(commands=commands, run=str(run), frame_count=episode["frame_count"],
        layout_sha256=episode["calibration"]["task_scene"]["layout_sha256"],
        scope="One recorded layout/run. Sampled surfaces and initialization diagnostics do not establish "
              "continuous collision freedom, material calibration, or success on other tasks/layouts.")
    if any(command["returncode"] != 0 for command in commands):
        report["accepted"] = report["quality_valid"] = False
    (run / "acceptance.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()
    report = audit(args.run, args.log.resolve(), args.workers)
    print(json.dumps({key: value for key, value in report.items() if key != "commands"}, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)
