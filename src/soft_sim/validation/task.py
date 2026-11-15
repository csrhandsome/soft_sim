"""Organize the independent quality checks for one run and record their result.

Task outcome and quality stay separate: a capture is only ``accepted`` when the
upstream RewardManager reported success *and* every required check passed. The
report is written under ``reports/<validation_id>/`` so re-running a check never
overwrites earlier evidence.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

from soft_sim.recording.reader import capture_directory
from soft_sim.runtime import run as run_process
from soft_sim.storage import RunStore, Storage

#: (name, environment, module, extra arguments) - arguments get the paths appended.
CHECKS = (
    ("camera_projection", "dataset", "soft_sim.validation.finger_camera", ()),
    ("finger_motion", "dataset", "soft_sim.validation.finger_motion", ()),
    ("fem_projection", "dataset", "soft_sim.validation.fem_projection", ()),
    ("rgb_stability", "dataset", "soft_sim.validation.camera_stability", ()),
    ("data_integrity", "dataset", "soft_sim.validation.episode", ()),
    ("clamp", "dataset", "soft_sim.validation.clamp", ("--output", "{output}/clamp")),
    ("physx", "planning", "soft_sim.validation.physx", ("--output", "{output}/physx.json")),
)


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


def _read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def _task_result(run_dir):
    for candidate in (run_dir / "results/task.json", run_dir / "result.json"):
        if candidate.is_file():
            return _read(candidate)
    return {}


def audit_run(run_path, *, storage_root=None, workers=2, log_path=None, validation_id=None,
              log=None):
    """Run every check for one run and write one acceptance report."""
    run_dir = Path(run_path).resolve()
    storage = Storage.create(storage_root)
    store = None
    if (storage.run_dir(run_dir.name) / "manifest.json").is_file():
        store = RunStore.open(storage, run_dir.name)
        run_dir = store.root
    validation_id = validation_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = run_dir / "reports" / validation_id
    output.mkdir(parents=True, exist_ok=True)
    capture = capture_directory(run_dir)
    log_path = Path(log_path or run_dir / "logs/simulation.log")
    log = log or (output / "validation.log")

    def execute(name, environment, module, arguments):
        process = run_process(environment, module, *arguments,
                              log_path=output / f"{name}.log")
        print("VALIDATION", name, process.returncode, flush=True)
        return process

    commands = []
    metadata = _read(capture / "episode.json")
    if metadata.get("status") == "staged":
        process = execute("finalize", "dataset", "soft_sim.cli.finalize", [capture])
        commands.append(process.to_dict())
        if not process.ok:
            raise RuntimeError(f"Finalize failed; see {process.log_path}")
        metadata = _read(capture / "episode.json")
    if metadata.get("status") != "committed":
        raise ValueError("Capture is not committed; refusing task acceptance")

    jobs = {}
    for name, environment, module, extra in CHECKS:
        if name == "data_integrity":
            continue  # Its summary depends on all independent audit files.
        base = capture if name not in ("clamp", "physx") else run_dir
        arguments = [base]
        if name == "physx":
            arguments.extend(["--log", log_path])
        arguments.extend(extra)
        jobs[name] = (environment, module,
                      [str(value).format(output=output) for value in arguments])
    jobs["collision_export"] = ("dataset", "soft_sim.validation.collisions",
                                [str(run_dir), "--states", str(output / "collision_states"),
                                 "--export-states"])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(jobs, pool.map(lambda name: execute(name, *jobs[name]), jobs)))
    commands.extend(process.to_dict() for process in results.values())
    if results["collision_export"].ok:
        process = execute("collisions", "planning", "soft_sim.validation.collisions",
                          [run_dir, "--states", output / "collision_states"])
        commands.append(process.to_dict())

    # This aggregates the other audit files, so it must run after their writes.
    process = execute("data_integrity", "dataset", "soft_sim.validation.episode", [capture])
    commands.append(process.to_dict())

    report = acceptance(_task_result(run_dir), _read(capture / "validation.json"),
        _read(output / "clamp/clamp_deformation.json"),
        _read(capture / "task_collision_validation.json"),
        _read(output / "physx.json"))
    report.update(validation_id=validation_id, run_id=run_dir.name, run=str(run_dir),
        capture=str(capture), frame_count=metadata["frame_count"],
        layout_sha256=metadata.get("calibration", {}).get("task_scene", {}).get("layout_sha256"),
        commands=commands,
        scope="One recorded layout/run. Sampled surfaces and initialization diagnostics do not "
              "establish continuous collision freedom, material calibration, or success on other "
              "tasks/layouts.")
    if any(command["returncode"] != 0 for command in commands):
        report["accepted"] = report["quality_valid"] = False
    (output / "acceptance.json").write_text(json.dumps(report, indent=2))
    if store is not None:
        store.update(validation="passed" if report["quality_valid"] else "failed",
                     acceptance=report["accepted"], reports=[storage.relative(output)])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args(argv)
    report = audit_run(args.run, storage_root=args.storage_root, workers=args.workers,
                       log_path=args.log)
    print(json.dumps({key: value for key, value in report.items() if key != "commands"}, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
