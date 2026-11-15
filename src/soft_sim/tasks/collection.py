"""YAML-driven batch collection orchestration for adapted RoboDojo tasks."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from soft_sim.config import ROOT
from soft_sim.runtime import child_environment, environment
from soft_sim.sim.robodojo import layout_path, validate_layout_assets
from soft_sim.tasks.adapters import IMPLEMENTED_TASKS


def _parse_scalar(text):
    value = text.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_parse_scalar(item) for item in inner.split(",")] if inner else []
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _strip_comment(line):
    in_quote = None
    for index, char in enumerate(line):
        if char in ('"', "'"):
            in_quote = None if in_quote == char else (char if in_quote is None else in_quote)
        elif char == "#" and in_quote is None:
            return line[:index]
    return line


def parse_collection_yaml(text):
    """Parse the collection YAML manifest (using PyYAML when available, else stdlib subset)."""
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    lines = []
    for raw in text.splitlines():
        cleaned = _strip_comment(raw).rstrip()
        if cleaned.strip():
            lines.append((len(cleaned) - len(cleaned.lstrip(" ")), cleaned.strip()))
    root, index = {}, 0

    def parse_block(min_indent):
        nonlocal index
        if index >= len(lines):
            return {}
        if lines[index][1].startswith("- "):
            items = []
            while index < len(lines) and lines[index][0] == min_indent and lines[index][1].startswith("- "):
                content = lines[index][1][2:].strip()
                index += 1
                if ":" in content:
                    k, v = content.split(":", 1)
                    item = {k.strip(): _parse_scalar(v) if v.strip() else None}
                    if not v.strip() and index < len(lines) and lines[index][0] > min_indent:
                        item[k.strip()] = parse_block(lines[index][0])
                    if index < len(lines) and lines[index][0] > min_indent and not lines[index][1].startswith("- "):
                        extra = parse_block(lines[index][0])
                        if isinstance(extra, dict):
                            item.update(extra)
                    items.append(item)
                else:
                    items.append(_parse_scalar(content))
            return items
        mapping = {}
        while index < len(lines) and lines[index][0] == min_indent and not lines[index][1].startswith("- "):
            k, v = lines[index][1].split(":", 1)
            key, val = k.strip(), v.strip()
            index += 1
            if val:
                mapping[key] = _parse_scalar(val)
            elif index < len(lines) and lines[index][0] > min_indent:
                mapping[key] = parse_block(lines[index][0])
            else:
                mapping[key] = {}
        return mapping

    if lines:
        root = parse_block(lines[0][0])
    return root if isinstance(root, dict) else {}


@dataclass(frozen=True)
class CollectionEntry:
    task: str
    seed: int
    attempt: int
    material: str
    success_only: bool
    dataset_id: str
    max_saved_episodes: int

    def to_dict(self):
        return asdict(self)


def load_collection_manifest(path):
    """Expand ``collection.yaml`` into explicit entries and per-task metadata."""
    raw = parse_collection_yaml(Path(path).read_text())
    defaults = raw.get("defaults", {}) or {}
    default_material = defaults.get("material", "bambu_tpu_90a")
    default_success_only = bool(defaults.get("success_only", True))
    default_attempts = int(defaults.get("attempts_per_seed", 1))
    tasks = raw.get("tasks", []) or []
    entries, task_configs = [], []
    for spec in tasks:
        task = spec["task"]
        enabled = bool(spec.get("enabled", True))
        seeds = [int(s) for s in (spec.get("seeds") or [])]
        attempts = int(spec.get("attempts_per_seed", default_attempts))
        material = str(spec.get("material", default_material))
        success_only = bool(spec.get("success_only", default_success_only))
        dataset_id = str(spec.get("dataset_id", f"x5_90a_{task}_success_v1"))
        max_saved = int(spec.get("max_saved_episodes", 100))
        task_configs.append(dict(task=task, enabled=enabled, seeds=seeds, attempts_per_seed=attempts,
                                 material=material, success_only=success_only, dataset_id=dataset_id,
                                 max_saved_episodes=max_saved))
        if not enabled:
            continue
        for seed in seeds:
            for attempt in range(attempts):
                entries.append(CollectionEntry(
                    task=task, seed=seed, attempt=attempt, material=material,
                    success_only=success_only, dataset_id=dataset_id, max_saved_episodes=max_saved))
    return dict(defaults=defaults, tasks=task_configs, entries=entries)


def committed_episode_count(task):
    info = ROOT / "data/datasets" / task / "meta/info.json"
    if not info.is_file():
        return 0
    return int(json.loads(info.read_text()).get("total_episodes", 0))


def check_entry_inputs(entry):
    """Return ``(ok, category, detail)`` without modifying or substituting inputs."""
    if entry.task not in IMPLEMENTED_TASKS:
        return False, "unsupported", f"No soft-finger adapter for {entry.task}"
    path = layout_path(entry.task, entry.seed)
    if not path.is_file():
        return False, "input_missing", f"Official layout missing: {path}"
    try:
        validate_layout_assets(json.loads(path.read_text()))
    except Exception as error:
        return False, "input_missing", str(error)
    return True, "ready", str(path)


def run_batch(manifest_path, *, batch_id=None, timeout_s=1500, dry_run=False, resume=True):
    """Execute or validate a YAML collection manifest serially."""
    manifest = load_collection_manifest(manifest_path)
    batch_id = batch_id or time.strftime("batch_%Y%m%d_%H%M%S")
    batch_dir = ROOT / "artifacts/batches" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "manifest_snapshot.json").write_text(json.dumps(dict(
        source=str(Path(manifest_path).resolve()),
        defaults=manifest["defaults"],
        tasks=manifest["tasks"],
        entries=[entry.to_dict() for entry in manifest["entries"]],
    ), indent=2))

    state_path = batch_dir / "summary.json"
    prior = json.loads(state_path.read_text())["runs"] if resume and state_path.is_file() else []
    done_keys = {(r["task"], r["seed"], r["attempt"]) for r in prior} if resume else set()
    runs = list(prior)

    for entry in manifest["entries"]:
        key = (entry.task, entry.seed, entry.attempt)
        if key in done_keys:
            continue
        saved = committed_episode_count(entry.task)
        if saved >= entry.max_saved_episodes:
            runs.append({**entry.to_dict(), "status": "skipped_budget",
                         "detail": f"Dataset already has {saved} >= {entry.max_saved_episodes} episodes"})
            continue
        ok, category, detail = check_entry_inputs(entry)
        if not ok:
            runs.append({**entry.to_dict(), "status": category, "detail": detail})
            continue
        if dry_run:
            runs.append({**entry.to_dict(), "status": "ready", "detail": detail})
            continue

        run_id = f"{batch_id}_{entry.task}_s{entry.seed}_a{entry.attempt}"
        log_path = batch_dir / f"{entry.task}_s{entry.seed}_a{entry.attempt}.log"
        command = environment("robodojo").module_command(
            "soft_sim.cli.collect", "--task", entry.task, "--seed", str(entry.seed),
            "--material", entry.material, "--dataset-id", entry.dataset_id, "--run-id", run_id,
            *(["--no-success-only"] if not entry.success_only else []))
        env = child_environment("robodojo", dict(
            ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES", PRIVACY_CONSENT="Y",
            SOFT_SIM_FOLLOW_BACKEND="gpu", OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        started = time.time()
        with log_path.open("w") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout_s)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = process.returncode
        run_dir = ROOT / "artifacts/runs" / run_id
        task_json = run_dir / "results/task.json"
        result = json.loads(task_json.read_text()) if task_json.is_file() else None
        if timed_out:
            status = "timeout"
        elif result and result.get("success") and returncode == 0:
            status = "success"
        elif result and result.get("outcome") == "planning_failed":
            status = "planning_failed"
        elif result and result.get("outcome") == "task_failed":
            status = "task_failed"
        else:
            status = "runtime_error"
        runs.append({
            **entry.to_dict(), "run_id": run_id, "status": status, "returncode": returncode,
            "wall_s": round(time.time() - started, 3), "log": str(log_path),
            "dataset_root": str(ROOT / "data/datasets" / entry.task),
        })
        state_path.write_text(json.dumps(_summarize(manifest, runs), indent=2))

    summary = _summarize(manifest, runs)
    state_path.write_text(json.dumps(summary, indent=2))
    return summary


def _summarize(manifest, runs):
    counts = {}
    for item in runs:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    datasets = {
        spec["task"]: dict(
            enabled=spec["enabled"],
            dataset_root=str(ROOT / "data/datasets" / spec["task"]),
            committed_episodes=committed_episode_count(spec["task"]),
        )
        for spec in manifest["tasks"]
    }
    return dict(
        implemented_tasks=len(IMPLEMENTED_TASKS),
        counts=counts,
        datasets=datasets,
        runs=runs,
    )
