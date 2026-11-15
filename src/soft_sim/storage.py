"""One configurable storage root; every generated file has a declared owner.

Three top-level buckets:

    <storage>/runs/<run_id>/                  single execution: parameters, scene, plans, logs
    <storage>/datasets/lerobot/<dataset_id>/  official LeRobot dataset
    <storage>/cache/                          rebuildable mesh/asset processing cache

Nothing else in the project writes run output, and no command writes into an
existing run directory.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import secrets

from soft_sim.config import ROOT


def default_storage_root():
    return Path(os.environ.get("SOFT_SIM_STORAGE") or ROOT / "artifacts").resolve()


def formal_datasets_root(project_root=ROOT):
    return Path(project_root) / "data" / "datasets"


def ensure_formal_datasets_ready(target_path=None, *, project_root=ROOT):
    """Refuse formal dataset writes when data/datasets is a dangling symlink or unmounted."""
    project_root = Path(project_root)
    base = formal_datasets_root(project_root)
    marker_path = project_root / "data" / ".datasets_storage_target.json"

    if base.is_symlink():
        raw_target = os.readlink(base)
        resolved = base.resolve(strict=False)
        if not base.exists() or not base.is_dir():
            raise RuntimeError(
                f"Formal dataset root {base} is a dangling or invalid symlink -> {raw_target}. "
                "Mount the external storage device before writing formal datasets."
            )
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text())
            mountpoint = marker.get("mountpoint")
            if mountpoint and not os.path.ismount(mountpoint):
                raise RuntimeError(
                    f"External dataset mountpoint {mountpoint} is not mounted; "
                    f"refusing to write under {base}."
                )
            expected_target = marker.get("target")
            if expected_target and str(resolved) != str(Path(expected_target).resolve()):
                raise RuntimeError(
                    f"Formal dataset symlink {base} -> {resolved} does not match expected target {expected_target}."
                )
        if not os.access(resolved, os.W_OK):
            raise RuntimeError(f"Formal dataset target {resolved} is not writable.")
    elif base.exists():
        if marker_path.is_file():
            raise RuntimeError(
                f"Formal dataset root {base} is a regular directory while storage marker "
                f"{marker_path} requires a symlink to external storage; refusing system-disk fallback."
            )
        if not base.is_dir():
            raise RuntimeError(f"Formal dataset root {base} exists and is not a directory.")
        if not os.access(base, os.W_OK):
            raise RuntimeError(f"Formal dataset root {base} is not writable.")
    elif marker_path.is_file():
        raise RuntimeError(
            f"Formal dataset symlink {base} is missing while storage marker {marker_path} is present."
        )

    if target_path is not None:
        target = Path(target_path)
        if target.is_symlink() and (not target.exists() or not target.is_dir()):
            raise RuntimeError(
                f"Dataset path {target} is a dangling symlink -> {os.readlink(target)}."
            )
    return base


@contextmanager
def formal_datasets_lock(project_root=ROOT):
    """Serialize formal dataset commits and storage switches without blocking simulation."""
    lock_path = Path(project_root) / "artifacts" / "cache" / ".formal_datasets.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_SLUG = re.compile(r"[^a-z0-9]+")


def slug(value, limit=32):
    text = _SLUG.sub("-", str(value).strip().lower()).strip("-")
    return (text or "task")[:limit]


def make_run_id(task, seed, *, now=None):
    """``20261114T080000Z_general-pickup_s22_a1b2c3``: stable from creation."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{slug(task)}_s{int(seed)}_{secrets.token_hex(3)}"


def json_default(value):
    """Serialize NumPy scalars/arrays and paths in run manifests and reports."""
    return _json_default(value)


def _write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")
    temporary.replace(path)


def _json_default(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


@dataclass(frozen=True)
class Storage:
    root: Path

    @classmethod
    def create(cls, path=None):
        root = Path(path).expanduser().resolve() if path else default_storage_root()
        return cls(root)

    def run_dir(self, run_id):
        return self.root / "runs" / run_id

    def dataset_dir(self, dataset_id):
        return self.root / "datasets" / "lerobot" / dataset_id

    def cache_dir(self, kind=""):
        return self.root / "cache" / kind if kind else self.root / "cache"

    def relative(self, path):
        """Storage-relative reference so a moved tree stays readable."""
        path = Path(path).resolve()
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


@dataclass
class RunStore:
    """Owns one immutable run directory and its single ``manifest.json``."""

    storage: Storage
    run_id: str
    command: str
    task: str | None = None
    seed: int | None = None

    @classmethod
    def create(cls, storage, *, command, task=None, seed=None, run_id=None):
        run_id = run_id or make_run_id(task or command, seed or 0)
        store = cls(storage, run_id, command, task, seed)
        if store.root.exists():
            raise FileExistsError(f"Run already exists; choose a new run id: {store.root}")
        store.root.mkdir(parents=True)
        store._write_manifest(dict(
            run_id=run_id, command=command, task=task, seed=seed,
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            storage_root=".", execution="created", capture="none",
            task_success=None, validation="pending", acceptance=None,
            artifacts={}, datasets=[], reports=[], errors=[]))
        return store

    @classmethod
    def open(cls, storage, run_id):
        root = storage.run_dir(run_id)
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError(f"Unknown run: {root}")
        manifest = json.loads((root / "manifest.json").read_text())
        return cls(storage, run_id, manifest["command"], manifest.get("task"), manifest.get("seed"))

    @property
    def root(self):
        return self.storage.run_dir(self.run_id)

    @property
    def manifest(self):
        return json.loads((self.root / "manifest.json").read_text())

    def path(self, *parts, create=False):
        path = self.root.joinpath(*parts)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure(self, *parts):
        return self.path(*parts, create=True)

    def _write_manifest(self, manifest):
        manifest.setdefault("run_id", self.run_id)
        manifest.setdefault("command", self.command)
        _write_json_atomic(self.root / "manifest.json", manifest)

    def update(self, **fields):
        manifest = self.manifest
        for key, value in fields.items():
            if key in ("artifacts", "datasets", "reports") and isinstance(value, dict):
                manifest.setdefault(key, {}).update(value)
            elif key in ("datasets", "reports") and isinstance(value, list):
                manifest.setdefault(key, []).extend(value)
            else:
                manifest[key] = value
        self._write_manifest(manifest)
        return manifest

    def append_error(self, message):
        manifest = self.manifest
        manifest.setdefault("errors", []).append(str(message))
        self._write_manifest(manifest)

    def record(self, name, relative_path):
        """Register an artifact by storage-relative path."""
        return self.update(artifacts={name: self.storage.relative(relative_path)})

    def write_json(self, relative, data, *, record=None):
        path = self.root / relative
        _write_json_atomic(path, data)
        if record:
            self.record(record, path)
        return path

    def read_json(self, relative, default=None):
        path = self.root / relative
        if not path.is_file():
            return {} if default is None else default
        return json.loads(path.read_text())

    def write_config_snapshot(self, config, cli_overrides=None):
        from soft_sim.config import CONFIG_VERSION
        snapshot = config.to_dict()
        snapshot["_format_version"] = CONFIG_VERSION
        snapshot["_cli_overrides"] = {k: v for k, v in (cli_overrides or {}).items() if v is not None}
        return self.write_json("config.resolved.json", snapshot, record="config")
