"""Explicit migration of pre-storage-root output into the artifact buckets.

Dry-run by default: it lists what would move and where. ``--apply`` copies and
leaves the original in place; nothing is deleted. Legacy dataset references stay
readable because the official shards are copied whole, never reorganized.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from soft_sim.config import ROOT
from soft_sim.storage import slug

LEGACY_DATASETS = ROOT / "data/datasets"
LEGACY_RUNS = ROOT / "data/runs"
LEGACY_GENERATED = ROOT / "data/generated"
MESH_SUFFIXES = (".npz", ".stl", ".obj", ".ply", ".msh")


@dataclass
class MigrationPlan:
    copies: list = field(default_factory=list)
    unclassified: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def add(self, source, destination, reason):
        self.copies.append(dict(source=str(source), destination=str(destination), reason=reason))

    def to_dict(self):
        return dict(copies=self.copies, unclassified=self.unclassified, notes=self.notes)


def build_plan(storage):
    plan = MigrationPlan()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    if LEGACY_DATASETS.is_dir():
        for dataset in sorted(LEGACY_DATASETS.iterdir()):
            if not (dataset / "meta/info.json").is_file():
                plan.unclassified.append(dict(source=str(dataset), reason="not a LeRobot dataset"))
                continue
            plan.add(dataset, storage.dataset_dir(dataset.name), "official LeRobot dataset")
    if LEGACY_RUNS.is_dir():
        for run in sorted(LEGACY_RUNS.iterdir()):
            if not run.is_dir():
                continue
            destination = storage.run_dir(f"legacy_{stamp}_{slug(run.name, 24)}")
            plan.add(run, destination, "historical run")
    if LEGACY_GENERATED.is_dir():
        for path in sorted(LEGACY_GENERATED.iterdir()):
            if path.suffix.lower() in MESH_SUFFIXES and path.is_file():
                plan.add(path, storage.cache_dir("meshes") / path.name, "rebuildable mesh cache")
            elif path.name.endswith("_manifest.json"):
                plan.add(path, ROOT / "docs/archive" / path.name, "historical download manifest")
            elif path.is_dir():
                plan.unclassified.append(dict(source=str(path), reason="generated scene/plan/diagnostic"))
    for media in sorted(ROOT.glob("*.mp4")) + sorted(ROOT.glob("*.png")):
        plan.add(media, storage.run_dir(f"legacy_unclassified_{stamp}") / "previews" / media.name,
                 "unclassified pre-refactor media")
    plan.notes.append("Originals are never deleted; verify each copy before removing the source.")
    plan.notes.append("Legacy dataset episode references are preserved because shards are copied whole.")
    return plan


def apply_plan(plan):
    for entry in plan.copies:
        source, destination = Path(entry["source"]), Path(entry["destination"])
        if destination.exists():
            entry["skipped"] = "destination exists"
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        entry["copied"] = True
    return plan


def verify(plan, storage):
    """Re-check the copies that must be readable on their own."""
    problems = []
    for entry in plan.copies:
        destination = Path(entry["destination"])
        if not destination.exists():
            problems.append(f"missing copy: {destination}")
            continue
        if (destination / "meta/info.json").is_file():
            info = json.loads((destination / "meta/info.json").read_text())
            if info.get("codebase_version") != "v3.0":
                problems.append(f"unexpected dataset version: {destination}")
    return problems
