"""Download selected RoboDojo assets from its public ModelScope dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import PurePosixPath
import urllib.parse
import urllib.request
import time

from soft_sim.config import ROOT
API = "https://www.modelscope.cn/api/v1/datasets/RoboDojo-Benchmark/RoboDojo/repo/"

def request_json(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = json.load(response)
            break
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if data.get("Code") != 200:
        raise RuntimeError(str(data))
    return data

def list_files(prefix, revision):
    queue, files = [prefix], []
    while queue:
        directory = queue.pop()
        page = 1
        while True:
            query = urllib.parse.urlencode(dict(Revision=revision, Root=directory,
                Recursive="false", PageNumber=page, PageSize=100))
            data = request_json(API + "tree?" + query)
            entries = data["Data"]["Files"]
            for entry in entries:
                path = PurePosixPath(entry["Path"])
                if path.is_absolute() or ".." in path.parts or not str(path).startswith(prefix + "/"):
                    raise ValueError("Unsafe asset path")
                if ".thumbs" in path.parts:
                    continue
                if entry["Type"] == "tree":
                    queue.append(str(path))
                else:
                    files.append(entry)
            if page * 100 >= data.get("TotalCount", data["Data"].get("TotalCount", len(entries))):
                break
            page += 1
    return files

def download(entry, revision):
    path = ROOT / "third_party/RoboDojo" / entry["Path"]
    expected = entry.get("Sha256")
    if path.is_file() and path.stat().st_size == entry["Size"]:
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode(dict(Revision=revision, FilePath=entry["Path"]))
    url = API.rstrip("/") + "?" + query
    partial = path.with_name(path.name + ".partial")
    for attempt in range(3):
        try:
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=90) as source, partial.open("wb") as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
            if size != entry["Size"] or (expected and digest.hexdigest() != expected):
                raise RuntimeError("Asset size/hash mismatch: " + entry["Path"])
            partial.replace(path)
            print("Downloaded", entry["Path"], size, flush=True)
            return
        except Exception:
            if attempt == 2:
                raise


def prepare_task_inputs(task, seed, *, revision="master", max_gb=1.):
    """Fetch one official layout, its selected models and environment assets.

    This is input preparation, not an episode runner or a layout generator.
    Keep a manifest with downloaded hashes even if a later model is unavailable.
    """
    from soft_sim.tasks.adapters import IMPLEMENTED_TASKS
    if task not in IMPLEMENTED_TASKS or seed < 0:
        raise ValueError("Input preparation requires an implemented task and nonnegative seed")
    prefix = "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    filename = f"{task}_{seed}.json"
    selected = [entry for entry in list_files(prefix, revision)
                if PurePosixPath(entry["Path"]).name == filename]
    if len(selected) != 1:
        raise FileNotFoundError(f"Official layout missing: {filename}; no substitute generated")
    total = selected[0]["Size"]
    if total > max_gb * 1e9:
        raise ValueError("Layout exceeds download budget")
    download(selected[0], revision)
    layout = json.loads((ROOT / "third_party/RoboDojo" / selected[0]["Path"]).read_text())
    models = set()
    from soft_sim.sim.robodojo import layout_materials
    models.update(f"Assets/Material/{name}" for name in layout_materials(layout))
    room = layout.get("Room", {}).get("default")
    if room:
        models.add(f"Assets/Room/{room}")
    for kind in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"):
        for category, records in layout.get(kind, {}).items():
            for record in records:
                if not isinstance(record.get("category_idx"), int):
                    raise ValueError("Official layout model index must be an integer")
                object_kind = "Clutter" if record.get("type") == "cluttered" else kind
                models.add(f"Assets/Object/RoboDojo/{object_kind}/{category}/{record['category_idx']:05d}")
    manifest_path = ROOT / "assets/downloads" / f"task_{task}_s{seed}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(source=API, revision=revision, task=task, seed=seed, files=[], errors=[])

    def record(path, origin):
        manifest["files"].append(dict(path=str(path.relative_to(ROOT / "third_party/RoboDojo")),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source=origin))

    record(ROOT / "third_party/RoboDojo" / selected[0]["Path"], API)
    try:
        background = layout.get("Background", {}).get("category_name")
        if background:
            background_path = ROOT / "third_party/RoboDojo/Assets/Background" / background
            if background_path.is_file():
                record(background_path, "existing_local_asset")
            else:
                matches = [entry for entry in list_files("Assets/Background", revision)
                           if PurePosixPath(entry["Path"]).name == background]
                if len(matches) != 1:
                    raise FileNotFoundError(f"Official background missing: {background}")
                total += matches[0]["Size"]
                if total > max_gb * 1e9:
                    raise ValueError("Task inputs exceed --max-gb")
                download(matches[0], revision)
                record(background_path, API)
        for model_prefix in sorted(models):
            model = ROOT / "third_party/RoboDojo" / model_prefix
            if (model / "metadata.json").is_file() and any((model / f"object.{ext}").is_file() for ext in ("usd", "usdz")):
                for path in sorted(model.rglob("*")):
                    if path.is_file():
                        record(path, "existing_local_asset")
                continue
            try:
                files = list_files(model_prefix, revision)
                if not files:
                    raise FileNotFoundError(f"Empty official model: {model_prefix}")
                total += sum(entry["Size"] for entry in files)
                if total > max_gb * 1e9:
                    raise ValueError("Task inputs exceed --max-gb")
                for entry in files:
                    download(entry, revision)
                    record(ROOT / "third_party/RoboDojo" / entry["Path"], API)
            except (OSError, RuntimeError, ValueError) as error:
                manifest["errors"].append(dict(model=model_prefix, message=str(error)))
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if manifest["errors"]:
        raise RuntimeError(f"Task inputs incomplete; see {manifest_path}")
    print("TASK_INPUTS_READY", task, seed, manifest_path, flush=True)
    return manifest

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="Assets/Robots/franka")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--max-gb", type=float, default=1.)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--task", help="Prepare one implemented task's official layout and selected models")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.task:
        if args.list:
            parser.error("--list is for prefix downloads; task preparation writes a provenance manifest")
        prepare_task_inputs(args.task, args.seed, revision=args.revision, max_gb=args.max_gb)
        return
    if not args.prefix.startswith("Assets/") or ".." in PurePosixPath(args.prefix).parts:
        parser.error("Prefix must be a relative Assets/ subdirectory")
    files = list_files(args.prefix.rstrip("/"), args.revision)
    total = sum(entry["Size"] for entry in files)
    print("ASSET_SELECTION", args.prefix, "files=", len(files), "bytes=", total, flush=True)
    if args.list:
        for entry in files:
            print(entry["Path"], entry["Size"])
        return
    if not files or total > args.max_gb * 1e9:
        raise RuntimeError("Empty selection or download exceeds --max-gb")
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda entry: download(entry, args.revision), files))
    downloads = ROOT / "assets/downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    manifest = downloads / ("robodojo_" + args.prefix.replace("/", "_") + "_manifest.json")
    manifest.write_text(json.dumps(dict(source=API, revision=args.revision,
        prefix=args.prefix, files=files), indent=2))
    print("ASSET_READY", args.prefix, flush=True)

if __name__ == "__main__":
    raise SystemExit(main())
