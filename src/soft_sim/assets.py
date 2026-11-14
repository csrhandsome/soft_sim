"""Download selected RoboDojo assets from its public ModelScope dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import PurePosixPath
import urllib.parse
import urllib.request

from soft_sim.config import ROOT
API = "https://www.modelscope.cn/api/v1/datasets/RoboDojo-Benchmark/RoboDojo/repo/"

def request_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
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

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="Assets/Robots/franka")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--max-gb", type=float, default=1.)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
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
