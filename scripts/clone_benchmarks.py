"""Clone pinned benchmark checkouts without initializing Git submodules."""
from pathlib import Path
import argparse
import os
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("target", nargs="?", choices=("all", "robodojo", "robotwin"), default="all")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--direct", action="store_true")
args = parser.parse_args()
if args.direct:
    for key in list(os.environ):
        if key.lower() in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy"):
            os.environ.pop(key)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
os.environ["GIT_TERMINAL_PROMPT"] = "0"
specs = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["soft-sim"]["checkouts"]
for name, spec in specs.items():
    if args.target not in ("all", name.lower()):
        continue
    destination = ROOT / name
    if args.dry_run:
        print(f"{spec['url']} @ {spec['rev']} -> {destination}; no submodules")
        continue
    if destination.is_symlink():
        raise RuntimeError(f"Refusing symlink checkout: {destination}")
    if not destination.exists():
        subprocess.run(["git", "clone", "--no-recurse-submodules", "--no-checkout", spec["url"], str(destination)], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", spec["rev"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    origin = subprocess.check_output(["git", "-C", str(destination), "remote", "get-url", "origin"], text=True).strip()
    if actual != spec["rev"] or origin != spec["url"]:
        raise RuntimeError(f"Unexpected revision/origin at {destination}; preserve it and inspect manually")
if not args.dry_run:
    if (ROOT / "RoboDojo").exists():
        subprocess.run(["python3", str(ROOT / "scripts/prepare_sources.py"), "--detach"], check=True)
    else:
        from prepare_sources import detach
        detach()
