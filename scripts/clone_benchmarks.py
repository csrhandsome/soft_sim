"""Prepare pinned benchmark source snapshots without Git repositories."""
import argparse
import os
import tomllib

from prepare_sources import ROOT, detach, prepare

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
specs = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["soft-sim"]["source-repositories"]
names = [name for name in ("robodojo", "robotwin") if args.target in ("all", name)]
if "robodojo" in names:
    names += ["isaaclab", "curobo"]
for name in names:
    spec = specs[name]
    if args.dry_run:
        print(f"{spec['url']} @ {spec['rev']} -> {ROOT / spec['path']}; snapshot without .git")
    else:
        prepare(name, spec)
if not args.dry_run:
    detach()
