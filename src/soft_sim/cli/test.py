"""``test``: run one test group in the environment that owns its dependencies.

A missing dependency is an environment error, never a silent all-skip pass.
"""

import argparse
import re
import subprocess
import sys

from soft_sim.config import ROOT
from soft_sim.runtime import child_environment, environment

#: group -> (environment, test directory)
GROUPS = {
    "unit": ("core", "tests/unit"),
    "performance": ("core", "tests/performance"),
    "gpu-performance": ("robodojo", "tests/integration/gpu_performance"),
    "gpu-solver": ("robodojo", "tests/integration/gpu_solver"),
    "gpu-follow": ("robodojo", "tests/integration/gpu_follow"),
    "gpu-physics": ("robodojo", "tests/integration/gpu_physics"),
    "planning": ("planning", "tests/integration/planning"),
    "dataset": ("dataset", "tests/integration/dataset"),
    "meshing": ("meshing", "tests/integration/meshing"),
    "sim": ("robodojo", "tests/integration/sim"),
    "e2e": ("robodojo", "tests/e2e"),
}
DEFAULT_GROUPS = ("unit", "planning", "dataset")
SUMMARY = re.compile(r"^(OK|FAILED)\b.*?(?:skipped=(\d+))?.*$")


def run_group(name, verbose=False):
    if name not in GROUPS:
        raise SystemExit(f"Unknown group {name}; choose from {', '.join(GROUPS)}")
    env_name, directory = GROUPS[name]
    path = ROOT / directory
    if not path.is_dir() or not any(path.rglob("test_*.py")):
        return dict(group=name, environment=env_name, tests=0, status="no-tests", returncode=1)
    command = [str(environment(env_name).interpreter), "-m", "unittest", "discover",
               "-s", directory, "-t", ".", "-v"]
    if not verbose:
        command.remove("-v")
    print(f"TEST {name} ({env_name}): {' '.join(command)}", flush=True)
    process = subprocess.run(command, cwd=ROOT, env=child_environment(env_name),
                             capture_output=True, text=True)
    output = process.stdout + process.stderr
    tail = [line for line in output.strip().splitlines() if line.strip()][-1:] or [""]
    skipped = re.search(r"skipped=(\d+)", tail[0])
    tests = re.search(r"Ran (\d+) test", output)
    print(output, flush=True)
    return dict(group=name, environment=env_name,
                tests=int(tests.group(1)) if tests else 0,
                skipped=int(skipped.group(1)) if skipped else 0,
                status="passed" if process.returncode == 0 else "failed",
                returncode=process.returncode, summary=tail[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=tuple(GROUPS),
                        help="Test group; repeatable. Default: unit, planning, dataset")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    groups = args.group or list(DEFAULT_GROUPS)
    results = [run_group(name, args.verbose) for name in groups]
    print("\nGROUP     ENV        TESTS  SKIPPED  STATUS")
    for entry in results:
        print(f"{entry['group']:9s} {entry['environment']:10s} {entry['tests']:5d} "
              f"{entry.get('skipped', 0):8d}  {entry['status']}")
    failed = [entry["group"] for entry in results if entry["returncode"] != 0]
    if failed:
        print(f"FAILED groups: {', '.join(failed)}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
