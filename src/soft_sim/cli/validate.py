"""``validate``: run every independent quality check for one run."""

import argparse
import json
from pathlib import Path

from soft_sim.storage import Storage


def resolve_run(run_id=None, run=None, storage_root=None):
    if run is not None:
        return Path(run).resolve()
    if run_id is None:
        raise SystemExit("Pass a run directory or --run-id")
    root = Storage.create(storage_root).run_dir(run_id)
    if not (root / "manifest.json").is_file():
        raise SystemExit(f"Unknown run: {root}")
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="Run directory")
    parser.add_argument("--run-id")
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--log", type=Path, help="Simulation log for the initialization diagnostics")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args(argv)
    from soft_sim.validation.task import audit_run
    report = audit_run(resolve_run(args.run_id, args.run, args.storage_root),
                       storage_root=args.storage_root, workers=args.workers, log_path=args.log)
    summary = {key: value for key, value in report.items() if key != "commands"}
    print(json.dumps(summary, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
