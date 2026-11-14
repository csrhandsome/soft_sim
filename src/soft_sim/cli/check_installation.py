"""``check-installation``: offline X5 mount geometry, gap and collision audit."""

import argparse
from pathlib import Path

from soft_sim.storage import RunStore, Storage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, help="Override the report directory")
    args = parser.parse_args(argv)
    if args.output is not None:
        target = args.output.resolve()
        run_id = None
    else:
        store = RunStore.create(Storage.create(args.storage_root), command="check-installation",
                                run_id=args.run_id)
        target = store.ensure("reports", "installation")
        run_id = store.run_id
        print(f"RUN {run_id} {store.root}", flush=True)
    from soft_sim.validation.installation import main as audit_main
    audit_main(["--output", str(target)])
    if run_id:
        store.record("installation_audit", target)
        store.update(execution="completed")
    print(f"REPORT {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
