"""``collect-batch``: run or validate a YAML task collection manifest."""

import argparse
import json

from soft_sim.config import ROOT
from soft_sim.tasks.collection import run_batch


def main(argv=None):
    parser = argparse.ArgumentParser(prog="collect-batch")
    parser.add_argument("--config", default=str(ROOT / "configs/collection.yaml"))
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    summary = run_batch(args.config, batch_id=args.batch_id, timeout_s=args.timeout,
                        dry_run=args.dry_run, resume=not args.no_resume)
    print(json.dumps(summary["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
