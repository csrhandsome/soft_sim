"""``validate-dataset``: verify a LeRobot v3 dataset root (metadata, state/action, and MP4 shards)."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from soft_sim.config import ROOT
from soft_sim.storage import formal_datasets_lock, formal_datasets_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, help="LeRobot v3 dataset root directory")
    parser.add_argument("--task", help="Resolve formal dataset root under data/datasets/<task>")
    parser.add_argument("--episode", type=int, help="Validate a single episode index within the dataset")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON report")
    parser.add_argument("--require-success", action=argparse.BooleanOptionalAction, default=True,
                        help="Require simulation provenance confirming task_success=True (default: True)")
    parser.add_argument("--require-external", action=argparse.BooleanOptionalAction, default=True,
                        help="Require external_rgb feature and MP4 (default: True)")
    parser.add_argument("--require-fem", action=argparse.BooleanOptionalAction, default=False,
                        help="Require FEM node features in info.json and parquet schema (default: False)")
    parser.add_argument("--skip-video-decode", action="store_true",
                        help="Only check MP4 existence/size instead of full-frame decoding")
    args = parser.parse_args(argv)

    if args.dataset is not None:
        dataset_root = args.dataset.resolve()
    elif args.task:
        dataset_root = (ROOT / "data/datasets" / args.task).resolve()
    else:
        parser.error("Provide a dataset directory or --task <name>")

    from soft_sim.validation.dataset import validate_dataset
    formal_root = formal_datasets_root().resolve(strict=False)
    is_formal = bool(args.task) or dataset_root == formal_root or formal_root in dataset_root.parents
    lock_ctx = formal_datasets_lock() if is_formal else nullcontext()
    with lock_ctx:
        report = validate_dataset(
            dataset_root,
            episode_index=args.episode,
            require_external_rgb=args.require_external,
            require_fem=args.require_fem,
            require_success=args.require_success,
            check_video_decode=not args.skip_video_decode,
        )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
