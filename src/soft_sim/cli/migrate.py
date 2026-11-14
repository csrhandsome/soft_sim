"""``migrate``: list (default) or copy legacy outputs into the storage root."""

import argparse
import json
from pathlib import Path

from soft_sim.storage import Storage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--apply", action="store_true", help="Copy instead of only listing")
    parser.add_argument("--report", type=Path, help="Write the plan/result JSON here")
    args = parser.parse_args(argv)
    from soft_sim.migration import apply_plan, build_plan, verify
    storage = Storage.create(args.storage_root)
    plan = build_plan(storage)
    if args.apply:
        apply_plan(plan)
        problems = verify(plan, storage)
        plan.notes.extend(problems)
    document = plan.to_dict()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(document, indent=2))
    print(json.dumps(document, indent=2))
    print(f"{len(plan.copies)} copy targets, {len(plan.unclassified)} unclassified items"
          + ("" if args.apply else " (dry run; pass --apply to copy)"))
    return 0 if not plan.notes or not args.apply else (1 if any("missing" in n for n in plan.notes) else 0)


if __name__ == "__main__":
    raise SystemExit(main())
