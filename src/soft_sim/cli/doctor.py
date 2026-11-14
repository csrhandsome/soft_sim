"""``doctor``: report interpreters, packages, sources and paths."""

import argparse
import json

from soft_sim.runtime import doctor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args(argv)
    report = doctor()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"project root : {report['root']}")
    print(f"source path  : {report['source']}")
    print(f"python       : {report['python']}")
    for name, entry in report["environments"].items():
        status = entry.get("interpreter") or f"MISSING ({entry.get('error')})"
        print(f"{name:10s} : {status}")
    for relative, present in report["sources"].items():
        print(f"{'present' if present else 'MISSING':7s} : {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
