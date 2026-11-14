"""``python -m soft_sim.cli <command>`` mirrors ``scripts/run.py``."""

import sys

from soft_sim.commands import COMMANDS, resolve


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("Commands:")
        for name, spec in COMMANDS.items():
            print(f"  {name:20s} {spec['help']} ({spec['env']})")
        return 0 if argv else 2
    name = resolve(argv[0])
    if name not in COMMANDS:
        print(f"Unknown command: {argv[0]}", file=sys.stderr)
        return 2
    import importlib
    module = importlib.import_module(COMMANDS[name]["module"])
    result = module.main(argv[1:])
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
