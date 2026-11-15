"""Load isolated baseline modules without changing the working source tree."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    args, replay_args = parser.parse_known_args()
    if args.baseline:
        for name, relative in (
                ("soft_sim.execution", "execution.py"),
                ("soft_sim.sim.soft_tool", "sim/soft_tool.py"),
                ("soft_sim.tasks.collect", "tasks/collect.py")):
            spec = importlib.util.spec_from_file_location(name, args.baseline / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    from soft_sim.cli.collect import main as collect
    collect(replay_args)


if __name__ == "__main__":
    main()
