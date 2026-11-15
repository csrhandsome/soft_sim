#!/usr/bin/env python3
"""Single dispatch entry point: pick an interpreter, then run ``soft_sim.cli.<cmd>``.

Only the standard library is used before the target interpreter starts, so this
script does not require Isaac Sim, LeRobot or mplib. Run ``--help`` for the
command list and ``python scripts/run.py doctor`` to check the environments.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soft_sim.commands import COMMANDS, resolve  # noqa: E402  (stdlib-only module)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Soft Sim dispatcher. Every command prints its run id and output locations.")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, spec in COMMANDS.items():
        commands.add_parser(name, help=spec["help"], description=spec["help"],
                            add_help=False)
    return parser


def run_with_log(command, environment):
    """Tee Kit's native stdout/stderr into the run as soon as it is created."""
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
    log = None
    pending = []
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log is None:
                pending.append(line)
                if line.startswith("RUN "):
                    parts = line.rstrip().split(maxsplit=2)
                    if len(parts) == 3:
                        directory = Path(parts[2]) / "logs"
                        directory.mkdir(parents=True, exist_ok=True)
                        log = (directory / "simulation.log").open("a")
                        log.writelines(pending)
                        pending.clear()
            else:
                log.write(line)
            if log is not None:
                log.flush()
        return process.wait()
    finally:
        process.stdout.close()
        if log is not None:
            log.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        print("\nEach command has its own help: python scripts/run.py <command> --help")
        return 0 if argv else 2
    name = resolve(argv[0])
    if name not in COMMANDS:
        print(f"Unknown command: {argv[0]}", file=sys.stderr)
        build_parser().print_help(sys.stderr)
        return 2
    spec = COMMANDS[name]
    interpreter = Path(os.environ.get(f"SOFT_SIM_{spec['env'].upper()}_PYTHON")
                       or ROOT / "envs" / spec["env"] / ".venv/bin/python") \
        if spec["env"] != "core" else Path(
            os.environ.get("SOFT_SIM_CORE_PYTHON") or ROOT / ".venv/bin/python")
    if not interpreter.is_file():
        print(f"Missing {spec['env']} interpreter at {interpreter}.\n"
              f"Run: bash scripts/sync_envs.sh {spec['env']}", file=sys.stderr)
        return 1
    environment = dict(os.environ)
    environment.setdefault("ACCEPT_EULA", "Y")
    environment.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    environment.setdefault("PRIVACY_CONSENT", "Y")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "src"), environment.get("PYTHONPATH"))))
    command = [str(interpreter), "-m", spec["module"], *argv[1:]]
    if name in ("collect", "replay", "export-scene", "assembly"):
        return run_with_log(command, environment)
    return subprocess.run(command, cwd=os.getcwd(), env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
