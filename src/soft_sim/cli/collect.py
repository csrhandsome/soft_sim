"""``collect`` / ``export-scene`` / ``replay`` / ``assembly``: RoboDojo environment."""

import argparse
import os
from pathlib import Path

from soft_sim.config import MATERIAL_PRESETS, ROOT

MODE_BY_COMMAND = {"collect": "collect", "export-scene": "export",
                   "replay": "replay", "assembly": "assembly"}


def build_parser(argv=None):
    default_mode = MODE_BY_COMMAND.get(os.environ.get("SOFT_SIM_COMMAND", ""), None)
    parser = argparse.ArgumentParser(
        prog="run.py collect",
        description="Run the original RoboDojo task with replacement soft fingers.")
    parser.add_argument("--config", type=Path,
                        help="Task configuration JSON (default: configs/<task>.json when present)")
    parser.add_argument("--material", choices=MATERIAL_PRESETS,
                        help="Material preset from assets/materials (default: Bambu TPU 90A)")
    parser.add_argument("--task", choices=("general_pickup", "stack_blocks"))
    parser.add_argument("--layout", type=Path, help="Saved layout JSON; defaults to the official one")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--active-arm", type=int, choices=(0, 1))
    parser.add_argument("--mode", choices=("export", "assembly", "replay", "collect"),
                        default=default_mode)
    parser.add_argument("--grasp-plan", type=Path, help="Saved plan for replay")
    parser.add_argument("--run-id", help="Explicit run id; the default is generated")
    parser.add_argument("--storage-root", type=Path, help="Storage root (default: artifacts/)")
    parser.add_argument("--dataset-id", help="LeRobot collection name; seeds are episodes in it")
    parser.add_argument("--finger-scale", type=float, help="Uniform soft-finger scale")
    parser.add_argument("--target-label")
    parser.add_argument("--fps", type=int, help="Recording frame rate")
    parser.add_argument("--dt", type=float, help="Physics timestep in seconds")
    parser.add_argument("--adapter-offset", type=float, help="Adapter offset in meters")
    parser.add_argument("--fem-mesh", type=Path, help="Conforming tetrahedral mesh NPZ")
    parser.add_argument("--solver-iterations", type=int)
    parser.add_argument("--vertex-velocity-damping", type=float)
    parser.add_argument("--joint-control", choices=("position", "kinematic"),
                        help="Position targets use the original X5 drives; kinematic is diagnostic")
    parser.add_argument("--contact-height", type=float, help="Contact band on the scaled finger, m")
    parser.add_argument("--indent", type=float, help="Per-side grasp preload, m")
    parser.add_argument("--planning-friction", type=float)
    parser.add_argument("--planning-attempts", type=int)
    parser.add_argument("--static-camera-probe-frames", type=int, default=0,
                        help="Optional zero-time diagnostic renders; never part of the dataset")
    parser.add_argument("--texture-max-resolution", type=int, choices=(512, 1024, 2048, 4096))
    return parser


def main(argv=None):
    args = build_parser(argv).parse_args(argv)
    if args.mode is None:
        args.mode = "collect"
    if args.config is None:
        candidate = ROOT / "configs" / f"{(args.task or 'general_pickup')}.json"
        args.config = candidate if candidate.is_file() else None
    from soft_sim.tasks.collect import run
    run(args)


if __name__ == "__main__":
    raise SystemExit(main())
