"""Identical collection CLI for both isolated Python environments."""

import argparse
from dataclasses import replace
from itertools import product
from pathlib import Path

from .config import ExperimentConfig
from .dataset import collect_episode
from .experiment import EpisodeSpec


def main(environment_class, *, add_arguments=None, configure_simulation=None):
    parser = argparse.ArgumentParser(description='Soft fingers, attached internal cameras and force/deformation collection')
    parser.add_argument('--config')
    parser.add_argument('--episodes', type=int, default=1, help='Number of randomized objects/seeds')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=f'data/runs/{environment_class.backend_name}')
    parser.add_argument('--profile', choices=['robotiq_2f85', 'panda', 'angular'])
    parser.add_argument('--gui', action='store_true', help='Open a local window (requires DISPLAY on Linux)')
    if add_arguments is not None:
        add_arguments(parser)
    parser.add_argument('--no-render', action='store_true', help='Physics/deformation only, no camera training data')
    parser.add_argument('--sweep-tilts', type=float, nargs='+', help='Symmetric soft-finger mount tilts, degrees')
    parser.add_argument('--sweep-focals', type=float, nargs='+', help='Same focal length in both cameras, mm')
    parser.add_argument('--sweep-widths', type=float, nargs='+', help='Final nominal jaw openings, meters')
    parser.add_argument('--sweep-angles', type=float, nargs='+', help='Angular-profile final closing angles, degrees')
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error('--episodes must be positive and --seed must be nonnegative')
    config = ExperimentConfig.load(args.config)
    if args.profile:
        config.gripper.profile = args.profile
        if args.profile == 'panda':
            config.gripper.open_jaw_width_m = min(.080, config.gripper.open_jaw_width_m)
    config.simulation.headless = not args.gui
    config.simulation.render = not args.no_render
    if configure_simulation is not None:
        configure_simulation(parser, args, config.simulation)
    if config.simulation.livestream and environment_class.backend_name != 'isaacsim':
        parser.error('WebRTC livestream is only supported by the Isaac Sim backend')
    config.validate()
    # Validate every sweep override before starting either simulation runtime.
    for values, name in ((args.sweep_tilts, 'tilt'), (args.sweep_focals, 'focal'),
                         (args.sweep_widths, 'width'), (args.sweep_angles, 'angle')):
        for value in values or []:
            import math
            if not math.isfinite(value):
                parser.error(f'Non-finite sweep {name}')
            if name == 'tilt' and not 0 <= value < 60:
                parser.error('Sweep tilts must be in [0,60)')
            if name == 'focal' and value <= 0:
                parser.error('Sweep focal lengths must be positive')
            if name == 'width' and not 0 <= value <= config.gripper.open_jaw_width_m:
                parser.error('Sweep widths must be between 0 and the open jaw width')
            if name == 'angle' and not config.gripper.open_angle_deg <= value < 80:
                parser.error('Sweep angles must be between the open angle and 80 degrees')
    if args.sweep_angles and config.gripper.profile != 'angular':
        parser.error('--sweep-angles requires --profile angular; parallel profiles use --sweep-widths')
    if args.sweep_widths and config.gripper.profile == 'angular':
        parser.error('--sweep-widths is for parallel profiles')
    with environment_class(config) as env:
        for object_index in range(args.episodes):
            seed = args.seed + object_index
            base = EpisodeSpec.sample(config, seed)
            combinations = product(args.sweep_tilts or [None], args.sweep_focals or [None],
                                   args.sweep_widths or [None], args.sweep_angles or [None])
            for index, (tilt, focal, width, angle) in enumerate(combinations):
                episode = replace(base)
                if tilt is not None:
                    episode.mount_angles_deg = (tilt, tilt)
                if focal is not None:
                    episode.focal_lengths_mm = (focal, focal)
                if width is not None:
                    episode.final_jaw_width_m = width
                if angle is not None:
                    episode.close_angles_deg = (angle, angle)
                path, meta = collect_episode(env, Path(args.output), seed, index, episode)
                print(f'{path}: {meta["frame_count"]} frames, peak native squeeze={meta["peak_native_squeeze_force_n"]} N', flush=True)
        if config.simulation.livestream:
            env.wait_for_close()
