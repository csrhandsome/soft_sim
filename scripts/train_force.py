"""Train an image-to-force baseline with validation grouped by object/episode seed."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from shared.learning import RidgeForceRegressor, image_features


def load_samples(root, target_source):
    x, y, groups, source_files = [], [], [], []
    for metadata_path in sorted(Path(root).rglob('episode.json')):
        if any(part.startswith('.') for part in metadata_path.relative_to(root).parts):
            continue
        metadata = json.loads(metadata_path.read_text())
        if not metadata.get('complete') or metadata.get('schema_version') != 1:
            continue
        directory = metadata_path.parent
        with np.load(directory / 'reference.npz', allow_pickle=False) as reference:
            if 'rgb' not in reference:
                continue
            for frame_path in sorted(directory.glob('frame_*.npz')):
                with np.load(frame_path, allow_pickle=False) as frame:
                    if 'rgb' not in frame:
                        continue
                    if target_source == 'native':
                        if not bool(frame['force_valid']):
                            continue
                        target = frame['normal_force_n']
                    else:
                        # Only late hold frames; quasi-static estimate is invalid during motion.
                        g = metadata['config']['gripper']
                        if bool(frame['force_valid']) or float(frame['time_s']) < g['close_duration_s'] + .5 * g['hold_duration_s']:
                            continue
                        if str(frame['force_source']) != 'quasistatic_corotated_fem_estimate':
                            continue
                        vector = frame['estimated_force_world_n']
                        target = np.maximum(0., [-vector[0, 0], vector[1, 0]])
                    if not np.isfinite(target).all():
                        continue
                    features = image_features(frame['rgb'], reference['rgb'], frame['camera_intrinsics'],
                                              frame['angles_deg'], float(frame['jaw_width_m']))
                    x.append(features)
                    y.append(target)
                    # Keep all frames AND all swept camera/tilt configurations of
                    # one sampled object seed on the same side of the split.
                    groups.append(metadata['episode']['seed'])
                    source_files.append(str(frame_path))
    if not x:
        raise ValueError('No eligible RGB + force pairs. Render data first. Isaac estimates require --target-source estimated.')
    return np.asarray(x), np.asarray(y), np.asarray(groups), source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--output', default='data/models/force_ridge.npz')
    parser.add_argument('--target-source', choices=['native', 'estimated'], default='native')
    parser.add_argument('--alpha', type=float, default=10.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or args.seed < 0:
        parser.error('Require 0 < validation fraction < 1 and seed >= 0')
    x, y, groups, paths = load_samples(args.dataset, args.target_source)
    unique = np.unique(groups)
    if len(unique) < 3:
        parser.error('Collect at least three independent episode seeds; sweeps of one seed are not independent episodes')
    if np.max(np.std(y, axis=0)) < 1e-8:
        parser.error('All force targets are constant; collect different contact/deformation levels first')
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique)
    count = min(len(unique) - 1, max(1, int(np.ceil(len(unique) * args.validation_fraction))))
    validation = np.isin(groups, unique[:count])
    model = RidgeForceRegressor.fit(x[~validation], y[~validation], args.alpha, args.target_source)
    predictions = model.predict_features(x[validation])
    error = predictions - y[validation]
    metrics = dict(target_source=args.target_source, train_frames=int((~validation).sum()),
                   validation_frames=int(validation.sum()), train_seeds=unique[count:].tolist(),
                   validation_seeds=unique[:count].tolist(), mae_per_finger_n=np.mean(abs(error), axis=0).tolist(),
                   squeeze_mae_n=float(np.mean(abs(error.mean(axis=1)))),
                   validation_files=[p for p, selected in zip(paths, validation) if selected],
                   note='Baseline validation measures simulation labels only, not real-world force accuracy.')
    model.save(args.output)
    Path(args.output).with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({key: value for key, value in metrics.items() if key != 'validation_files'}, indent=2))


if __name__ == '__main__':
    main()
