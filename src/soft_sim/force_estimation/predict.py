"""Predict per-finger contact force from saved images/calibration, without a simulator."""

import argparse
import json
from pathlib import Path

import numpy as np
from soft_sim.force_estimation.regressor import RidgeForceRegressor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--frame', required=True)
    parser.add_argument('--reference', help='Defaults to reference.npz beside the frame')
    args = parser.parse_args(argv)
    model = RidgeForceRegressor.load(args.model)
    reference_path = args.reference or Path(args.frame).parent / 'reference.npz'
    with np.load(args.frame, allow_pickle=False) as frame, np.load(reference_path, allow_pickle=False) as reference:
        force = model.predict(frame, reference)
    print(json.dumps(dict(normal_force_per_finger_n=force.tolist(), squeeze_force_n=float(force.mean()),
                          target_source=model.target_source), indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
