"""A small NumPy image-to-force baseline, shared across simulation backends.

Uses two camera images, their open-gripper reference images, lens intrinsics,
mount tilts and commanded jaw opening. Does not use simulator deformation,
object identity, or force data as input features.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np


def image_features(rgb, reference_rgb, intrinsics, angles_deg, jaw_width_m, grid=8):
    rgb, reference_rgb = np.asarray(rgb), np.asarray(reference_rgb)
    if rgb.shape != reference_rgb.shape or rgb.ndim != 4 or rgb.shape[0] != 2 or rgb.shape[-1] != 3:
        raise ValueError('Expected matching left/right RGB images (2,H,W,3)')
    _, height, width, _ = rgb.shape
    if min(height, width) < grid:
        raise ValueError('Images are smaller than the feature grid')
    edges_y, edges_x = np.linspace(0, height, grid + 1, dtype=int), np.linspace(0, width, grid + 1, dtype=int)
    def pool(images):
        return np.stack([images[:, y0:y1, x0:x1].mean(axis=(1, 2))
                         for y0, y1 in zip(edges_y[:-1], edges_y[1:])
                         for x0, x1 in zip(edges_x[:-1], edges_x[1:])]).ravel() / 255.
    current, reference = pool(rgb), pool(reference_rgb)
    k = np.asarray(intrinsics)
    calibration = np.r_[k[:, 0, 0] / width, k[:, 1, 1] / height,
                        k[:, 0, 2] / width, k[:, 1, 2] / height,
                        np.asarray(angles_deg) / 60., np.nan_to_num(jaw_width_m) / .085,
                        float(not np.isfinite(jaw_width_m))]
    return np.r_[current, current - reference, calibration]


@dataclass
class RidgeForceRegressor:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    target_source: str = 'native'
    grid: int = 8

    @classmethod
    def fit(cls, x, y, alpha=10., target_source='native'):
        if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Need at least two finite training observations')
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError('Ridge alpha must be finite and positive')
        mean, scale, bias = x.mean(axis=0), x.std(axis=0), y.mean(axis=0)
        scale[scale < 1e-6] = 1.
        z, target = (x - mean) / scale, y - bias
        if len(x) < x.shape[1]:
            weights = z.T @ np.linalg.solve(z @ z.T + alpha * np.eye(len(z)), target)
        else:
            weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ target)
        return cls(mean, scale, weights, bias, target_source)

    def predict_features(self, features):
        return np.maximum(0., (features - self.mean) / self.scale @ self.weights + self.bias)

    def predict(self, frame, reference):
        feature = image_features(frame['rgb'], reference['rgb'], frame['camera_intrinsics'],
                                 frame['angles_deg'], float(frame['jaw_width_m']), self.grid)
        return self.predict_features(feature)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as stream:
            np.savez_compressed(stream, mean=self.mean, scale=self.scale, weights=self.weights, bias=self.bias,
                                target_source=self.target_source, grid=self.grid, model_version=1)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data['model_version']) != 1:
                raise ValueError('Unsupported force model version')
            return cls(data['mean'], data['scale'], data['weights'], data['bias'],
                       str(data['target_source']), int(data['grid']))
