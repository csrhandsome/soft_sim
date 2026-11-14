"""A frozen-state check must distinguish renderer motion from advancing physics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

if cv2 is not None:
    from soft_sim.validation.frozen_render import probe


@unittest.skipUnless(cv2 is not None, "Frozen render tests use the RoboDojo environment's OpenCV")
class FrozenRenderMeasurements(unittest.TestCase):
    def run_probe(self, *, shift=False, advance=False):
        rng = np.random.default_rng(79)
        gray = cv2.resize(rng.integers(0, 256, (128, 128), dtype=np.uint8), (256, 256))
        texture = np.repeat(gray[..., None], 3, axis=2)
        index = 0
        def capture():
            nonlocal index
            index += 1
            image = cv2.warpAffine(texture, np.float32([[1, 0, 2 * (index % 2)], [0, 1, 0]]),
                                   (256, 256)) if shift else texture.copy()
            return dict(right_rgb=image)
        def state():
            return dict(nodes=np.zeros((4, 3)), time_s=np.asarray(.001 * index if advance else 0.))
        with tempfile.TemporaryDirectory() as folder:
            return probe(capture, state, Path(folder) / "probe", count=4)

    def test_same_state_and_image_pass(self):
        self.assertTrue(self.run_probe()["valid"])

    def test_render_jitter_with_frozen_nodes_is_rejected(self):
        report = self.run_probe(shift=True)
        self.assertTrue(report["physical_state_unchanged"])
        self.assertFalse(report["cameras"]["right_rgb"]["passed"])
        self.assertFalse(report["valid"])

    def test_physics_advancing_with_stationary_image_is_rejected(self):
        report = self.run_probe(advance=True)
        self.assertTrue(report["cameras"]["right_rgb"]["passed"])
        self.assertFalse(report["physical_state_unchanged"])
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()
