"""Legacy NPZ recorder retained for explicit backwards-compatible diagnostics."""
import json
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    camera_names = ("left_rgb", "right_rgb", "external_rgb")

    def __init__(self, directory, fps, calibration):
        import cv2

        self.cv2 = cv2
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "frames").mkdir()
        self.fps = float(fps)
        self.calibration = calibration
        self.rows = []
        self.writers = {}
        self.shapes = None
        self.closed = False
        self._metadata("incomplete")

    def _metadata(self, status, result=None):
        data = dict(status=status, fps=self.fps, frame_count=len(self.rows),
                    duration_s=len(self.rows) / self.fps,
                    camera_order=list(self.camera_names), color_order="RGB",
                    camera_coordinates="optical: +X right, +Y down, +Z forward",
                    calibration=self.calibration, frames=self.rows, result=result)
        (self.directory / "episode.json").write_text(json.dumps(data, indent=2))

    def _writer(self, name, width, height):
        path = self.directory / (name + ".partial.mp4")
        writer = self.cv2.VideoWriter(str(path), self.cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open MP4 encoder: {path}")
        self.writers[name] = writer

    def append(self, left, right, external, *, time_s, phase, **state):
        if self.closed:
            raise RuntimeError("Episode recorder is already closed")
        images = [np.asarray(value) for value in (left, right, external)]
        for name, rgb in zip(self.camera_names, images):
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or rgb.std() < 1.:
                raise ValueError(f"Invalid or uniform RGB frame: {name}")
        if self.rows and not np.isclose(time_s - self.rows[-1]["time_s"], 1 / self.fps,
                                        atol=1e-7, rtol=0):
            raise ValueError("RGB samples must have a constant simulation-time interval")
        shapes = [rgb.shape for rgb in images]
        if self.shapes is None:
            self.shapes = shapes
            for name, rgb in zip(self.camera_names, images):
                self._writer(name, rgb.shape[1], rgb.shape[0])
            self._writer("episode", 1280, 480)
        elif shapes != self.shapes:
            raise ValueError("Camera resolution changed during the episode")
        index = len(self.rows)
        frame_file = f"frames/frame_{index:06d}.npz"
        np.savez_compressed(self.directory / frame_file,
                            **dict(zip(self.camera_names, images)), time_s=time_s, **state)
        row = dict(index=index, time_s=float(time_s), phase=phase, file=frame_file)
        self.rows.append(row)
        for name, rgb in zip(self.camera_names, images):
            self.writers[name].write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))
        # External view at left; the two square finger views are letterboxed.
        canvas = np.full((480, 1280, 3), 20, dtype=np.uint8)
        canvas[:, :640] = self.cv2.resize(external, (640, 480))
        for rgb, x in ((left, 640), (right, 960)):
            canvas[80:400, x:x + 320] = self.cv2.resize(rgb, (320, 320))
        for title, x in (("External RGB", 12), ("Left gripper RGB", 652), ("Right gripper RGB", 972)):
            self.cv2.putText(canvas, title, (x, 30), self.cv2.FONT_HERSHEY_SIMPLEX,
                             .65, (255, 255, 255), 1, self.cv2.LINE_AA)
        label = f"{time_s:.2f} s   {phase}"
        self.cv2.putText(canvas, label, (12, 460), self.cv2.FONT_HERSHEY_SIMPLEX,
                         .65, (255, 255, 255), 1, self.cv2.LINE_AA)
        self.writers["episode"].write(self.cv2.cvtColor(canvas, self.cv2.COLOR_RGB2BGR))

    def close(self, *, complete=False, result=None):
        if self.closed:
            return
        self.closed = True
        for writer in self.writers.values():
            writer.release()
        if complete and self.rows:
            for name in self.writers:
                (self.directory / (name + ".partial.mp4")).rename(self.directory / (name + ".mp4"))
        self._metadata("complete" if complete and self.rows else "incomplete", result)
