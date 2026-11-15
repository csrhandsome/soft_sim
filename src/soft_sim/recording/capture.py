"""Simulation-side capture staging: NPZ frames plus a JSON index.

This module must not import LeRobot, Torch or any dataset dependency. The
episode is written as a self-contained stream that the dataset environment
later converts; the two sides only exchange files.
"""

import json
from pathlib import Path

import numpy as np

from soft_sim.contracts import CAPTURE_PROTOCOL_VERSION, validate_capture_metadata

CAMERA_NAMES = ("left_rgb", "right_rgb", "external_rgb")


class EpisodeRecorder:
    """Stage one sequential capture file, then finalize in another process.

    NumPy 1.26 (Isaac Sim) and NumPy 2 (LeRobot) cannot share an interpreter.
    The staged stream is deleted only after the official writer reports success.
    """

    camera_names = CAMERA_NAMES

    def __init__(self, directory, fps, calibration, *, dataset_root, repo_id="local/softfinger",
                 task="soft finger grasp", run_id=None, success_only=True):
        self.directory = Path(directory).resolve()
        if self.directory.exists():
            raise FileExistsError(self.directory)
        self.directory.mkdir(parents=True)
        self.fps = float(fps)
        if not self.fps.is_integer() or self.fps <= 0:
            raise ValueError("LeRobot requires a positive integer FPS")
        self.root = Path(dataset_root).resolve()
        self.repo_id, self.task, self.calibration = repo_id, task, calibration
        self.run_id = run_id
        self.success_only = success_only
        self.rows, self.phases = [], []
        self.state_keys = None
        self.shapes = None
        self.closed = False
        self.result = None
        self.committed = False
        self.stream = (self.directory / "capture.partial.npy").open("xb")
        self._metadata("capturing")

    # -- index ------------------------------------------------------------
    def _read_metadata(self):
        return json.loads((self.directory / "episode.json").read_text())

    def _metadata(self, status, result=None):
        data = self._read_metadata() if (self.directory / "episode.json").exists() else {}
        data.update(
            storage_format="lerobot_v3", protocol_version=CAPTURE_PROTOCOL_VERSION,
            status=status, fps=self.fps,
            dataset_root=self._dataset_reference(), dataset_id=self.root.name,
            repo_id=self.repo_id, episode_index=data.get("episode_index"),
            frame_count=len(self.rows), duration_s=len(self.rows) / self.fps,
            camera_order=list(self.camera_names), color_order="RGB",
            camera_coordinates="optical: +X right, +Y down, +Z forward",
            units="SI meters, seconds, radians for joints",
            calibration=self.calibration, phases=self.phases,
            frames=self.rows, run_id=self.run_id, task=self.task,
            state_keys=self.state_keys, result=result if result is not None else data.get("result"))
        data["success_only"] = self.success_only
        path = self.directory / "episode.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(path)
        return data

    def _dataset_reference(self):
        """Portable reference: relative when both live under the same storage tree."""
        import os
        return os.path.relpath(self.root, self.directory)

    # -- frames -----------------------------------------------------------
    def append(self, left, right, external, *, time_s, phase, **state):
        if self.closed:
            raise RuntimeError("Episode recorder is already closed")
        if not np.isfinite(time_s) or (self.rows and not np.isclose(
                time_s - self.rows[-1]["time_s"], 1 / self.fps, atol=1e-7, rtol=0)):
            raise ValueError("RGB samples must have a constant simulation-time interval")
        images = [np.asarray(v) for v in (left, right, external)]
        if any(v.dtype != np.uint8 or v.ndim != 3 or v.shape[-1] != 3 or v.std() < 1 for v in images):
            raise ValueError("Invalid RGB capture")
        keys = sorted(state)
        values = images + [np.asarray(state[key]) for key in keys]
        shapes = [(v.shape, str(v.dtype)) for v in values]
        if self.state_keys is None:
            self.state_keys, self.shapes = keys, shapes
        elif keys != self.state_keys or shapes != self.shapes:
            raise ValueError("Sensor schema changed during the episode")
        for value in values:
            np.save(self.stream, value, allow_pickle=False)
        self.rows.append(dict(index=len(self.rows), time_s=float(time_s), phase=phase))
        if phase not in self.phases:
            self.phases.append(phase)
        if len(self.rows) == 1 or len(self.rows) % int(self.fps) == 0:
            self.stream.flush()
            self._metadata("capturing")

    # -- completion -------------------------------------------------------
    def close(self, *, complete=False, result=None):
        if self.closed:
            return
        self.closed = True
        self.result = result
        self.stream.close()
        if complete and self.success_only and (result or {}).get("success") is not True:
            self._metadata("discarded", result)
            (self.directory / "capture.partial.npy").unlink(missing_ok=True)
            (self.directory / "finger_reference.npz").unlink(missing_ok=True)
            return
        self._metadata("staged" if complete and self.rows else "incomplete", result)
        validate_capture_metadata(self._read_metadata())
        if complete and self.rows:
            self.finalize()

    def finalize(self):
        """Convert the staged stream in the dataset environment."""
        from soft_sim.runtime import run
        process = run("dataset", "soft_sim.cli.finalize", self.directory,
                      log_path=self.directory / "finalize.log")
        process.check()
        metadata = self._read_metadata()
        if metadata.get("status") != "committed":
            raise RuntimeError(f"Finalize did not commit the capture: {metadata.get('status')}")
        self.committed = True
        return metadata
