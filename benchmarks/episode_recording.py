"""LeRobot v3 camera videos and typed Parquet state; never uploads data."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

os.environ.setdefault("HF_DATASETS_CACHE", str(Path(tempfile.gettempdir()) / "soft-sim-hf-datasets"))


class LeRobotRecorder:
    camera_names = ("left_rgb", "right_rgb", "external_rgb")

    def __init__(self, directory, fps, calibration, *, dataset_root=None,
                 repo_id="local/softfinger75a", task="soft finger grasp"):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.root = Path(dataset_root).resolve() if dataset_root else self.directory / "dataset"
        self.fps = float(fps)
        if not self.fps.is_integer() or self.fps <= 0:
            raise ValueError("LeRobot requires a positive integer FPS")
        self.repo_id, self.task, self.calibration = repo_id, task, calibration
        self.rows, self.phases = [], []
        self.dataset = None
        self.closed = False
        self.episode_index = None
        self.schema = None
        self._metadata("incomplete")

    def _metadata(self, status, result=None):
        data = dict(storage_format="lerobot_v3", status=status, fps=self.fps,
                    dataset_root=os.path.relpath(self.root, self.directory),
                    repo_id=self.repo_id, episode_index=self.episode_index,
                    frame_count=len(self.rows), duration_s=len(self.rows) / self.fps,
                    camera_order=list(self.camera_names), color_order="RGB",
                    camera_coordinates="optical: +X right, +Y down, +Z forward",
                    calibration=self.calibration, phases=self.phases,
                    frames=self.rows, result=result)
        path = self.directory / "episode.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(path)

    def append(self, left, right, external, *, time_s, phase, **state):
        if self.closed:
            raise RuntimeError("Episode recorder is already closed")
        if not np.isfinite(time_s) or (self.rows and not np.isclose(
                time_s - self.rows[-1]["time_s"], 1 / self.fps, atol=1e-7, rtol=0)):
            raise ValueError("RGB samples must have a constant simulation-time interval")
        frame = {}
        for name, value in zip(self.camera_names, (left, right, external)):
            value = np.asarray(value)
            if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8 or value.std() < 1:
                raise ValueError(f"Invalid or uniform RGB frame: {name}")
            frame["observation.images." + name] = value.copy()
        if phase not in self.phases:
            self.phases.append(phase)
        frame["observation.phase"] = np.array([self.phases.index(phase)], dtype=np.int64)
        frame["observation.sim_time_s"] = np.array([time_s], dtype=np.float32)
        for name, value in state.items():
            key = {"joint_positions": "observation.state",
                   "commanded_joint_positions": "action"}.get(name, "observation." + name)
            value = np.atleast_1d(np.asarray(value, dtype=np.int64 if name == "operation_index" else np.float32))
            if name == "finger_depth_m":
                # LeRobot statistics require finite arrays. Keep a validity mask
                # and reconstruct NaN for no-hit rays when reading diagnostics.
                valid = np.isfinite(value) & (value > 0)
                frame["observation.finger_depth_valid"] = valid.astype(np.uint8)
                value = np.where(valid, value, 0).astype(np.float32)
            elif not np.isfinite(value).all():
                raise ValueError(f"Non-finite state: {name}")
            frame[key] = value.copy()
        features = {key: dict(dtype="video" if key.startswith("observation.images.") else str(value.dtype),
                              shape=value.shape,
                              names=["height", "width", "channels"] if key.startswith("observation.images.") else None)
                    for key, value in frame.items()}
        if self.dataset is None:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            if (self.root / "meta/info.json").exists():
                self.dataset = LeRobotDataset(self.repo_id, root=self.root, video_backend="pyav",
                                              vcodec="h264", encoder_threads=2)
                if self.dataset.meta.info["codebase_version"] != "v3.0" or self.dataset.fps != int(self.fps):
                    raise ValueError("Existing dataset must be v3.0 with matching FPS")
                existing = {k: v for k, v in self.dataset.features.items() if k in features}
                if set(existing) != set(features) or any(
                        tuple(existing[k]["shape"]) != tuple(v["shape"]) or existing[k]["dtype"] != v["dtype"]
                        for k, v in features.items()):
                    raise ValueError("Dataset schema changed; use a separate dataset for this mesh/task")
                self.dataset.start_image_writer(num_processes=0, num_threads=4)
            else:
                self.dataset = LeRobotDataset.create(
                    repo_id=self.repo_id, root=self.root, fps=int(self.fps),
                    robot_type=self.calibration.get("robot_type", "franka_softfinger"), features=features,
                    use_videos=True, image_writer_threads=4, video_backend="pyav",
                    vcodec="h264", encoder_threads=2)
            self.schema = features
            self.episode_index = self.dataset.meta.total_episodes
            self._metadata("incomplete")
        elif features != self.schema:
            raise ValueError("Sensor schema changed during the episode")
        self.dataset.add_frame({**frame, "task": self.task})
        self.rows.append(dict(index=len(self.rows), time_s=float(time_s), phase=phase))

    def close(self, *, complete=False, result=None):
        if self.closed:
            return
        self.closed = True
        saved = False
        try:
            if self.dataset is not None:
                if complete and self.rows:
                    self.dataset.save_episode()
                    saved = True
                else:
                    self.dataset.clear_episode_buffer()
                self.dataset.finalize()
        finally:
            self._metadata("complete" if saved else "incomplete", result)


class EpisodeRecorder(LeRobotRecorder):
    """Stage one sequential file in Isaac, then finalize in the data environment.

    NumPy 1.26 (Isaac) and NumPy 2 (LeRobot/Rerun) cannot share an interpreter.
    The temporary stream is removed only after the official writer succeeds.
    """
    def __init__(self, directory, fps, calibration, **kwargs):
        if Path(directory).exists():
            raise FileExistsError(directory)
        super().__init__(directory, fps, calibration, **kwargs)
        self.state_keys = None
        self.shapes = None
        self.stream = (self.directory / "capture.partial.npy").open("xb")

    def append(self, left, right, external, *, time_s, phase, **state):
        if self.closed:
            raise RuntimeError("Episode recorder is already closed")
        if not np.isfinite(time_s) or (self.rows and not np.isclose(time_s - self.rows[-1]["time_s"], 1 / self.fps, atol=1e-7, rtol=0)):
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
            self._metadata("incomplete")
            path = self.directory / "episode.json"
            metadata = json.loads(path.read_text())
            metadata.update(state_keys=self.state_keys, task=self.task)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, indent=2))
            temporary.replace(path)

    def close(self, *, complete=False, result=None):
        if self.closed:
            return
        self.closed = True
        self.stream.close()
        self._metadata("staged" if complete and self.rows else "incomplete", result)
        path = self.directory / "episode.json"
        metadata = json.loads(path.read_text())
        metadata.update(state_keys=self.state_keys, task=self.task)
        path.write_text(json.dumps(metadata, indent=2))
        if complete and self.rows:
            root = Path(__file__).resolve().parents[1]
            subprocess.run([str(root / "envs/dataset/.venv/bin/python"),
                            str(root / "benchmarks/finalize_lerobot.py"), str(self.directory)], check=True)
