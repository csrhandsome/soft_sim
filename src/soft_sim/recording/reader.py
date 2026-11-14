"""Read diagnostic frames from LeRobot v3, with legacy NPZ support."""
from contextlib import nullcontext
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault("HF_DATASETS_CACHE", str(Path(tempfile.gettempdir()) / "soft-sim-hf-datasets"))


@lru_cache(maxsize=2)
def _dataset(directory):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    folder = Path(directory)
    metadata = json.loads((folder / "episode.json").read_text())
    dataset = LeRobotDataset(metadata["repo_id"], root=(folder / metadata["dataset_root"]).resolve(),
                            video_backend="pyav")
    return dataset, metadata


def open_frame(directory, row, *, include_images=True):
    directory = Path(directory)
    if "file" in row:
        return np.load(directory / row["file"], allow_pickle=False)
    dataset, metadata = _dataset(str(directory.resolve()))
    episode = dataset.meta.episodes[metadata["episode_index"]]
    index = int(episode["dataset_from_index"]) + row["index"]
    # Physics/depth audits need the exact recorded arrays, without decoding
    # three unrelated MP4 frames for every ray query. RGB audits use default.
    item = dataset[index] if include_images else dataset.hf_dataset[index]
    frame = {"time_s": np.array(row["time_s"])}
    for key, value in item.items():
        if key.startswith("observation.images."):
            if not include_images:
                continue
            image = value.numpy().transpose(1, 2, 0)
            frame[key.removeprefix("observation.images.")] = np.rint(image * 255).astype(np.uint8)
        elif key in ("observation.state", "action"):
            frame["joint_positions" if key == "observation.state" else "commanded_joint_positions"] = value.numpy()
        elif key.startswith("observation."):
            frame[key.removeprefix("observation.")] = value.numpy()
    if "finger_depth_valid" in frame:
        frame["finger_depth_m"] = np.where(frame["finger_depth_valid"], frame["finger_depth_m"], np.nan)
    return nullcontext(frame)


# -- run layout helpers ---------------------------------------------------
def capture_directory(path):
    """Accept a run root, ``run/capture`` or a legacy ``run/recording`` path."""
    path = Path(path)
    for candidate in (path, path / "capture", path / "recording"):
        if (candidate / "episode.json").is_file():
            return candidate
    raise FileNotFoundError(f"No capture index (episode.json) under {path}")


def run_directory(path):
    path = Path(path)
    return path.parent if path.name in ("capture", "recording") else path


def first_existing(*paths):
    for path in paths:
        if Path(path).is_file():
            return Path(path)
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")
