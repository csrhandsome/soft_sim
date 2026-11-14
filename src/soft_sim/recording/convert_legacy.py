"""Convert complete legacy NPZ recordings into a shared LeRobot v3 dataset.

Runs in the dataset environment and writes only through the official writer.
Existing recordings are never overwritten; the new diagnostic indices live in
``--index-root``.
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from soft_sim.recording.lerobot_io import LeRobotRecorder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True,
                        help="New diagnostic indices; existing recordings are never overwritten")
    args = parser.parse_args(argv)
    for index, directory in enumerate(args.recordings):
        metadata = json.loads((directory / "episode.json").read_text())
        if metadata["status"] != "complete" or metadata.get("storage_format") == "lerobot_v3":
            raise ValueError(f"Expected a complete legacy recording: {directory}")
        output = args.index_root / f"episode_{index:06d}"
        recorder = LeRobotRecorder(output, metadata["fps"], metadata["calibration"],
            dataset_root=args.dataset_root,
            task=(metadata.get("result") or {}).get("task", "soft finger grasp"))
        if (directory / "finger_reference.npz").exists():
            shutil.copyfile(directory / "finger_reference.npz", output / "finger_reference.npz")
        try:
            for row in metadata["frames"]:
                with np.load(directory / row["file"], allow_pickle=False) as frame:
                    state = {key: frame[key] for key in frame.files
                             if key not in (*recorder.camera_names, "time_s")}
                    recorder.append(*(frame[key] for key in recorder.camera_names),
                                    time_s=row["time_s"], phase=row["phase"], **state)
            recorder.close(complete=True, result=metadata.get("result"))
        finally:
            recorder.close()


if __name__ == "__main__":
    raise SystemExit(main())
