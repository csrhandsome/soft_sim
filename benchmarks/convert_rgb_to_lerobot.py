"""Convert complete legacy recordings into a shared LeRobot v3 dataset."""
import argparse
import json
from pathlib import Path

import numpy as np
from episode_recording import EpisodeRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path, required=True,
                        help="New diagnostic indices; existing recordings are never overwritten")
    args = parser.parse_args()
    for index, directory in enumerate(args.recordings):
        metadata = json.loads((directory / "episode.json").read_text())
        if metadata["status"] != "complete" or metadata.get("storage_format") == "lerobot_v3":
            raise ValueError(f"Expected a complete legacy recording: {directory}")
        output = args.index_root / f"episode_{index:06d}"
        recorder = EpisodeRecorder(output, metadata["fps"], metadata["calibration"],
            dataset_root=args.dataset_root, task=(metadata.get("result") or {}).get("task", "soft finger grasp"))
        import shutil
        if (directory / "finger_reference.npz").exists():
            shutil.copyfile(directory / "finger_reference.npz", output / "finger_reference.npz")
        try:
            for row in metadata["frames"]:
                with np.load(directory / row["file"], allow_pickle=False) as frame:
                    state = {key:frame[key] for key in frame.files if key not in (*recorder.camera_names,"time_s")}
                    recorder.append(*(frame[key] for key in recorder.camera_names),
                                    time_s=row["time_s"], phase=row["phase"], **state)
            recorder.close(complete=True, result=metadata.get("result"))
        finally:
            recorder.close()


if __name__ == "__main__":
    main()
