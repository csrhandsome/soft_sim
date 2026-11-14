"""Finalize the single-file Isaac capture using the isolated LeRobot environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from episode_recording import LeRobotRecorder


def finalize(directory):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "episode.json").read_text())
    if metadata["status"] != "staged":
        raise ValueError("Only complete staged captures can be finalized")
    scene = metadata.get("calibration", {}).get("task_scene")
    if scene and scene.get("layout"):
        layout = Path(scene["layout"])
        if layout.is_file():
            raw = layout.read_bytes()
            if hashlib.sha256(raw).hexdigest() != scene["layout_sha256"]:
                raise ValueError("Task layout changed after capture; refusing incorrect provenance")
            scene["layout_snapshot"] = json.loads(raw)
    stream_path = directory / "capture.partial.npy"
    recorder = LeRobotRecorder(directory, metadata["fps"], metadata["calibration"],
        dataset_root=directory / metadata["dataset_root"],
        repo_id=metadata["repo_id"], task=metadata["task"])
    try:
        with stream_path.open("rb") as stream:
            for row in metadata["frames"]:
                images = [np.load(stream, allow_pickle=False) for _ in range(3)]
                state = {key: np.load(stream, allow_pickle=False) for key in metadata["state_keys"]}
                recorder.append(*images, time_s=row["time_s"], phase=row["phase"], **state)
            if stream.read(1):
                raise ValueError("Unexpected trailing capture data")
        recorder.close(complete=True, result=metadata["result"])
    except BaseException:
        recorder.close()
        # Keep the original stream and index for explicit recovery.
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2))
        raise
    provenance = recorder.root / "meta/simulation"
    provenance.mkdir(parents=True, exist_ok=True)
    saved = json.loads((directory / "episode.json").read_text())
    saved["dataset_root"] = "../.."
    stem = f"episode_{recorder.episode_index:06d}"
    reference = directory / "finger_reference.npz"
    if reference.is_file():
        saved["reference_file"] = stem + "_reference.npz"
        shutil.copyfile(reference, provenance / saved["reference_file"])
    (provenance / (stem + ".json")).write_text(json.dumps(saved, indent=2))
    summary_path = provenance / "collection_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    entries = {entry["episode_index"]: entry for entry in summary.get("episodes", [])}
    for episode_path in sorted(provenance.glob("episode_*.json")):
        episode = json.loads(episode_path.read_text())
        index = episode["episode_index"]
        entry = entries.setdefault(index, {})
        entry.update(episode_index=index, frame_count=episode["frame_count"],
                     success=(episode.get("result") or {}).get("success"))
    episodes = sorted(entries.values(), key=lambda entry: entry["episode_index"])
    summary.update(codebase_version="v3.0", episodes=episodes,
        successful_episode_indices=[entry["episode_index"] for entry in episodes if entry["success"] is True],
        failed_episode_indices=[entry["episode_index"] for entry in episodes if entry["success"] is False])
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(summary_path)
    stream_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    finalize(parser.parse_args().directory)
