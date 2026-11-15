"""Convert a staged capture into the official dataset; safe to re-run.

Runs only in the dataset environment. The staged stream is removed after the
official writer has saved and reopened the episode, so a crash keeps the data.
"""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from soft_sim.config import ROOT
from soft_sim.contracts import ContractError, validate_capture_metadata
from soft_sim.recording.lerobot_io import LeRobotRecorder


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, data):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def _resolve_layout(scene):
    """Snapshot the task layout; tolerate captures made under an older root."""
    layout = Path(scene["layout"])
    if not layout.is_file() and "RoboDojo" in layout.parts:
        parts = layout.parts
        tail = Path(*parts[len(parts) - parts[::-1].index("RoboDojo"):])
        layout = ROOT / "third_party" / tail
    if not layout.is_file():
        return
    raw = layout.read_bytes()
    if hashlib.sha256(raw).hexdigest() != scene["layout_sha256"]:
        raise ContractError("Task layout changed after capture; refusing incorrect provenance")
    scene["layout_snapshot"] = json.loads(raw)


def committed_episode(directory, metadata):
    """Return the existing episode reference when a retry finds a finished commit."""
    if metadata.get("status") != "committed":
        return None
    root = (directory / metadata["dataset_root"]).resolve()
    info = root / "meta/info.json"
    if not info.is_file() or metadata.get("episode_index") is None:
        return None
    if int(metadata["episode_index"]) >= int(_read(info).get("total_episodes", 0)):
        return None
    return metadata


def finalize(directory):
    directory = Path(directory).resolve()
    metadata = validate_capture_metadata(_read(directory / "episode.json"))
    if metadata.get("success_only") and (metadata.get("result") or {}).get("success") is not True:
        raise ContractError("Success-only capture requires a successful task result; refusing commit")
    previous = committed_episode(directory, metadata)
    if previous is not None:
        return previous
    if metadata["status"] not in ("staged", "failed", "converting"):
        raise ContractError("Only complete staged captures can be finalized")

    scene = metadata.get("calibration", {}).get("task_scene")
    if scene and scene.get("layout"):
        _resolve_layout(scene)
    stream_path = directory / "capture.partial.npy"
    if not stream_path.is_file():
        raise ContractError("Staged capture stream is missing; the capture cannot be committed")

    recorder = LeRobotRecorder(directory, metadata["fps"], metadata["calibration"],
        dataset_root=(directory / metadata["dataset_root"]).resolve(),
        repo_id=metadata["repo_id"], task=metadata["task"])
    try:
        with stream_path.open("rb") as stream:
            for row in metadata["frames"]:
                images = [np.load(stream, allow_pickle=False) for _ in range(3)]
                state = {key: np.load(stream, allow_pickle=False) for key in metadata["state_keys"]}
                recorder.append(*images, time_s=row["time_s"], phase=row["phase"], **state)
            if stream.read(1):
                raise ContractError("Unexpected trailing capture data")
        saved = recorder.close(complete=True, result=metadata["result"])
        if not saved:
            raise ContractError("Official writer did not save the episode")
    except BaseException as error:
        recorder.close()
        _write(directory / "episode.json", dict(metadata, status="failed",
                                                finalize_error=f"{type(error).__name__}: {error}"))
        raise

    # Confirm by reopening the written episode before deleting the staged stream.
    info = _read(recorder.root / "meta/info.json")
    if int(recorder.episode_index) >= int(info.get("total_episodes", 0)):
        raise ContractError("Saved episode is not present in dataset metadata")

    provenance = recorder.root / "meta/simulation"
    provenance.mkdir(parents=True, exist_ok=True)
    written = _read(directory / "episode.json")
    # Keep the capture protocol fields; the dataset writer only owns the frame
    # index, calibration echo and timing it actually wrote.
    saved_metadata = dict(metadata, status="committed",
                          episode_index=recorder.episode_index,
                          frames=written.get("frames", metadata["frames"]),
                          phases=written.get("phases", metadata["phases"]),
                          duration_s=written.get("duration_s", metadata.get("duration_s")),
                          calibration=written.get("calibration", metadata["calibration"]))
    stem = f"episode_{recorder.episode_index:06d}"
    reference = directory / "finger_reference.npz"
    if reference.is_file():
        saved_metadata["reference_file"] = stem + "_reference.npz"
        shutil.copyfile(reference, provenance / saved_metadata["reference_file"])
    _write(directory / "episode.json", saved_metadata)
    _write(provenance / (stem + ".json"), saved_metadata)
    _update_summary(provenance)
    stream_path.unlink()
    return saved_metadata


def _update_summary(provenance):
    summary_path = provenance / "collection_summary.json"
    summary = _read(summary_path) if summary_path.exists() else {}
    entries = {entry["episode_index"]: entry for entry in summary.get("episodes", [])}
    for episode_path in sorted(provenance.glob("episode_*.json")):
        episode = _read(episode_path)
        index = episode["episode_index"]
        entries.setdefault(index, {}).update(
            episode_index=index, frame_count=episode["frame_count"],
            success=(episode.get("result") or {}).get("success"))
    episodes = sorted(entries.values(), key=lambda entry: entry["episode_index"])
    summary.update(codebase_version="v3.0", episodes=episodes,
        successful_episode_indices=[entry["episode_index"] for entry in episodes if entry["success"] is True],
        failed_episode_indices=[entry["episode_index"] for entry in episodes if entry["success"] is False])
    _write(summary_path, summary)
    return summary
