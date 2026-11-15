"""Convert a staged capture into the official dataset; safe to re-run.

Runs only in the dataset environment. The target episode_index and a pending
conversion journal are persisted before ``save_episode()`` runs, and the staged
stream is removed only after full dataset/MP4 verification confirms readability.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from soft_sim.config import ROOT
from soft_sim.contracts import ContractError, validate_capture_metadata
from soft_sim.recording.lerobot_io import LeRobotRecorder
from soft_sim.storage import ensure_formal_datasets_ready, formal_datasets_lock, formal_datasets_root
from soft_sim.validation.dataset import validate_dataset


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def _resolve_layout(scene):
    """Snapshot the task layout; tolerate captures made under an older/remote root."""
    layout = Path(scene["layout"])
    if not layout.is_file():
        parts = layout.parts
        if "third_party" in parts:
            layout = ROOT.joinpath(*parts[parts.index("third_party"):])
        elif "RoboDojo" in parts:
            layout = ROOT / "third_party" / Path(*parts[parts.index("RoboDojo"):])
    if not layout.is_file():
        raise ContractError(f"Required task layout file is missing for provenance check: {layout}")
    raw = layout.read_bytes()
    if hashlib.sha256(raw).hexdigest() != scene["layout_sha256"]:
        raise ContractError("Task layout changed after capture; refusing incorrect provenance")
    scene["layout_snapshot"] = json.loads(raw)


def _required_features_for(metadata):
    features = ["observation.images.left_rgb", "observation.images.right_rgb", "observation.state"]
    if "external_rgb" in (metadata.get("camera_order") or []):
        features.append("observation.images.external_rgb")
    is_formal = metadata.get("success_only", True) is True
    if is_formal or "commanded_joint_positions" in (metadata.get("state_keys") or []):
        features.append("action")
    return features


def _verify_written_episode(root, episode_index, metadata):
    """Verify the written episode's parquet state/action and MP4 shards."""
    report = validate_dataset(
        root,
        episode_index=int(episode_index),
        required_features=_required_features_for(metadata),
        require_success=False,
        check_video_decode=True,
    )
    if not report["valid"]:
        raise ContractError(
            f"Committed dataset verification failed for episode {episode_index}: "
            + "; ".join(report["errors"])
        )
    ep_report = report["episodes"][0]
    expected_frames = len(metadata["frames"])
    if ep_report["length"] != expected_frames:
        raise ContractError(
            f"Episode {episode_index} length mismatch: dataset has {ep_report['length']}, "
            f"capture had {expected_frames}"
        )
    return report


def committed_episode(directory, metadata):
    """Return the existing episode reference when a retry finds a verified commit."""
    if metadata.get("status") != "committed":
        return None
    root = (directory / metadata["dataset_root"]).resolve()
    info = root / "meta/info.json"
    if not info.is_file() or metadata.get("episode_index") is None:
        return None
    ep_idx = int(metadata["episode_index"])
    if ep_idx >= int(_read(info).get("total_episodes", 0)):
        return None
    _verify_written_episode(root, ep_idx, metadata)
    return metadata


def _pending_marker_path(root, ep_idx):
    return root / "meta/simulation" / f".pending_episode_{int(ep_idx):06d}.json"


def _episode_already_written(directory, root, ep_idx, metadata):
    """Check whether a crashed finalize already wrote this exact episode to disk."""
    info = root / "meta/info.json"
    if not info.is_file():
        return False
    if ep_idx >= int(_read(info).get("total_episodes", 0)):
        return False

    provenance_file = root / "meta/simulation" / f"episode_{ep_idx:06d}.json"
    pending_marker = _pending_marker_path(root, ep_idx)
    run_id = metadata.get("run_id")

    if provenance_file.is_file():
        existing = _read(provenance_file)
        if not run_id or existing.get("run_id") != run_id:
            raise ContractError(
                f"Episode {ep_idx} belongs to run_id={existing.get('run_id')!r}, not {run_id!r}"
            )
    elif pending_marker.is_file():
        pending = _read(pending_marker)
        if run_id:
            if pending.get("run_id") != run_id or pending.get("frame_count") != len(metadata["frames"]):
                raise ContractError(
                    f"Pending episode {ep_idx} belongs to run_id={pending.get('run_id')!r} "
                    f"(frames={pending.get('frame_count')}), not {run_id!r}"
                )
        elif pending.get("capture_dir") != str(directory):
            raise ContractError(
                f"Pending episode {ep_idx} belongs to {pending.get('capture_dir')!r}, "
                f"not {str(directory)!r}"
            )
    else:
        raise ContractError(
            f"Episode {ep_idx} exists in dataset without provenance or pending journal for {str(directory)!r}"
        )

    _verify_written_episode(root, ep_idx, metadata)
    return True


def _is_formal_dataset_target(directory, metadata, dataset_root):
    raw_target = Path(os.path.normpath(str(directory / metadata["dataset_root"])))
    formal_root = formal_datasets_root()
    resolved_formal = formal_root.resolve(strict=False)
    return (
        raw_target == formal_root
        or formal_root in raw_target.parents
        or dataset_root == resolved_formal
        or resolved_formal in dataset_root.parents
    )


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
    raw_dataset_root = Path(os.path.normpath(str(directory / metadata["dataset_root"])))
    dataset_root = raw_dataset_root.resolve(strict=False)

    if _is_formal_dataset_target(directory, metadata, dataset_root):
        with formal_datasets_lock():
            dataset_root = raw_dataset_root.resolve(strict=False)
            ensure_formal_datasets_ready(dataset_root)
            return _finalize_locked(directory, metadata, dataset_root, stream_path)

    if dataset_root.is_symlink() and (not dataset_root.exists() or not dataset_root.is_dir()):
        raise RuntimeError(f"Dataset path {dataset_root} is a dangling symlink.")
    return _finalize_locked(directory, metadata, dataset_root, stream_path)


def _finalize_locked(directory, metadata, dataset_root, stream_path):

    written_index = metadata.get("episode_index")
    already_written = (
        written_index is not None
        and _episode_already_written(directory, dataset_root, int(written_index), metadata)
    )

    if not already_written:
        if not stream_path.is_file():
            raise ContractError("Staged capture stream is missing; the capture cannot be committed")
        if written_index is not None:
            _pending_marker_path(dataset_root, int(written_index)).unlink(missing_ok=True)
        metadata = dict(metadata, status="converting", episode_index=None)
        metadata.pop("finalize_error", None)
        _write(directory / "episode.json", metadata)

        recorder = LeRobotRecorder(
            directory, metadata["fps"], metadata["calibration"],
            dataset_root=dataset_root, repo_id=metadata["repo_id"],
            task=metadata["task"], write_metadata=False,
        )
        pending_marker = None
        try:
            with stream_path.open("rb") as stream:
                for row in metadata["frames"]:
                    images = [np.load(stream, allow_pickle=False) for _ in range(3)]
                    state = {key: np.load(stream, allow_pickle=False) for key in metadata["state_keys"]}
                    recorder.append(*images, time_s=row["time_s"], phase=row["phase"], **state)
                if stream.read(1):
                    raise ContractError("Unexpected trailing capture data")
            if recorder.episode_index is None:
                raise ContractError("Official writer did not assign an episode_index")

            # Persist the reserved episode_index and capture journal BEFORE save_episode()
            # runs, closing the pre-index crash window.
            reserved_index = int(recorder.episode_index)
            pending_marker = _pending_marker_path(dataset_root, reserved_index)
            _write(pending_marker, dict(
                episode_index=reserved_index,
                run_id=metadata.get("run_id"),
                capture_dir=str(directory),
                frame_count=len(metadata["frames"]),
            ))
            metadata = dict(metadata, episode_index=reserved_index)
            _write(directory / "episode.json", metadata)

            saved = recorder.close(complete=True, result=metadata["result"])
            if not saved or recorder.episode_index is None:
                raise ContractError("Official writer did not save the episode")
            written_index = int(recorder.episode_index)
            if written_index != reserved_index:
                raise ContractError(
                    f"Reserved episode_index {reserved_index} != written episode_index {written_index}"
                )
            _verify_written_episode(dataset_root, written_index, metadata)
        except BaseException as error:
            recorder.close()
            _write(
                directory / "episode.json",
                dict(metadata, status="failed", finalize_error=f"{type(error).__name__}: {error}"),
            )
            raise
    else:
        written_index = int(written_index)

    provenance = dataset_root / "meta/simulation"
    provenance.mkdir(parents=True, exist_ok=True)
    saved_metadata = dict(metadata, status="committed", episode_index=written_index)
    saved_metadata.pop("finalize_error", None)
    stem = f"episode_{written_index:06d}"
    reference = directory / "finger_reference.npz"
    if reference.is_file():
        saved_metadata["reference_file"] = stem + "_reference.npz"
        shutil.copyfile(reference, provenance / saved_metadata["reference_file"])
    _write(directory / "episode.json", saved_metadata)
    _write(provenance / (stem + ".json"), saved_metadata)
    _update_summary(provenance)
    _pending_marker_path(dataset_root, written_index).unlink(missing_ok=True)
    stream_path.unlink(missing_ok=True)
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
            active_arm=(episode.get("calibration") or {}).get("active_arm"),
            success=(episode.get("result") or {}).get("success"))
    episodes = sorted(entries.values(), key=lambda entry: entry["episode_index"])
    summary.update(codebase_version="v3.0", episodes=episodes,
        successful_episode_indices=[entry["episode_index"] for entry in episodes if entry["success"] is True],
        failed_episode_indices=[entry["episode_index"] for entry in episodes if entry["success"] is False])
    _write(summary_path, summary)
    return summary
