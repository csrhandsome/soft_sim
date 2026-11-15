"""LeRobot v3 dataset delivery validation (metadata, state/action, and MP4 shards).

Runs in the ``dataset`` environment. Decodes each unique MP4 shard at most once
and reads only lightweight parquet columns so large FEM node arrays do not stall
verification.
"""

import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from soft_sim.config import ROOT

DEFAULT_REQUIRED_FEATURES = (
    "observation.images.left_rgb",
    "observation.images.right_rgb",
    "observation.images.external_rgb",
    "observation.state",
    "action",
)
OPTIONAL_FEM_FEATURES = (
    "observation.finger_nodes_left",
    "observation.finger_nodes_right",
    "observation.finger_depth_m",
)
LIGHTWEIGHT_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.sim_time_s",
    "observation.state",
    "action",
    "observation.object_position",
    "observation.task_object_positions",
)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _infer_repo_id(root):
    sim_dir = root / "meta/simulation"
    if sim_dir.is_dir():
        for ep_file in sorted(sim_dir.glob("episode_*.json")):
            try:
                data = _read_json(ep_file)
                if data.get("repo_id"):
                    return data["repo_id"]
            except Exception:
                continue
    return "local/softfinger"


def _load_provenance(root):
    sim_dir = root / "meta/simulation"
    episodes = {}
    if sim_dir.is_dir():
        for ep_file in sorted(sim_dir.glob("episode_*.json")):
            try:
                data = _read_json(ep_file)
            except Exception:
                continue
            idx = data.get("episode_index")
            if idx is not None:
                episodes[int(idx)] = data
    summary_path = sim_dir / "collection_summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else None
    return dict(present=bool(episodes or summary), episodes=episodes, summary=summary)


def _lookup_run_quality(prov):
    """Read-only lookup of independent acceptance.json via run_id provenance."""
    if prov is None:
        return None, "no_provenance", None
    if "quality_valid" in prov and prov["quality_valid"] is not None:
        return prov["quality_valid"], "dataset_provenance", None
    run_id = prov.get("run_id")
    if not run_id:
        return None, "no_run_id", None
    reports_dir = ROOT / "artifacts/runs" / run_id / "reports"
    if not reports_dir.is_dir():
        return None, "no_run_report", None
    candidates = sorted(reports_dir.glob("*/acceptance.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None, "no_run_report", None
    chosen = candidates[-1]
    try:
        data = _read_json(chosen)
        return data.get("quality_valid"), "run_acceptance_report", str(chosen)
    except Exception:
        return None, "unreadable_run_report", str(chosen)


def _decode_video_shard(path, expected_shape, expected_fps, min_rgb_std=1.0):
    """Decode one MP4 shard once and return strictly monotonic frame PTS and any error."""
    if not path.is_file() or path.stat().st_size == 0:
        return dict(ok=False, path=str(path), error=f"Missing or empty MP4 shard: {path}", timestamps=[])
    expected_h, expected_w = int(expected_shape[0]), int(expected_shape[1])
    timestamps = []
    min_std = float("inf")
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return dict(ok=False, path=str(path), error=f"No video stream in {path}", timestamps=[])
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (expected_w, expected_h):
                return dict(
                    ok=False, path=str(path), timestamps=[],
                    error=f"Resolution mismatch in {path}: {(stream.width, stream.height)} != {(expected_w, expected_h)}",
                )
            rate = float(stream.average_rate) if stream.average_rate is not None else 0.0
            if abs(rate - expected_fps) > 1e-3:
                return dict(
                    ok=False, path=str(path), timestamps=[],
                    error=f"FPS mismatch in {path}: {rate} != {expected_fps}",
                )
            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame.pts is None or frame.time_base is None:
                    return dict(
                        ok=False, path=str(path), timestamps=[],
                        error=f"Missing PTS or time_base on frame {frame_idx} in {path}",
                    )
                pts_s = float(frame.pts * frame.time_base)
                if timestamps and pts_s <= timestamps[-1]:
                    return dict(
                        ok=False, path=str(path), timestamps=[],
                        error=f"Duplicate or non-monotonic PTS in {path} at frame {frame_idx}: {pts_s} <= {timestamps[-1]}",
                    )
                arr = frame.to_ndarray(format="rgb24")
                if arr.shape != (expected_h, expected_w, 3):
                    return dict(
                        ok=False, path=str(path), timestamps=[],
                        error=f"Decoded frame shape mismatch in {path}: {arr.shape}",
                    )
                min_std = min(min_std, float(arr.std()))
                timestamps.append(pts_s)
    except Exception as exc:
        return dict(ok=False, path=str(path), error=f"Failed decoding {path}: {type(exc).__name__}: {exc}", timestamps=[])

    if not timestamps:
        return dict(ok=False, path=str(path), error=f"Decoded 0 frames from {path}", timestamps=[])
    if min_rgb_std > 0 and min_std < min_rgb_std:
        return dict(
            ok=False, path=str(path), timestamps=timestamps, min_rgb_std=min_std,
            error=f"Uniform/blank frame detected in {path} (min std={min_std:.3f} < {min_rgb_std})",
        )
    return dict(ok=True, path=str(path), frame_count=len(timestamps), fps=rate,
                resolution=[expected_w, expected_h], min_rgb_std=min_std, timestamps=timestamps)


def _check_parquet_shard(path, required_columns):
    """Read only lightweight columns from one parquet data shard."""
    if not path.is_file() or path.stat().st_size == 0:
        return dict(ok=False, error=f"Missing or empty parquet shard: {path}")
    try:
        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)
        missing_cols = [col for col in required_columns if col not in schema_names]
        if missing_cols:
            return dict(ok=False, error=f"Parquet {path} missing columns: {missing_cols}")
        cols_to_read = [col for col in LIGHTWEIGHT_COLUMNS if col in schema_names]
        table = pq.read_table(path, columns=cols_to_read)
        data = {col: table[col].to_pylist() for col in cols_to_read}
        return dict(ok=True, schema_names=sorted(schema_names), data=data)
    except Exception as exc:
        return dict(ok=False, error=f"Failed reading parquet {path}: {type(exc).__name__}: {exc}")


def _check_column_shape_and_finite(values, expected_shape, name):
    try:
        arr = np.asarray(values, dtype=np.float64)
    except Exception as exc:
        return f"Non-numeric or ragged values in {name}: {exc}"
    norm_expected = tuple(int(d) for d in expected_shape)
    target_shape = (len(values),) + norm_expected
    if arr.shape != target_shape and not (norm_expected == (1,) and arr.shape == (len(values),)):
        return f"Shape mismatch in {name}: {arr.shape} != {target_shape}"
    if arr.size == 0 or not np.isfinite(arr).all():
        return f"Non-finite or empty values in {name}"
    return None


def validate_dataset(
    dataset_root,
    *,
    episode_index=None,
    required_features=None,
    require_external_rgb=True,
    require_fem=False,
    require_success=True,
    check_video_decode=True,
    min_rgb_std=1.0,
):
    """Validate a LeRobot v3 dataset directory and return a structured JSON-ready report."""
    root = Path(dataset_root).resolve()
    errors = []
    report = dict(
        valid=False,
        dataset_root=str(root),
        codebase_version=None,
        fps=None,
        total_episodes=0,
        total_frames=0,
        checked_episodes=0,
        features_present=[],
        missing_features=[],
        fem_features_present=[],
        fem_check_mode="schema_only" if require_fem else "not_required",
        provenance_present=False,
        video_shards={},
        episodes=[],
        errors=errors,
    )

    info_path = root / "meta/info.json"
    if not root.is_dir():
        errors.append(f"Dataset root does not exist: {root}")
        return report
    if not info_path.is_file():
        errors.append(f"Missing meta/info.json under {root}")
        return report

    try:
        info = _read_json(info_path)
    except Exception as exc:
        errors.append(f"Invalid meta/info.json: {exc}")
        return report

    report["codebase_version"] = info.get("codebase_version")
    report["fps"] = info.get("fps")
    report["total_episodes"] = int(info.get("total_episodes", 0))
    report["total_frames"] = int(info.get("total_frames", 0))

    if info.get("codebase_version") != "v3.0":
        errors.append(f"Expected codebase_version 'v3.0', got {info.get('codebase_version')!r}")
    fps = info.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        errors.append(f"Invalid fps in meta/info.json: {fps!r}")
        return report
    fps = float(fps)
    if report["total_episodes"] <= 0 or report["total_frames"] <= 0:
        errors.append(
            f"Dataset has no episodes/frames (total_episodes={report['total_episodes']}, total_frames={report['total_frames']})"
        )
        return report

    features = info.get("features") or {}
    report["features_present"] = sorted(features.keys())
    report["fem_features_present"] = [f for f in OPTIONAL_FEM_FEATURES if f in features]

    if required_features is None:
        req = list(DEFAULT_REQUIRED_FEATURES)
        if not require_external_rgb and "observation.images.external_rgb" in req:
            req.remove("observation.images.external_rgb")
    else:
        req = list(required_features)
    if require_fem:
        for fem_key in ("observation.finger_nodes_left", "observation.finger_nodes_right"):
            if fem_key not in req:
                req.append(fem_key)

    missing_features = [f for f in req if f not in features]
    report["missing_features"] = missing_features
    if missing_features:
        errors.append(f"Missing required features in meta/info.json: {missing_features}")

    video_features = [f for f in req if f.startswith("observation.images.") and f in features]
    for vf in video_features:
        if features[vf].get("dtype") != "video":
            errors.append(f"Feature {vf} must have dtype 'video', got {features[vf].get('dtype')!r}")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(_infer_repo_id(root), root=root)
    except Exception as exc:
        errors.append(f"Failed loading LeRobotDatasetMetadata: {type(exc).__name__}: {exc}")
        return report

    if meta.episodes is None or len(meta.episodes) != report["total_episodes"]:
        actual = 0 if meta.episodes is None else len(meta.episodes)
        errors.append(f"Episode index count mismatch: meta.episodes={actual} != total_episodes={report['total_episodes']}")
        return report

    expected_from = 0
    for idx in range(report["total_episodes"]):
        ep = meta.episodes[idx]
        length = int(ep["length"])
        ep_from = int(ep["dataset_from_index"])
        ep_to = int(ep["dataset_to_index"])
        if length <= 0 or ep_from != expected_from or ep_to - ep_from != length:
            errors.append(
                f"Episode {idx} index range invalid: from={ep_from}, to={ep_to}, length={length}, expected_from={expected_from}"
            )
        expected_from = ep_to
    if expected_from != report["total_frames"]:
        errors.append(f"Sum of episode lengths ({expected_from}) != total_frames ({report['total_frames']})")

    if episode_index is not None:
        if episode_index < 0 or episode_index >= report["total_episodes"]:
            errors.append(f"Requested episode_index {episode_index} out of range [0, {report['total_episodes']})")
            return report
        target_indices = [int(episode_index)]
    else:
        target_indices = list(range(report["total_episodes"]))

    provenance = _load_provenance(root)
    report["provenance_present"] = provenance["present"]

    parquet_cache = {}
    video_cache = {}
    data_path_tpl = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    non_video_req = [f for f in req if not f.startswith("observation.images.")]
    pts_tol = max(1e-3, 0.05 / fps)

    for idx in target_indices:
        ep = meta.episodes[idx]
        length = int(ep["length"])
        ep_errors = []
        prov = provenance["episodes"].get(idx)
        task_success = (prov.get("result") or {}).get("success") if prov else None
        quality_valid, quality_source, quality_report_path = _lookup_run_quality(prov)
        sim_time_alignment = "not_present"

        if require_success:
            if prov is None:
                ep_errors.append(f"Episode {idx}: missing simulation provenance in meta/simulation")
            elif task_success is not True:
                ep_errors.append(f"Episode {idx}: task_success is {task_success!r} (require_success=True)")

        # 1. Parquet check (lightweight columns only; FEM columns checked in schema only)
        data_key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        if data_key not in parquet_cache:
            pq_path = root / data_path_tpl.format(chunk_index=data_key[0], file_index=data_key[1])
            parquet_cache[data_key] = _check_parquet_shard(pq_path, ["episode_index", "frame_index", "timestamp", *non_video_req])
        pq_res = parquet_cache[data_key]
        parquet_rows = 0
        if not pq_res["ok"]:
            ep_errors.append(pq_res["error"])
        else:
            pdata = pq_res["data"]
            row_ids = [i for i, ep_val in enumerate(pdata["episode_index"]) if int(ep_val) == idx]
            parquet_rows = len(row_ids)
            if parquet_rows != length:
                ep_errors.append(f"Episode {idx}: parquet rows ({parquet_rows}) != episode length ({length})")
            elif parquet_rows > 0:
                frame_indices = [int(pdata["frame_index"][i]) for i in row_ids]
                if frame_indices != list(range(length)):
                    ep_errors.append(f"Episode {idx}: non-contiguous frame_index in parquet")
                ts = np.array([pdata["timestamp"][i] for i in row_ids], dtype=np.float64)
                expected_ts = np.arange(length, dtype=np.float64) / fps
                if not np.isfinite(ts).all() or not np.allclose(ts, expected_ts, atol=1e-3, rtol=0):
                    ep_errors.append(f"Episode {idx}: parquet timestamp does not match frame_index / fps ({fps})")
                if "observation.sim_time_s" in pdata:
                    sim_ts = np.array(
                        [
                            pdata["observation.sim_time_s"][i][0]
                            if isinstance(pdata["observation.sim_time_s"][i], (list, tuple))
                            else pdata["observation.sim_time_s"][i]
                            for i in row_ids
                        ],
                        dtype=np.float64,
                    )
                    if not np.isfinite(sim_ts).all():
                        ep_errors.append(f"Episode {idx}: non-finite observation.sim_time_s")
                    elif length > 1 and not np.allclose(np.diff(sim_ts), 1.0 / fps, atol=1e-3, rtol=0):
                        ep_errors.append(f"Episode {idx}: observation.sim_time_s step does not match fps={fps}")
                    elif prov and "frames" in prov:
                        if not isinstance(prov["frames"], list) or len(prov["frames"]) != length:
                            actual_len = len(prov["frames"]) if isinstance(prov.get("frames"), list) else "invalid"
                            ep_errors.append(
                                f"Episode {idx}: provenance.frames length ({actual_len}) != episode length ({length})"
                            )
                            sim_time_alignment = "provenance_length_mismatch"
                        else:
                            expected_sim = np.array([row["time_s"] for row in prov["frames"]], dtype=np.float64)
                            if not np.allclose(sim_ts, expected_sim, atol=1e-4, rtol=0):
                                ep_errors.append(
                                    f"Episode {idx}: observation.sim_time_s disagrees with provenance.frames.time_s"
                                )
                                sim_time_alignment = "mismatch"
                            else:
                                sim_time_alignment = "matched_provenance"
                    else:
                        sim_time_alignment = "unverified_no_provenance"
                elif prov is not None or require_success:
                    ep_errors.append(f"Episode {idx}: missing observation.sim_time_s column in parquet")
                    sim_time_alignment = "missing_sim_time_s"

                cols_to_check = set(non_video_req) | (
                    set(pdata.keys()) & {"observation.state", "action", "observation.sim_time_s",
                                         "observation.object_position", "observation.task_object_positions"}
                )
                for col in sorted(cols_to_check):
                    if col in pdata and col in features:
                        err = _check_column_shape_and_finite(
                            [pdata[col][i] for i in row_ids],
                            features[col]["shape"],
                            f"episode {idx} {col}",
                        )
                        if err:
                            ep_errors.append(err)

        # 2. Video shard checks (per-frame PTS alignment with from_timestamp + k / fps)
        ep_videos = {}
        for vf in video_features:
            cam_name = vf.removeprefix("observation.images.")
            v_rel = meta.get_video_file_path(idx, vf)
            v_path = root / v_rel
            from_ts = float(ep[f"videos/{vf}/from_timestamp"])
            to_ts = float(ep[f"videos/{vf}/to_timestamp"])
            expected_dur = length / fps
            if to_ts <= from_ts or abs((to_ts - from_ts) - expected_dur) > (0.5 / fps):
                ep_errors.append(
                    f"Episode {idx} {cam_name}: video timestamp range [{from_ts}, {to_ts}] does not match length={length} at fps={fps}"
                )

            v_key = str(v_rel)
            if check_video_decode:
                if v_key not in video_cache:
                    video_cache[v_key] = _decode_video_shard(
                        v_path, features[vf]["shape"], fps, min_rgb_std=min_rgb_std
                    )
                vres = video_cache[v_key]
                if not vres["ok"]:
                    ep_errors.append(f"Episode {idx} {cam_name}: {vres['error']}")
                    ep_videos[cam_name] = dict(path=str(v_rel), from_timestamp=from_ts, to_timestamp=to_ts,
                                                decoded_frames=0, ok=False, error=vres["error"])
                else:
                    tol = 0.5 / fps
                    ep_frame_ts = [t for t in vres["timestamps"] if (from_ts - tol) <= t < (to_ts - tol)]
                    if len(ep_frame_ts) != length:
                        err_msg = (
                            f"Episode {idx} {cam_name}: decoded {len(ep_frame_ts)} frames in "
                            f"[{from_ts}, {to_ts}), expected {length}"
                        )
                        ep_errors.append(err_msg)
                        ep_videos[cam_name] = dict(path=str(v_rel), from_timestamp=from_ts, to_timestamp=to_ts,
                                                    decoded_frames=len(ep_frame_ts), ok=False, error=err_msg)
                    else:
                        expected_pts = from_ts + np.arange(length, dtype=np.float64) / fps
                        actual_pts = np.asarray(ep_frame_ts, dtype=np.float64)
                        max_pts_err = float(np.max(np.abs(actual_pts - expected_pts)))
                        if max_pts_err > pts_tol:
                            err_msg = (
                                f"Episode {idx} {cam_name}: per-frame PTS drift/gap max_err={max_pts_err:.6f}s > {pts_tol:.6f}s"
                            )
                            ep_errors.append(err_msg)
                            ep_videos[cam_name] = dict(path=str(v_rel), from_timestamp=from_ts, to_timestamp=to_ts,
                                                        decoded_frames=len(ep_frame_ts), max_pts_error_s=max_pts_err,
                                                        ok=False, error=err_msg)
                        else:
                            ep_videos[cam_name] = dict(path=str(v_rel), from_timestamp=from_ts, to_timestamp=to_ts,
                                                        decoded_frames=len(ep_frame_ts), max_pts_error_s=max_pts_err, ok=True)
            else:
                exists = v_path.is_file() and v_path.stat().st_size > 0
                if not exists:
                    ep_errors.append(f"Episode {idx} {cam_name}: missing or empty MP4 {v_path}")
                ep_videos[cam_name] = dict(path=str(v_rel), from_timestamp=from_ts, to_timestamp=to_ts, ok=exists)

        errors.extend(ep_errors)
        ep_entry = dict(
            episode_index=idx,
            length=length,
            task_success=task_success,
            quality_valid=quality_valid,
            quality_source=quality_source,
            sim_time_alignment=sim_time_alignment,
            parquet_rows=parquet_rows,
            videos=ep_videos,
            errors=ep_errors,
        )
        if quality_report_path is not None:
            ep_entry["quality_report_path"] = quality_report_path
        report["episodes"].append(ep_entry)

    report["checked_episodes"] = len(report["episodes"])
    report["video_shards"] = {
        k: {key: val for key, val in v.items() if key != "timestamps"}
        for k, v in video_cache.items()
    }
    report["valid"] = len(errors) == 0
    return report
