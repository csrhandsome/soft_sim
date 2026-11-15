"""Versioned file contracts exchanged between the three runtime environments.

Only JSON/NPZ plus this module's validators cross a process boundary. No pickle
and no Python objects are exchanged. Bump a version when the field layout of a
shared file changes; readers reject unknown versions instead of guessing.
"""

import hashlib
import json

CAPTURE_PROTOCOL_VERSION = 2
SCENE_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
DATASET_SIGNATURE_VERSION = 1

#: Fields the dataset signature pins; a mismatch requires a new dataset id.
DATASET_SIGNATURE_KEYS = (
    "robot_type", "joint_names", "active_arm", "finger_scale",
    "finger_resolution", "external_resolution", "finger_intrinsics",
)

CAPTURE_STATES = ("none", "capturing", "incomplete", "staged", "converting", "committed", "failed", "discarded")
EXECUTION_STATES = ("created", "running", "completed", "failed")
VALIDATION_STATES = ("pending", "passed", "failed")


class ContractError(ValueError):
    """A file on disk does not match the declared contract."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ContractError(message)


def validate_capture_metadata(metadata):
    """Validate the staged-capture index written by the simulation environment."""
    require(metadata.get("storage_format") == "lerobot_v3", "Unknown capture storage format")
    require(int(metadata.get("protocol_version", 0)) == CAPTURE_PROTOCOL_VERSION,
            f"Capture protocol {metadata.get('protocol_version')} is not {CAPTURE_PROTOCOL_VERSION}")
    require(metadata.get("status") in CAPTURE_STATES, f"Unknown capture status: {metadata.get('status')}")
    require(float(metadata.get("fps", 0)) > 0, "Capture FPS must be positive")
    require(isinstance(metadata.get("camera_order"), list) and len(metadata["camera_order"]) == 3,
            "Capture must declare three cameras")
    require(metadata.get("color_order") == "RGB", "Capture color order must be RGB")
    frames = metadata.get("frames")
    require(isinstance(frames, list), "Capture frames must be a list")
    require(int(metadata.get("frame_count", -1)) == len(frames), "Capture frame count mismatch")
    for index, row in enumerate(frames):
        require(row.get("index") == index, "Capture frame indices must be contiguous")
        require("time_s" in row and "phase" in row, "Capture row needs time_s and phase")
    if frames:
        require(list(metadata.get("state_keys") or []), "Staged capture must declare state keys")
        require(metadata.get("state_keys") == sorted(metadata["state_keys"]),
                "Capture state keys must be sorted")
    return metadata


def dataset_signature(calibration, fps, features=None):
    """Compatibility signature for appending to an existing dataset."""
    signature = {key: calibration.get(key) for key in DATASET_SIGNATURE_KEYS}
    signature["fps"] = int(fps)
    if features is not None:
        signature["features"] = {name: [list(meta["shape"]), meta["dtype"]]
                                 for name, meta in sorted(features.items())}
    return dict(version=DATASET_SIGNATURE_VERSION, **signature)


def check_dataset_signature(existing, expected):
    """Compare two signatures and name the first differing key."""
    for key in sorted(set(existing) | set(expected)):
        if existing.get(key) != expected.get(key):
            raise ContractError(
                f"Dataset signature differs in {key}: {existing.get(key)!r} != {expected.get(key)!r}. "
                "Use a new dataset id instead of appending incompatible episodes.")
