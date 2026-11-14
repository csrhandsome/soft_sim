"""``finalize``: convert a staged capture into the official dataset."""

import argparse
import json
from pathlib import Path

from soft_sim.storage import Storage


def resolve_capture(path=None, run_id=None, storage_root=None):
    if path is not None:
        return Path(path).resolve()
    if run_id is None:
        raise SystemExit("Pass a capture directory or --run-id")
    store = Storage.create(storage_root).run_dir(run_id)
    capture = store / "capture"
    if not (capture / "episode.json").is_file():
        raise SystemExit(f"Run has no staged capture: {capture}")
    return capture


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a staged capture into the official dataset.")
    parser.add_argument("capture", nargs="?", type=Path, help="Capture directory containing episode.json")
    parser.add_argument("--run-id")
    parser.add_argument("--storage-root", type=Path)
    args = parser.parse_args(argv)
    from soft_sim.recording.finalize import finalize
    capture = resolve_capture(args.capture, args.run_id, args.storage_root)
    metadata = finalize(capture)
    print(json.dumps(dict(capture=str(capture), dataset_id=metadata.get("dataset_id"),
                          dataset_root=str((capture / metadata["dataset_root"]).resolve()),
                          episode_index=metadata.get("episode_index"),
                          status=metadata.get("status")), indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
