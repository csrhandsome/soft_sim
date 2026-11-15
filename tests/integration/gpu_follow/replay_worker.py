"""Retain numerical capture states while running the real, unchanged recorder."""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    from soft_sim.recording.capture import EpisodeRecorder
    append = EpisodeRecorder.append
    close = EpisodeRecorder.close
    states = []

    def checked_append(self, *images, **state):
        append(self, *images, **state)
        states.append({key: np.asarray(value).copy() for key, value in state.items()
                       if key not in ("phase", "finger_depth_m")})

    def checked_close(self, *a, **kw):
        if states:
            np.savez_compressed(args.states,
                **{key: np.stack([row[key] for row in states]) for key in states[0]})
        return close(self, *a, **kw)

    EpisodeRecorder.append, EpisodeRecorder.close = checked_append, checked_close
    from soft_sim.cli.collect import main as collect
    collect(remaining)


if __name__ == "__main__":
    main()
