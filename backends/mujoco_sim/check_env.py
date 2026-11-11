"""Check the independent MuJoCo environment and simulate free fall on CPU."""

import sys
from importlib.metadata import PackageNotFoundError, version


def main() -> int:
    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Executable: {sys.executable}", flush=True)
    if sys.version_info[:2] != (3, 12):
        print("ERROR: This environment is configured for Python 3.12.", flush=True)
        return 1

    missing = []
    for name in ("mujoco",):
        try:
            print(f"{name}: {version(name)}", flush=True)
        except PackageNotFoundError:
            missing.append(name)
            print(f"MISSING: {name}", flush=True)
    if missing:
        print("From the repository root, run `bash scripts/sync_envs.sh mujoco`.", flush=True)
        return 1

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 1"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1"/>'
        '</body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    initial_height = float(data.qpos[2])
    for _ in range(100):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or data.qpos[2] >= initial_height:
        raise RuntimeError("MuJoCo free-fall simulation failed.")
    print(f"MuJoCo physics OK: t={data.time:.3f}, z={data.qpos[2]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
