"""Run from the repository root using the isolated MuJoCo environment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.cli import main
from backends.mujoco_sim.soft_gripper_env import MuJoCoGripperEnv

if __name__ == '__main__':
    main(MuJoCoGripperEnv)
