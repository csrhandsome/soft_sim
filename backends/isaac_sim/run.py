"""Run from the repository root using the isolated Isaac Sim environment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.cli import main
from backends.isaac_sim.soft_gripper_env import IsaacSimGripperEnv
from backends.isaac_sim.app_launcher import add_livestream_arguments, configure_visualization

if __name__ == '__main__':
    main(IsaacSimGripperEnv, add_arguments=add_livestream_arguments,
         configure_simulation=configure_visualization)
