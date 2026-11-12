"""GPU smoke test for RoboDojo's own CuRobo planner, stock Franka only."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="data/runs/robodojo_planner")
args = parser.parse_args()
sys.path.insert(0, str(ROOT / "RoboDojo"))
import numpy as np
import yaml
import env.global_configs as global_config
# This is a single-scene validation, so do not allocate ten batch slots.
global_config.BATCH_NUM = 1
from env.planner_manager.curobo_planner import CuroboPlanner
source = ROOT / "RoboDojo/Assets/Robots/franka/curobo.yml"
config = yaml.safe_load(source.read_text())
kinematics = config["robot_cfg"]["kinematics"]
asset_root = ROOT / "RoboTwin/assets/embodiments/franka-panda"
kinematics["asset_root_path"] = str(asset_root)
kinematics["urdf_path"] = str(asset_root / "panda.urdf")
output = ROOT / args.output / "curobo.yml"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(config, sort_keys=False))
joints = ["panda_joint" + str(i) for i in range(1, 8)]
planner = CuroboPlanner(robot_origin_pose=[0., 0., .12, 1., 0., 0., 0.],
    active_joints_name=joints, all_joints=joints+["panda_finger_joint1", "panda_finger_joint2"],
    dt=.01, yml_path=str(output), table_height=0.)  # Scene is in the robot-base frame
start = np.array([0., -.4, 0., -1.8, 0., 1.4, .8])
goal = start.copy()
goal[0] += .12
plan = planner.plan_joint(start, goal)
if plan["status"] != "Success":
    raise RuntimeError("RoboDojo CuRobo failed: " + str(plan["status"]))
report = dict(status=plan["status"], trajectory_shape=list(np.asarray(plan["position"]).shape),
    stock_franka=True, custom_soft_tool_collision_model=False)
path = ROOT / args.output
path.mkdir(parents=True, exist_ok=True)
(path / "result.json").write_text(json.dumps(report, indent=2))
print("PLANNER_RESULT", json.dumps(report), flush=True)
