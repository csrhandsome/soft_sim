"""Check RoboTwin SAPIEN rendering and mplib planning without downloading all tasks."""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import sapien
import mplib
from shared.dataset import write_png
from shared.geometry import camera_pose, matrix_to_quat_wxyz

scene = sapien.Scene()
scene.set_timestep(.005)
scene.add_ground(0.)
scene.set_ambient_light([.5, .5, .5])
scene.add_directional_light([1, .5, -1], [1, 1, 1], shadow=True)
loader = scene.create_urdf_loader()
loader.fix_root_link = True
asset = ROOT / "RoboTwin/assets/embodiments/franka-panda/panda.urdf"
robot = loader.load(str(asset))
if robot is None:
    raise RuntimeError("Cannot load benchmark robot URDF")
joints = [joint.name for joint in robot.get_active_joints()]
links = [link.name for link in robot.get_links()]
values = dict(panda_joint1=0., panda_joint2=-.4, panda_joint3=0., panda_joint4=-1.8,
              panda_joint5=0., panda_joint6=1.4, panda_joint7=.8, panda_finger_joint1=.03, panda_finger_joint2=.03)
q = np.array([values.get(name, 0.) for name in joints])
robot.set_qpos(q)
for joint in robot.get_active_joints():
    joint.set_drive_properties(stiffness=1000., damping=100.)
for joint, value in zip(robot.get_active_joints(), q):
    joint.set_drive_target(float(value))
planner = mplib.Planner(urdf=str(asset), srdf=str(asset.with_suffix(".srdf")),
    move_group="panda_hand", user_link_names=links, user_joint_names=joints)
goal = q.copy()
goal[joints.index("panda_joint1")] += .12
plan = planner.plan_qpos([goal], q, time_step=.01, planning_time=5.)
if plan["status"] != "Success":
    raise RuntimeError("mplib plan failed: " + str(plan["status"]))
for target in plan["position"]:
    for joint, value in zip(robot.get_active_joints(), target):
        joint.set_drive_target(float(value))
    scene.step()
camera = scene.add_camera(name="overview", width=640, height=480, fovy=.9, near=.01, far=10.)
optical = camera_pose([1., -1., .9], [.25, 0., .35])
rotation = np.column_stack([optical[:3, 2], -optical[:3, 0], -optical[:3, 1]])
camera.set_pose(sapien.Pose(optical[:3, 3], matrix_to_quat_wxyz(rotation)))
scene.update_render()
camera.take_picture()
color = camera.get_picture("Color")[..., :3]
if not np.isfinite(color).all() or color.std() < .001:
    raise RuntimeError("Empty SAPIEN camera frame")
out = ROOT / "data/runs/robotwin_smoke"
out.mkdir(parents=True, exist_ok=True)
write_png(out / "overview.png", (np.clip(color, 0, 1) * 255).astype(np.uint8))
result = dict(sapien=sapien.__version__, mplib=mplib.__version__,
    planning_status=plan["status"], trajectory_steps=len(plan["position"]),
    camera_shape=list(color.shape), soft_body_simulated=False,
    robot_urdf=str(asset))
(out / "result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
