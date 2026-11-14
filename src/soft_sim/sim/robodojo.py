"""Upstream RoboDojo loading; the only place that adapts its package layout.

RoboDojo is not an installed package: its modules import ``task.*``, ``utils.*``
and ``env_cfg`` relative to the snapshot root. Adding that one root to the
import path is the documented exception; project code never injects its own
search paths. Everything here is imported only after Isaac Sim has started.
"""

from soft_sim.config import ROOT

SNAPSHOT = ROOT / "third_party/RoboDojo"


def _ensure_snapshot_importable():
    import sys
    if str(SNAPSHOT) not in sys.path:
        sys.path.insert(0, str(SNAPSHOT))


def load_task_class(task):
    _ensure_snapshot_importable()
    from task.RoboDojo.task_registry import load_task_class as upstream
    return upstream(task)


def load_yaml(relative):
    _ensure_snapshot_importable()
    from utils.load_file import load_yaml as upstream
    return upstream(SNAPSHOT / relative)


def layout_path(task, seed):
    return SNAPSHOT / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_{seed}.json"


def simulator_config(*, seed, dt, solver_iterations, texture_max_resolution):
    """Original sim configuration with only the validated overrides applied."""
    import numpy as np
    config = load_yaml("env_cfg/sim/sim_config.yml")
    config.update(device="cuda:0", use_fabric=True, seed=seed, dt=dt, decimation=1)
    config["scene"]["num_envs"] = 1
    config["physx"] = dict(
        min_position_iteration_count=solver_iterations, max_position_iteration_count=255,
        gpu_max_rigid_contact_count=1048576, gpu_max_rigid_patch_count=65536,
        gpu_found_lost_pairs_capacity=262144, gpu_found_lost_aggregate_pairs_capacity=1048576,
        gpu_total_aggregate_pairs_capacity=262144, gpu_collision_stack_size=33554432,
        gpu_max_soft_body_contacts=262144, gpu_max_particle_contacts=65536)
    mip = int(np.log2(texture_max_resolution)) + 1
    config["render"] = dict(
        antialiasing_mode="FXAA", rendering_mode="balanced",
        carb_settings={"/rtx-transient/resourcemanager/texturestreaming/memoryBudget": .15,
                       "/rtx-transient/resourcemanager/maxMipCount": mip})
    return config


def robot_config(arm_count=2):
    """Original dual X5 description with the upstream planner disabled."""
    config = load_yaml("env_cfg/robot/dual_x5.yml")
    for robot in config["robots"][:arm_count]:
        robot["need_planner"] = False
    return config


def scene_config():
    return load_yaml("env_cfg/scene/default.yml")


def task_env_config(task_name):
    return load_yaml(f"task/RoboDojo/config/{task_name}.yml")


def task_configuration(task_name, *, seed, dt, solver_iterations, texture_max_resolution):
    """The OmegaConf object the upstream task class expects."""
    from omegaconf import OmegaConf
    return OmegaConf.create(dict(
        sim=simulator_config(seed=seed, dt=dt, solver_iterations=solver_iterations,
                             texture_max_resolution=texture_max_resolution),
        robot=robot_config(), scene=scene_config(),
        camera=head_camera_config(),
        task_env=task_env_config(task_name)))


def head_camera_config():
    """Keep the original table camera; attach its RGB product explicitly.

    The robot manager injects factory wrist cameras into the task config.
    Reusing this config as CameraManager.init_config keeps only cam_head.
    A null annotator disables the upstream tiled capture products.
    """
    from omegaconf import OmegaConf
    config = load_yaml("env_cfg/camera/camera_config.yml")
    return OmegaConf.create(dict(default_frequency=None, annotator=None,
                                 cam_head=config["cam_head"]))


def numpy_pose(pose):
    """Normalize a CUDA scene-wrapper pose for the upstream NumPy reward parser."""
    import numpy as np
    return tuple(v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v) for v in pose)


def original_reward_poses(manager):
    """Return a manager whose poses are NumPy tuples, as the reward parser expects."""
    original = manager.get_instance_pose
    manager.get_instance_pose = lambda *a, **kw: numpy_pose(original(*a, **kw))
    return manager


def arm_root_transform(robot_description, arm_index, yaw_deg=90):
    """World transform of the original X5 base (yaw 90 degrees, original position)."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    base[:3, 3] = robot_description["robots"][arm_index]["default_root_pos"]
    return base
