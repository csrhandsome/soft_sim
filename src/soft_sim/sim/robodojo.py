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


def _to_host_array(value):
    """Copy a device tensor to CPU host memory without altering CPU/NumPy inputs."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def ensure_articulation_dof_properties_compat(articulation_cls=None):
    """Copy CUDA articulation view getters to host before SingleArticulation.dof_properties packs NumPy.

    Idempotent, bounded to ``dof_properties``, and preserves all original DOF limits,
    efforts, velocities, gains, and articulation view device/backend state.
    """
    if articulation_cls is None:
        from isaacsim.core.prims.impl.single_articulation import SingleArticulation as articulation_cls

    prop = getattr(articulation_cls, "dof_properties", None)
    if not isinstance(prop, property) or prop.fget is None:
        return False
    if getattr(prop.fget, "_soft_sim_dof_properties_compat", False):
        return False

    orig_fget = prop.fget

    def _dof_properties_host_compat(self):
        try:
            return orig_fget(self)
        except TypeError:
            import numpy as np

            dtype = np.dtype(
                [
                    ("type", int),
                    ("hasLimits", bool),
                    ("lower", float),
                    ("upper", float),
                    ("driveMode", int),
                    ("maxVelocity", float),
                    ("maxEffort", float),
                    ("stiffness", float),
                    ("damping", float),
                ]
            )
            properties = np.zeros(self.num_dof, dtype=dtype)
            view = self._articulation_view
            properties["type"] = _to_host_array(view.get_dof_types()[0])
            limits = _to_host_array(view.get_dof_limits()[0])
            properties["lower"] = limits[:, 0]
            properties["upper"] = limits[:, 1]
            properties["hasLimits"] = properties["lower"] < properties["upper"]
            properties["driveMode"] = _to_host_array(view.get_drive_types()[0])
            properties["maxEffort"] = _to_host_array(view.get_max_efforts()[0])
            properties["maxVelocity"] = _to_host_array(view.get_joint_max_velocities()[0])
            stiffnesses, dampings = view.get_gains()
            properties["stiffness"] = _to_host_array(stiffnesses[0])
            properties["damping"] = _to_host_array(dampings[0])
            return properties

    _dof_properties_host_compat._soft_sim_dof_properties_compat = True
    articulation_cls.dof_properties = property(
        _dof_properties_host_compat,
        prop.fset,
        prop.fdel,
        prop.__doc__,
    )
    return True


def load_task_class(task):
    _ensure_snapshot_importable()
    ensure_articulation_dof_properties_compat()
    from task.RoboDojo.task_registry import load_task_class as upstream
    return upstream(task)


def load_yaml(relative):
    _ensure_snapshot_importable()
    from utils.load_file import load_yaml as upstream
    return upstream(SNAPSHOT / relative)


def layout_path(task, seed):
    return SNAPSHOT / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_{seed}.json"


def layout_materials(layout):
    """Material packages referenced by the saved table, ground and objects."""
    import re
    found = set()

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and re.fullmatch(r"material_\d+", value):
            found.add(value)

    visit(layout)
    return sorted(found)


def validate_layout_assets(layout):
    """Fail before Kit startup when an official layout's selected input is absent."""
    missing = []
    room = layout.get("Room", {}).get("default")
    if room:
        directory = SNAPSHOT / "Assets/Room" / room
        if not any(directory.glob("*.usd")):
            missing.append(str(directory / "*.usd"))
    for material in layout_materials(layout):
        directory = SNAPSHOT / "Assets/Material" / material
        if not any(directory.glob("*.mdl")):
            missing.append(str(directory / "*.mdl"))
    background = layout.get("Background", {}).get("category_name")
    if background and not (SNAPSHOT / "Assets/Background" / background).is_file():
        missing.append(str(SNAPSHOT / "Assets/Background" / background))
    for kind in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"):
        for category, records in layout.get(kind, {}).items():
            for record in records:
                source_kind = "Clutter" if record.get("type") == "cluttered" else kind
                directory = SNAPSHOT / "Assets/Object/RoboDojo" / source_kind / category / f"{record['category_idx']:05d}"
                if not any((directory / f"object.{ext}").is_file() for ext in ("usd", "usdz")):
                    missing.append(str(directory / "object.usd[z]"))
                if not (directory / "metadata.json").is_file():
                    missing.append(str(directory / "metadata.json"))
    if missing:
        raise FileNotFoundError("Official layout inputs missing: " + ", ".join(sorted(set(missing))))


def robot_init_joint_pos(robot_name):
    """Joint pose the upstream robot description applies at reset, by joint name."""
    _ensure_snapshot_importable()
    import importlib
    module = importlib.import_module(f"env.robot_manager.robot_config.{robot_name}")
    return dict(module.get_robot_config().init_state.joint_pos)


def simulator_config(*, seed, dt, solver_iterations, texture_max_resolution):
    """Original sim configuration with only the validated overrides applied."""
    import numpy as np
    config = load_yaml("env_cfg/sim/sim_config.yml")
    config.update(device="cuda:0", use_fabric=True, seed=seed, dt=dt, decimation=1)
    config["scene"]["num_envs"] = 1
    # Match IsaacLab PhysxCfg soft-body contact default (2**20) and size the
    # GPU collision stack (2**27 = 128 MiB) above the 76.5 MB peak observed
    # during multi-object FEM grasping (e.g. cover_blocks).
    config["physx"] = dict(
        min_position_iteration_count=solver_iterations, max_position_iteration_count=255,
        gpu_max_rigid_contact_count=1048576, gpu_max_rigid_patch_count=65536,
        gpu_found_lost_pairs_capacity=262144, gpu_found_lost_aggregate_pairs_capacity=1048576,
        gpu_total_aggregate_pairs_capacity=262144, gpu_collision_stack_size=134217728,
        gpu_max_soft_body_contacts=1048576, gpu_max_particle_contacts=65536)
    mip = int(np.log2(texture_max_resolution)) + 1
    config["render"] = dict(
        antialiasing_mode="FXAA", rendering_mode="balanced",
        carb_settings={"/rtx-transient/resourcemanager/texturestreaming/memoryBudget": .15,
                       "/rtx-transient/resourcemanager/maxMipCount": mip})
    return config


def robot_config(robot_names=("x5", "x5")):
    """Two-robot episode; the active arm may be replaced, the parked one stays stock.

    Upstream robot descriptions are read, never copied. A replacement arm keeps
    the X5 station it takes over so the official layout stays inside its reach,
    and drops the upstream ``type: support`` tag because it is now the active
    end effector rather than a passive third arm.
    """
    from copy import deepcopy
    stations = load_yaml("env_cfg/robot/dual_x5.yml")["robots"]
    mixed = load_yaml("env_cfg/robot/dual_x5_and_franka_competition.yml")["robots"]
    candidates = {"x5": stations,
                  "franka": [entry for entry in mixed if entry.get("robot_name") == "franka"]}
    robots = []
    for index, name in enumerate(robot_names):
        if name not in candidates or not candidates[name]:
            raise ValueError(f"Upstream robot description missing for {name!r}")
        entry = deepcopy(candidates[name][index] if name == "x5" else candidates[name][0])
        if name != "x5":
            entry.pop("type", None)
            entry["default_root_pos"] = list(stations[index]["default_root_pos"])
            entry["default_root_rot"] = list(stations[index]["default_root_rot"])
        entry["need_planner"] = False
        robots.append(entry)
    return {"robots": robots}


def scene_config():
    return load_yaml("env_cfg/scene/default.yml")


def task_env_config(task_name):
    return load_yaml(f"task/RoboDojo/config/{task_name}.yml")


def task_configuration(task_name, *, seed, dt, solver_iterations, texture_max_resolution, robot=None):
    """The OmegaConf object the upstream task class expects."""
    from omegaconf import OmegaConf
    return OmegaConf.create(dict(
        sim=simulator_config(seed=seed, dt=dt, solver_iterations=solver_iterations,
                             texture_max_resolution=texture_max_resolution),
        robot=robot if robot is not None else robot_config(), scene=scene_config(),
        camera=head_camera_config(),
        task_env=task_env_config(task_name)))


def head_camera_config():
    """Keep the original table camera; attach its RGB product explicitly.

    The robot manager injects factory wrist cameras into the task config.
    Reusing this config as CameraManager.init_config keeps only cam_head.
    Disable upstream annotators because RGB is captured explicitly. The
    upstream capture manager requires an annotator mapping even when disabled.
    """
    from omegaconf import OmegaConf
    config = load_yaml("env_cfg/camera/camera_config.yml")
    return OmegaConf.create(dict(default_frequency=None,
                                 annotator=dict(common=dict(enabled=False)),
                                 cam_head=config["cam_head"]))


def robot_link_path(stage, robot_index, link_name):
    """USD path of one articulation link, whatever the upstream asset nests.

    The X5 USD exposes its links directly under the spawned robot prim, while
    Franka wraps them in a ``panda`` default prim. Search by name instead of
    assuming a depth, so both snapshots load without a per-robot path table.
    """
    import sys as _sys
    from pxr import Usd
    root = stage.GetPrimAtPath(f"/World/envs/env_0/robot{robot_index}")
    if not root or not root.IsValid():
        raise RuntimeError(f"Missing robot prim for index {robot_index}")
    names = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetName() == link_name and prim.GetTypeName() != "Mesh":
            return str(prim.GetPath())
        names.append(prim.GetName())
    raise RuntimeError(f"Link {link_name!r} not found under {root.GetPath()}; "
                       f"available prims: {sorted(set(names))[:40]}")


def numpy_pose(pose):
    """Normalize a CUDA scene-wrapper pose for the upstream NumPy reward parser."""
    import numpy as np
    return tuple(v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v) for v in pose)


def original_reward_poses(manager):
    """Return a manager whose poses are NumPy tuples, as the reward parser expects."""
    for inst_name, itype in getattr(manager, "instance_type_by_env", [{}])[0].items():
        if itype in ("garment", "geometry"):
            obj = manager.get_scene_object(0, inst_name)
            if hasattr(obj, "env_origin") and hasattr(obj.env_origin, "cpu"):
                obj.env_origin = obj.env_origin.cpu()
    original = manager.get_instance_pose
    manager.get_instance_pose = lambda *a, **kw: numpy_pose(original(*a, **kw))
    return manager


def arm_root_transform(robot_description, arm_index, yaw_deg=90):
    """World transform of the episode's arm base (original position, yaw 90)."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    base[:3, 3] = robot_description["robots"][arm_index]["default_root_pos"]
    return base
