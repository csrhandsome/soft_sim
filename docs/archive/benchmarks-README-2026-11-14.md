# benchmarks：唯一任务入口、规划与独立复核

当前只有一条任务实现路径：原 RoboDojo 双 X5 任务 + 替换软指，入口为
`robodojo_task_soft.py`。RoboDojo 快照在 `third_party/RoboDojo/`；规划几何、安装变换与
LeRobot v3 采集都由本目录脚本共享同一份参数。安装倾角、闭合间隙与适用范围见
[倾斜安装验证](../docs/x5-tilted-installation-2026-11-14.md)。

## 环境

| 目录 | Python | 核心组件 |
|---|---|---|
| `envs/planning/.venv` | 3.10 | SAPIEN、mplib（软指碰撞规划） |
| `envs/robodojo/.venv` | 3.11 | Isaac Sim 5.1、IsaacLab 0.54.3、CuRobo |
| `envs/meshing/.venv` | 3.11 | 可选的 TetGen 贴合网格 |
| `envs/dataset/.venv` | 3.11 | LeRobot v3 写入、读取与验证 |

安装与资产准备见 [根 README](../README.md)。

## 入口与主要内容

| 文件 | 作用 |
|---|---|
| `robodojo_task_soft.py` | **唯一任务入口**。加载原任务类、官方布局、双 X5 和 RewardManager；支持 `export`／`collect`／`replay`／`assembly` |
| `plan_soft_grasp.py` | 基于真实软指外廓、安装楔块、相机与物体几何的 mplib 抓取／抬升／放置规划 |
| `x5_mount_geometry.py` | 测量安装：倾角、根部位姿、楔块与相机碰撞代理，物理与规划共用 |
| `x5_soft_tool.py` | 软指缩放、FEM 安装与规划 URDF/TCP 生成 |
| `isaac51_tool.py` | `MountedSoftTool`：在原 Isaac Sim 5.1 场景中创建并跟随 FEM 软指与双相机 |
| `motion_execution.py` | 与仿真引擎无关的任务阶段和动作序列 |
| `episode_recording.py` | LeRobot v3 录制器（相机视频 + Parquet 状态） |
| `episode_reader.py` | 读取 LeRobot v3 诊断帧 |
| `finalize_lerobot.py`、`audit_task_episode.py` | 在独立数据集环境中收尾并分别记录任务成败与采集质量 |

## 安装与几何检查

| 文件 | 作用 |
|---|---|
| `check_x5_installation.py` | 离线测量安装：倾角、闭合间隙、原滑块全行程、楔块与腕部碰撞 |
| `compare_x5_finger_sizes.py` | 原装 X5 与新装配在安装坐标下的外廓对比 |
| `test_soft_grasp_frames.py` | 规划 URDF、模拟安装变换、TCP、楔块与相机的一致性测试 |
| `compare_deformation_runs.py` | 在相同保持状态下比较不同运行的原始 RGB |

## 物理与采集复核

| 文件 | 作用 |
|---|---|
| `validate_rgb_episode.py` | 数据完整性、相机投影、FEM 与任务成败的联合校验 |
| `validate_grasp_run.py` | 不重启 Isaac Sim 的抓取运行复核 |
| `check_finger_camera.py` | 指内相机初始投影、遮挡与接触前运动检查 |
| `check_fem_camera_projection.py` | 记录深度与同帧形变 FEM 边界的独立射线比较 |
| `check_finger_motion.py` | 分离相机刚性安装运动与 FEM 形变 |
| `check_camera_stability.py` | 保持段原始 RGB 光流稳定性 |
| `check_clamp_deformation.py` | 区分夹持滑移、整体运动与局部弹性应变 |
| `check_surface_deformation.py` | 检查平直 STL 边遗漏的弯曲显示误差 |
| `check_box_contact.py` | 用独立刚体几何检查记录的碰撞顶点 |
| `check_task_collisions.py` | 执行臂 FEM、楔块与任务物体的碰撞审计 |
| `check_x5_assembly_motion.py` | 原驱动全行程 FEM 采样与楔块、腕部、桌面、积木的碰撞复核 |
| `check_physx_mesh_diagnostics.py` | 解释 PhysX 质量／体积诊断，不用日志条数代替真实单元检查 |
| `check_frozen_render.py` | 同一物理状态下重复渲染的稳定性 |
| `probe_x5_fem_initialization.py` | 隔离 X5 软指初始化的数值探针，不写训练数据 |
| `x5_assembly_motion.py` | 原驱动全行程诊断的执行部分，不评估任务成功 |
| `convert_rgb_to_lerobot.py` | 仅用于把历史完整录制转换为统一 LeRobot v3 格式 |

## 运行

```bash
# 规划安装检查
MPLCONFIGDIR=/tmp/soft-sim-mpl OPENBLAS_NUM_THREADS=1 \
envs/planning/.venv/bin/python benchmarks/check_x5_installation.py \
  --output data/generated/x5_mount/x5_mount_v3_audit

# 任务采集（general_pickup / stack_blocks）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
envs/robodojo/.venv/bin/python benchmarks/robodojo_task_soft.py \
  --task general_pickup --seed 22 --mode collect --finger-scale .75 \
  --dt .001 --solver-iterations 80 --vertex-velocity-damping 20 \
  --texture-max-resolution 512 --fem-mesh data/generated/assets/mesh/soft_finger_tetgen_4mm.npz \
  --dataset-root data/datasets/robodojo_x5_75a_general_pickup22 \
  --output data/generated/x5_pickup22_NEW
```

## 边界

- 物体保持自由刚体，没有绑定或吸附；软指由原机械臂驱动，FEM 未向机械臂反馈反作用力矩。
- 任务成功由原 RewardManager 判定；规划通过、场景加载或装配诊断都不等于任务成功。
- FEM 材料是 TPU 75A 的官方密度与标记为估算的弹性参数，未按实物打印件标定。
- 相机外壳、楔块与紧固结构是仿真包络，螺孔、公差与实际相机包装仍需实物 CAD 校准。
