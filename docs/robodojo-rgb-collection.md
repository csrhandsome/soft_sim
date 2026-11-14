# RoboDojo 双 X5 软指的 LeRobot v3 采集

> **路径变更（2026-11-14 重构）**：日常入口改为 `python scripts/run.py <command>`，实现位于 `src/soft_sim/`，生成物位于 `artifacts/`。本文保留写作时的目录与命令，当前命令见 [README](../README.md)。


唯一入口：`benchmarks/robodojo_task_soft.py`。任务类、官方布局、机械臂、物体、材质与成功判定
来自 `third_party/RoboDojo/`；软指、楔块与指内相机由本项目安装，三路相机在同一仿真时刻采样。

## 采集

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
envs/robodojo/.venv/bin/python benchmarks/robodojo_task_soft.py \
  --task general_pickup --seed 22 --mode collect --finger-scale .75 \
  --dt .001 --solver-iterations 80 --vertex-velocity-damping 20 \
  --texture-max-resolution 512 --fem-mesh data/generated/assets/mesh/soft_finger_tetgen_4mm.npz \
  --dataset-root data/datasets/robodojo_x5_75a_general_pickup22 \
  --output data/generated/x5_pickup22_NEW
```

- `--mode collect` 从实时场景调用规划器，执行接近、闭合、保持、试提、抬升和保持。
- `stack_blocks` 依次完成两次抓取、转移、放置、松爪、撤离与回位，第二块前重新导出并规划。
- `--mode export` 只导出场景与安装校验；`--mode replay` 用已有 plan 执行（仅 `general_pickup`）；
  `--mode assembly` 是原驱动全行程诊断，不评估任务成功。
- 每次运行使用新的输出目录；`--fps` 必须整除物理步频。

## 输出

| 路径 | 内容 |
| --- | --- |
| `recording/dataset` | LeRobot v3 数据集（视频 + Parquet 状态） |
| `recording/episode.json` | 帧索引、阶段、标定、完成状态与任务结果 |
| `scene.json`、`mount_validation.json` | 场景、安装变换与实际/规划安装误差 |
| `initial_fem.npz`、`initial_fem_validation.json` | 初始 FEM 状态与单元检查 |
| `object_geometry.npz`、`obstacle_geometry.npz` | 目标与障碍物几何 |
| `run_failure.json` | 异常或未完成运行的失败原因，不写入训练数据 |

三路相机为执行臂左右指内相机与外部固定相机。指内相机固定在原 `link8/link7` 刚体下，
随实际 PhysX 位姿更新；深度与彩色在同一时刻渲染。关节位置、速度、指令位置、物体位姿和
FEM 节点一并保存，`active_arm`、颜色、材料和安装参数写入标定元数据。

## 复核

```bash
envs/dataset/.venv/bin/python benchmarks/validate_rgb_episode.py OUTPUT/recording
envs/dataset/.venv/bin/python benchmarks/check_finger_camera.py OUTPUT
envs/dataset/.venv/bin/python benchmarks/check_finger_motion.py OUTPUT/recording
envs/dataset/.venv/bin/python benchmarks/check_fem_camera_projection.py OUTPUT
envs/dataset/.venv/bin/python benchmarks/check_camera_stability.py OUTPUT/recording
envs/dataset/.venv/bin/python benchmarks/audit_task_episode.py OUTPUT
```

任务成败、数据完整性、相机与 FEM 质量分别记录。任何一项不通过都保留为 incomplete；
成功抓取一例不代表随机布局成功率，也不代表 `stack_blocks` 已完成。
历史逐帧 NPZ 记录与旧入口的调试过程见
[归档：三路 RGB 采集](archive/robodojo-rgb-collection.md)。

## 边界

- 物体保持自由刚体，没有绑定或吸附；软指由原驱动执行，FEM 不反馈反作用力矩给机械臂。
- 任务成功由原 RewardManager 判定；场景加载、装配检查或规划成功都不算成功。
- 房间墙体未纳入规划碰撞集合；碰撞范围写入 `scene.json`。
