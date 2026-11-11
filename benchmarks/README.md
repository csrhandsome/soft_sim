# RoboTwin / RoboDojo 与带相机软夹爪

新增构形适配规划见 [软夹爪抓取规划](SOFT_GRASP_PLANNING.md)。优先使用该文档的 TCP/碰撞模型/候选筛选流程；原固定姿态入口保留作对照。

当前工作目录：`~/DataFromRoot/codes/grade_two/my_projects/robot/soft_sim`（ssh asus）。

## 独立环境

| 目录 | Python | 核心组件 |
|---|---|---|
| `RoboTwin/.venv` | 3.10.18 | SAPIEN 3.0.0b1、mplib 0.2.1、Torch 2.4.1+cu121、CuRobo v0.7.8 |
| `RoboDojo/.venv` | 3.11.13 | Isaac Sim 5.1.0、Isaac Lab 0.54.3、Torch 2.7.0、RoboDojo 固定的 CuRobo v2 分支 |

两个 checkout 的固定提交见根 `pyproject.toml`；Python 安装配置和锁文件分别在 `envs/planning/` 与 `envs/robodojo/`。
具体安装命令见 [根 README](../README.md)。旧的 pip freeze 文本已由 uv 锁文件替代。

### Git 子模块清理（2026-11-11）

RoboDojo 与 RoboTwin 均已移除全部子模块登记。IsaacLab 与 RoboDojo CuRobo 使用 `.deps/` 下的固定提交源码快照，通过 TOML editable sources 安装；保持原有 YAML、extension config 与 CUDA 源码可用。
上游 checkout 的可重放改动保存在 `patches/`，准备脚本会重复执行检查。XPolicyLab 不再作为当前流程的依赖；原工作目录、Git 对象与配置都保存在 `data/backups/`。

IsaacLab 的 Starlette 兼容约束保持 `>=0.45.3,<=0.49.1`，Isaac Sim 仍使用 0.45.3。依赖版本约束来自迁移前环境，避免迁移时升级 CUDA/PyTorch。
当前流程不使用 PyTorch3D；双 RGB/depth 相机和原有规划流程继续保留。

## 已下载的资产

- RoboTwin：官方 `embodiments.zip`，219,847,741 bytes，ZIP CRC 校验通过；
  解压后包含 `franka-panda`、`piper`、`aloha-agilex`、`ARX-X5`、`ur5-wsg`。
  已执行官方 embodiment 路径配置脚本。
- RoboDojo：`Assets/Robots/franka` 完整目录、`Assets/Object/RoboDojo/Clutter/ball/00000`、
  `Assets/Material/material_0122`（Mahogany Planks），另有 Rigid 下 bottle/mug/block 的 00000 版本。按需补充物体，避免下载全部数据集或策略权重。
- 下载脚本 `scripts/download_benchmark_assets.py` 从 RoboDojo 官方 ModelScope 数据集下载，
  校验路径、大小和 SHA-256，默认单次限制 1 GB，并在 `data/generated/` 记录清单。

```bash
python3 scripts/download_benchmark_assets.py --prefix Assets/Object/RoboDojo/Rigid/bottle/00000 --list
python3 scripts/download_benchmark_assets.py --prefix Assets/Object/RoboDojo/Rigid/bottle/00000
# 可替换为 mug/00000、block/00000，或其他已确认的目录。
```

RoboDojo 资产库还包含 rigid、articulation、garment、fluid、dynamic 等类型，以及材质和任务布局。
clone 仓库不等于下载所有资产；当前没有下载全量任务、训练数据、模型权重。RoboTwin 完整任务还需要 objects.zip 和 background_texture.zip；当前验证的是已下载机器人、本例物体、规划器、相机和自定义软指流程。

## 运行

以下命令均在 soft_sim 根目录执行，不使用原来的两个环境。

```bash
# SAPIEN + 原生 RoboTwin Franka + mplib 真实轨迹规划 + 相机渲染
RoboTwin/.venv/bin/python benchmarks/check_robotwin.py

# RoboDojo Franka、球体、木纹桌面 + shared 软指与双相机
OMNI_KIT_ACCEPT_EULA=YES RoboDojo/.venv/bin/python benchmarks/robodojo_soft_pick.py --mode smoke
OMNI_KIT_ACCEPT_EULA=YES RoboDojo/.venv/bin/python benchmarks/robodojo_soft_pick.py --mode pick

# 无物体碰撞对照；物体保持运动学静止，不参与软指接触
OMNI_KIT_ACCEPT_EULA=YES RoboDojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --mode no-contact --output data/runs/robodojo_no_contact

# 使用基本方块排除资产碰撞网格的影响
OMNI_KIT_ACCEPT_EULA=YES RoboDojo/.venv/bin/python benchmarks/robodojo_soft_pick.py --primitive-box

# 单独指定现成 USD/USDZ 物体；沿夹持方向（世界 Y）归一化到 40 mm，可用 --object-width 修改（便于当前夹爪测试）
OMNI_KIT_ACCEPT_EULA=YES RoboDojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --object-usd RoboDojo/Assets/Object/RoboDojo/Rigid/bottle/00000/object.usdz

# 远程查看（独立进程，避免与另一个 Isaac Sim 实例争用 GPU）
OMNI_KIT_ACCEPT_EULA=YES RoboDojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --livestream --livestream-ip SERVER_IP --mode pick
```

默认无窗口。每次运行有限时长并自动退出；日志通过 shell 重定向保留。
结果在 `data/runs/robodojo_soft_pick/`：`trajectory.json`、`result.json`、各阶段两相机 PNG 和 NPZ。
RGB/depth、相机内外参分别存储。静止图像会检查是否因外壳遮挡而成为均匀黑图。

## 接入方式与边界

`isaac51_tool.py::MountedSoftTool` 不创建 app、world 或独立物理场景，适合放入 benchmark 已有场景：

1. host 开启 GPU dynamics，启动 Isaac Sim 5.1 的 PhysX 场景。
2. host reset 前，用两侧实际安装位姿创建 MountedSoftTool。
3. host reset 后调用 `initialize()`；从 5.1 实际生成的仿真节点中确定底部固定节点。
4. 每个物理步前，从机器人关节/FK 或实际刚体位姿生成安装矩阵并调用 `follow()`。
5. host 的原有 clock 推进一步；按需 `capture()` 获取两相机图像。由 host 关闭 Kit。

软指复用 `shared` 的 STL、材料、相机参数和安装矩阵工具。
5.1 legacy FEM 重新从 STL 生成仿真/碰撞网格，不能直接套用 6.2 TetMesh 的节点顺序。
默认只固定底部 4 mm，自由节点由 PhysX 求解，不写入预制形变。
两侧网格的节点数可能不同，逐侧计算 pin mask，不能共用节点编号。

Franka 原装刚性指尖隐藏并禁用碰撞，保留原生关节；软指底座跟随测量到的手掌和开口。
底座距 palm frame 前方 75 mm，防止相机被掌部外壳挡住。
`--jaw-width` 是旧共享参数；实际底座间距 = jaw-width + 2*adapter-offset。
资产按夹持 Y 轴宽度统一缩放，保持长宽高比例；不能用最大 XY 尺寸当作实际夹持宽度。
STL 底部宽 40 mm、指端逐渐变窄，因此该参数不等于每个高度的实际指内间隙。
当前检查底座间距至少 41 mm，避免两个底部实体重叠。

默认 arm-control=kinematic，逐步设置手臂关节，再由测量 FK 驱动软指底座。
对象是自由刚体，由碰撞与摩擦决定是否提起；没有固定关节、吸附或物体位姿写入。
也提供 `--arm-control drive` 试验关节驱动，但两种方式都没有把 FEM 反作用力矩反馈给机械臂。

这是资产和软工具的接入示例，未启动完整 RoboDojo task/observation/reward manager，也不是官方 benchmark 分数。
当前抓取入口用 Lula IK 与空桌面脚本路径。两个环境已安装 CuRobo，RoboTwin 的 mplib 规划另有实跑验证；
要把 CuRobo 用于自定义软夹爪的避障，仍需增加软指、底座和相机的碰撞代理，以及正确的 TCP/抓取点偏置，
再把原有任务动作与相机 observation 接到此工具。不能沿用原装夹爪的碰撞包络就声称新夹爪已完成避障。

RoboTwin 主分支是 SAPIEN 后端。这里的 PhysX FEM 不能作为 SAPIEN articulation 子部件直接加载；
若要保留相同软体物理，优先使用 RoboDojo/Isaac Sim。SAPIEN 侧若仅换视觉网格，不能等价于软体仿真。

## “未接触就形变”的定位

历史 `data/logs/contact_diagnostic.json`（原 6.2 环境）已显示：关闭物体碰撞后，
运动过程仍有约 2.3 mm 位移，静止重力下约 0.8 mm。说明形变包含重力与惯性响应，不能直接当接触标签。
旧的 2 mm voxel FEM 与原 STL 外形也有近似误差，显示表面与碰撞表面可能不一致。
新 5.1 示例显式把刚体和软体 contact offset 设为 0.25 mm、rest offset=0；接触仍有有限求解容差。

相机形变估力需使用无接触对照和稳定段，不能把准静态 FEM 估计当成求解器原生接触力。
当前示例不输出原生软体接触力真值。
