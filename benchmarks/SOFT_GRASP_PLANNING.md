# 针对新软夹爪的抓取规划

## 2026-11-13：原双 X5 与非对称物体

当前规划支持 RoboDojo 原双 X5 的原生滑块和等比缩放软指，见
[原生移植说明](../docs/robodojo-native-x5.md)。新增两项与实际夹持直接有关的修正：

- 从原始三角表面求射线接触点和法线。凸包仍用于保守避碰与负载模型，但不再代替真实接触面。
- TCP 位于两个实际接触点的中点，使左右预压量对称；不再把包围盒中心直接当作夹持中心。

官方 `general_pickup_22` 的旧候选凸包宽 56.1 mm，实际同一截面约 51.0 mm。
改用实际表面后选出的截面宽约 48.83 mm，其接触中点仍偏离包围盒中心约 9.59 mm。
未居中的试验中，左指相对接触前最大形变仅约 0.047 mm，右指约 18.27 mm，
两次 12 mm 试提均未带起目标。因此新规划同时修正宽度、法线和接触中点。
新增 U 形目标及非对称接触点回归，分别防止夹持凸包空隙与左右预压失衡。

下面 2026-11-10 的结果是历史 Franka/85A 实验；不能当作新 X5/75A 的成功结果。

## 实测结果（2026-11-10）

同一 RoboDojo 不规则球体资产，均匀缩放到夹持 Y 方向 40 mm，质量 30 g，Isaac Sim 5.1 FEM。
新规划已实际完成夹持、12 mm 试提、后续提起和保持；不是仅通过几何规划。

- 物体最终抬升约 81 mm；整个最后 1 秒保持段最低抬升约 77.5 mm。
- 最终 XY 漂移约 15.7 mm，两台相机均有有效 RGB/depth。
- 旧固定顶抓姿态的对照最大抬升约 0.11 mm，未抓起物体。
- 当前仅一件物体、一次成功运行，尚无跨物体/随机种子成功率统计。

原始结果与轨迹：`data/runs/soft_planned_grasp_55/`。
可独立复核：`envs/robodojo/.venv/bin/python benchmarks/validate_grasp_run.py`。
`validation.json` 是从保存的原始轨迹及相机文件重算的检查结果。
`plan.json` 记录实际执行轨迹和 URDF/STL 的校验和。

## 改动重点

保留 Franka 的 `panda_hand` 腕部/掌部坐标，增加具有明确物理意义的 `soft_grasp_tcp`。
新 TCP 的 X 是夹持方向，Z 是手指从底部到指端的方向；它不等于原装夹爪的 finger tip。
对于当前装配：

- 掌部到软指底座：75 mm。
- 新候选采用的接触区：从软指底部向前 55 mm；因此掌部到 TCP 为 130 mm。
- TCP 相对掌部有绕 Z 的 90° 旋转，与 `shared` 两指安装轴一致。
- 原固定脚本采用 73 mm 指端附近接触点，对应 148 mm 掌部偏置。

一般变换应写为：`T_world_hand = T_world_tcp @ inverse(T_hand_tcp)`。
规划器、桌面、物体和被抓负载必须统一到机器人基座坐标。
当前机器人基座在世界 Z=120 mm；`plan.json` 显式记录这一变换。

`test_soft_grasp_frames.py` 通过独立的 mplib FK 比较生成的 URDF 和 `shared.tool_mount`，
覆盖张开/闭合两种开口；底座与 TCP 的位置、旋转误差需小于 2e-7。

## 为什么需要搜索姿态

STL 高约 88.18 mm，横截面沿长度逐渐变细：

| 距底座高度 | 朝内方向半径 |
|---|---|
| 40 mm | 12.77 mm |
| 55 mm | 9.58 mm |
| 73 mm | 5.76 mm |

竖直抓桌上矮物体时，让较粗的指身接触物体会使长指端穿入桌面。
旧脚本还把不规则球体的最大 XY 尺寸当作夹持方向宽度，实际截面并没有夹到。
因此新规划同时搜索夹持 yaw、接近倾角、接触高度与接触中心高度。

## 实现

`plan_soft_grasp.py`：

1. 使用 RoboDojo 物体的实际网格凸包，而不是把资产名称 ball 当成理想球体。
2. 从 TCP 两侧沿夹持轴射线求接触位置及法线；用摩擦锥条件筛选对向接触。
3. 根据 STL 接触截面的半径、物体实际接触宽度和预压量计算闭合开口；
   不把 Panda 的两关节位移直接当作软指表面间隙。
4. 拒绝会使 STL、掌部或转接件碰到桌面的候选，并检查底部构件闭合后自碰撞。
5. 生成带 `soft_grasp_tcp`、双软指 STL、底座和相机碰撞模型的独立 URDF，原资产不改动。
6. 用 mplib RRT 规划预抓取路径，以逐点 IK 和碰撞检查规划最后 90 mm 接近及 12 mm 试提。
7. 在规划模型中附加物体负载，用 RRT 为后续 100 mm 提起规划路径；实际物体抬升受软指形变影响。

手指开口对应的两个非机械臂关节在 IK 过程中固定，避免随机初始种子改变夹爪构形。
转接件/相机代理采用明确尺寸（40×40×16.6 mm、12×12×8 mm）；它们尚未根据实物 CAD 标定。
转接件用同形三角网格表示，保持与 STL 一致的碰撞处理路径，不移除自碰撞检查。

已生成的 55 mm 接触区方案：80° 倾角、90° 夹持 yaw、表面间隙约 2.65 mm，
TCP 在掌部前方 130 mm，预抓取开口参数 80 mm、闭合参数 27 mm（底座间距 43 mm）。
这条保存的轨迹已通过上述单轮 FEM 验证；换物体或重新生成轨迹后仍需重新验证。

## 运行

在 soft_sim 根目录执行：

```bash
# 导出相同场景的物体真实网格和初始关节；不启动软体长时间仿真
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --mode export --output data/generated/soft_grasp_scene

# 生成候选、适配 URDF、RRT/直线接近/试提/携带物体提起轨迹
envs/planning/.venv/bin/python benchmarks/plan_soft_grasp.py \
  --contact-height .055 --output data/generated/soft_grasp_plan_55

# 用真正的 PhysX FEM 软指执行；物体没有绑定或吸附到机器人
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/soft_grasp_plan_55/plan.json \
  --dt .001 --solver-iterations 80 --output data/runs/soft_planned_grasp_55

# 检查规划模型与仿真的安装坐标一致
envs/planning/.venv/bin/python benchmarks/test_soft_grasp_frames.py
```

试提时物体必须跟随上升至少 5 mm，否则执行器记录 `GRASP_REJECTED`，停止后续动作。
完整验证要求提起超过 60 mm，并在最后保持段仍高于原始位置 60 mm。
计划中的负载附加只用于碰撞规划；PhysX 中的物体始终是自由刚体。
手臂仍以给定关节轨迹驱动 FEM 底部节点，没有 FEM 反作用力矩反馈。

## 结果文件

- `data/generated/soft_grasp_plan_55/candidates.json`：候选参数和几何评分。
- `soft_franka.urdf` / `.srdf`：新 TCP、软指与碰撞代理。
- `plan.json` / `joints.npz`：确定的初始状态和分段轨迹。
- `data/runs/soft_planned_grasp_55/result.json`：真实软体执行结果。
- `trajectory.json`、各阶段相机 PNG / NPZ：物体跟随、形变及相机证据。

换物体后重新 export 和 plan，不能仅替换模型而继续执行旧轨迹。
若真实相机或安装件尺寸不同，先更新对应碰撞代理和 `T_hand_tcp`。
当前使用静态桌面与上述单物体；软指在运动中的重力形变和动态接触仍需用 FEM 执行验证，
不能把刚性碰撞规划的成功等同于抓取成功。

## 官方参考

- [cuRobo tool frames 与机器人模型](https://nvlabs.github.io/curobo/latest/api/curobo.kinematics.html)
- [cuRobo 抓取接近/夹持/提起分段规划](https://nvlabs.github.io/curobo/latest/getting-started/motion_planning.html)
- [机器人和被抓物体的碰撞球](https://nvlabs.github.io/curobo/latest/reference/sphere_fitting.html)
- [Lula 在 URDF 任意末端坐标求解 IK](https://docs.isaacsim.omniverse.nvidia.com/latest/manipulators/manipulators_lula_kinematics.html)
- [Dex-Net 2.0：对向接触候选与抓取质量](https://www.roboticsproceedings.org/rss13/p58.pdf)
