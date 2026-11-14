# X5 对称倾斜软指安装验证（2026-11-14）

> **路径变更（2026-11-14 重构）**：日常入口改为 `python scripts/run.py <command>`，实现位于 `src/soft_sim/`，生成物位于 `artifacts/`。本文保留写作时的目录与命令，当前命令见 [README](../README.md)。


当前默认选择 **0.75 倍软指、两侧各向内倾斜约 12°、独立楔形连接件**，配置版本为
`measured_wedge_v3`。角度来自源 STL 的外侧抓取斜面法向，不来自示意图。
原双 X5 的关节、滑块、限位、驱动及任务物体参数保持原值。

## 几何与方案选择

源 STL 中朝本指局部 +X 的外侧抓取面法向约为 `(0.978148, 0, 0.207912)`，
由 `atan2(nz,nx)` 得到 12.00000014°。两侧在原滑块坐标中施加镜像安装旋转。
最闭合位置不靠初始穿插：按整个网格的闭合方向支撑值设置根部间距，保留 0.20 mm 间隙。
0.75 倍时闭合根部半间距为 15.70782 mm；保留原 URDF 左右安装原点的微小不对称。

| 安装对照，0.75 倍 | 可见外侧抓取面间隙 P95 | 对向法向余弦下界 |
| --- | ---: | ---: |
| 旧平行安装 | 22.114 mm | 0.913545 |
| 内倾 6° | 11.350 mm | 0.978148 |
| 内倾 10° | 4.017 mm | 0.997564 |
| 内倾 12° | 0.200 mm | 约 1 |
| 内倾 14° | 3.393 mm | 0.997564 |
| 内倾 16° | 6.500 mm | 0.990268 |

间隙报告在共同高度/横向位置发射射线，只有首先命中源 STL 外侧抓取斜面的对应射线
用于抓取面统计。0.75 倍 12° 时为 279 对采样中的 120 对；格栅开孔和命中内壁的
射线另存覆盖率与腔体间隙，不当作连续实心接触面。完整网格仍全部参加碰撞检查。

倾斜底座的后角会靠近原腕部，因此根部沿原滑块前向前移 5.11868 mm，
使底面最低角位于原滑块前方 2 mm。楔块底面位于原滑块安装平面，顶面随软指倾斜；
底部采用不随软指缩放的 24 mm 原滑块安装座包络。第一次宽底楔块方案的结果保留在
`data/generated/x5_stack/exports/x5_stack0_measured_live_export`，不混作最终 v2 验证。

直接调整安装角度会得到同样的软指位姿，但原滑块没有已确认的斜安装面，需加工或更换
安装座；当前选择独立楔块来保留原滑块接口与运动学。这是仿真连接件包络，螺孔、紧固、
强度、公差及实际相机包装还需实物 CAD 标定；12×12×8 mm 相机外壳仍是随软指缩放的代理。

| 原滑块坐标下的单指完整外廓 | 原装 X5 | 新装配，0.75 倍 |
| --- | ---: | ---: |
| 前向外廓长度，含安装区 | 86.000 mm | 70.203 mm |
| 从滑块向前伸出 | 71.000 mm | 70.203 mm |
| 闭合方向外廓宽度 | 37.350 mm | 36.800 mm |
| 厚度方向外廓宽度 | 61.000 mm | 30.000 mm |

0.65 倍新装配前伸约 61.110 mm，0.8 倍约 74.750 mm。0.8 倍在倾斜并加上避腕部偏置后
已长于原指，因此本次选择 0.75。外廓尺寸更小不表示每个工作截面都包在原爪内，也不表示
堆叠不会推碰；任务中的几何和真实形变仍需分别验证。

对照图与机器可读报告：

- `data/generated/x5_mount/x5_mount_v3_audit/closed_comparison.png`、`installation.json`
- `data/generated/x5_mount/x5_measured_v3_finger_size_comparison/comparison.png`、`comparison.json`

## 静态行程与原驱动物理检查

每根滑块保留原 `[0, 0.044] m` 限位。静态扫描 45 个位置，检查左右指体、连接件、
相机外壳、腕部和原腕部相机。v3 的 0.65/0.75/0.8 倍均无非预期穿插。
0.75 倍左右完整组件在最闭合位置仍被 0.20 mm 中间平面间隙分隔；滑块只向外平移，
因此非对称开合也不会产生左右组件穿插。固定腕部障碍物的距离使用采样间距的
Lipschitz 下界补偿，0.75 倍完整连续行程的保守间隙下界约 3.347 mm。
同侧楔块顶面与指底是预期安装接触；不再忽略新指/楔块与 `link6` 的碰撞。

`data/generated/x5_stack/scenes/x5_stack0_measured_v3_native_assembly002` 是最终原 `stack_blocks_0` 场景诊断目录。
原驱动检查覆盖
四指原驱动闭合、闭合保持、张开、张开保持，5 秒，保存 50 个真实 FEM 状态：

- 所有采样单元 Jacobian 为正，最小约 0.897986。
- 最大固定区误差约 0.001659 mm。
- 保持阶段单滑块最大到位误差约 `2.38e-10 m`。
- `assembly_motion/collision_validation.json` 独立检查变形后的四面体边界与相机、楔块、
  腕部、桌面和原三块积木，50 帧未发现非预期表面穿插。
- 50 帧记录的相机相对本指安装矩阵最大变化约 `1.11e-16`；相机随实际滑块安装位姿更新。
- 同时刻的真实滑块/连接件安装误差为零；初始、全闭合和重新张开的 10 组相机近场深度复核全部通过。

形变碰撞复核是 0.1 秒采样检查，不是连续软体避碰证明，也不包含所有场景家具。
对向软指在真实闭合后的接触允许出现，不当作相机或连接件避碰失败。
装配诊断不评估任务成功，不写入 LeRobot 任务成功数据。引擎的小正体积单元诊断仍存在，
不以其日志条数代替真实单元体积检查，也未通过修改材料或放宽阈值绕过问题。
`physx_mesh_diagnostics.json` 核对到初始化的 110,456 条质量诊断与小正体积单元数完全一致，
重置后无该类新增诊断；另有 4 条 cooking 参数错误，尚未解决，不能写成引擎日志全部通过。

## 共享安装变换与任务规划

`x5_mount_geometry.py` 提供同一套测量、根部位姿和连接件网格；
`x5_soft_tool.py`、任务 FEM 固定区、指内相机、实际 Isaac 楔块/相机碰撞形状和规划 URDF
均使用它。两侧分别跟随原滑块实际位姿，支持非对称开合。任务元数据保存
`finger_installation`、`finger_in_slide_transforms` 以及逐帧实际安装/相机矩阵。
计划与实际安装配置不一致时拒绝执行；统一入口不再提供旧平行安装选项，旧计划只作为历史数据保留。

楔块与相机碰撞形状直接固定在原 `link8/link7` 刚体下，不另外驱动独立刚体。
相机显示外壳使用 9 mm 开口与后盖，光学中心保持原值；规划和 PhysX 碰撞仍共用
保守的 12×12×8 mm 包络。全部尺寸随软指缩放。镜头全视场的射线必须通过开口，
不会用隐藏图像内容或图像过滤解决遮挡。

记录中保存从原滑块物理张量视图读取的 `fixture_to_world`；安装误差在相同的步进前
时刻比较。步进后的原刚体已积分下一步，而 FEM 固定区使用步进前的规定目标；
该 1 ms 耦合偏移单独记录，不混作安装误差。全行程试验最大偏移约 0.03305 mm。

9 项规划/安装集成测试覆盖新旧比例、非对称滑块、相机、TCP、连接件世界顶点、
接触闭合量、过期源网格/比例/安装座参数拒绝以及外壳全视场通光。
共享几何 8 项、动作执行 6 项、挂载采集 1 项测试通过。

`data/generated/x5_stack/plans/x5_stack0_measured_v3_native_plan/plan.json` 使用本次真实导出的原场景，
规划摩擦筛选系数 0.51、单侧预压 4 mm，通过首块的接近、闭合、试提、抬升、
转移、放置、松爪、撤离和回位碰撞检查。抓取处刚性外廓到桌面约 2.044 mm；
该间隙不保证变形后仍通过，因此文件保留 `physical_grasp_validated=false`。
第一块路径通过不代表完整三块堆叠任务已完成。

早期 CPU 对照 `x5_pickup22_measured_offline_*` / `x5_stack0_measured_offline_*` 明确标记
从旧快照重新安装、未运行新物理；其中 v1 楔块与最终 v2 的配置不同，不能复用为最终验收。
所有旧平行安装记录保持原值。

## 复现

```bash
MPLCONFIGDIR=/tmp/soft-sim-mpl OPENBLAS_NUM_THREADS=1 \
envs/planning/.venv/bin/python benchmarks/check_x5_installation.py \
  --output data/generated/x5_mount/x5_mount_v3_audit

MPLCONFIGDIR=/tmp/soft-sim-mpl OPENBLAS_NUM_THREADS=1 \
envs/planning/.venv/bin/python benchmarks/test_soft_grasp_frames.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
envs/robodojo/.venv/bin/python benchmarks/robodojo_task_soft.py \
  --task stack_blocks --seed 0 --mode assembly --finger-scale .75 \
  --dt .001 --solver-iterations 80 \
  --vertex-velocity-damping 20 --texture-max-resolution 512 \
  --fem-mesh data/generated/assets/mesh/soft_finger_tetgen_4mm.npz \
  --output data/generated/x5_measured_NEW_assembly

MPLCONFIGDIR=/tmp/soft-sim-mpl OPENBLAS_NUM_THREADS=1 \
envs/planning/.venv/bin/python benchmarks/check_x5_assembly_motion.py \
  data/generated/x5_measured_NEW_assembly
```

## v2 抓取及相机问题记录

`data/generated/x5_pickup/measured/x5_pickup22_measured_v2_collect001` 使用旧 v2 安装、0.75 倍软指、
原 `general_pickup_22` 布局与物体属性，通过原驱动完成实际抓取。接触高度 54.75 mm，
规划摩擦筛选系数 0.51，单侧预压 8 mm；闭合总滑块参数约 32.633 mm。
12 mm 试提时物体抬升约 11.033 mm，最终约 136.996 mm，原 RewardManager 返回 1。

独立 LeRobot v3 数据位于 `recording/dataset`，268 帧、13.4 秒。该结果是一例原任务
物理成功，不代表随机布局成功率或完整堆叠成功；采集质量报告与任务成败分别记录。
268 帧三路视频、状态同步与 LeRobot v3 回读的数据完整性检查通过。

执行臂独立逐帧复核的最小 Jacobian 约为左 0.65745、右 0.67978，最大固定误差
约 0.002917 mm，相机刚性安装检查通过。整体稳定性仍不通过：未载形变超过既有
2 mm 阈值；保持段左指最大帧 P95 节点投影运动约 0.701 px，原始 RGB 最坏约
0.710 px，超过既有 0.5 px 阈值。`finger_motion_validation.json` 中的
`physics_initialization_valid=false` 在此表示未载位移阈值未通过；本次初始化正体积和
固定区检查分别保存在 `initial_fem_validation.json`，不能混淆。没有放宽阈值或过滤图像。

该运行另有 45 个相机帧的近场深度不通过：实心相机外壳挡住初始视线，且独立刚体
仅更新 USD 时未可靠跟随 PhysX/Fabric。v2 的相机验收为失败，先前基于重建位姿的
连接件避碰也不足以证明其真实移动。旧记录保持不变，不作为最终 v3 装配验收。
v3 通过镜头开口和直接固定到原滑块刚体修复，并保存实际物理位姿及逐帧深度检查。
