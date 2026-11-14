# RoboDojo 原双 X5 的软指移植

> **路径变更（2026-11-14 重构）**：日常入口改为 `python scripts/run.py <command>`，实现位于 `src/soft_sim/`，生成物位于 `artifacts/`。本文保留写作时的目录与命令，当前命令见 [README](../README.md)。


**2026-11-14 更新：**当前 X5 默认采用 0.75 倍、网格测得的双侧内倾约 12°及楔形连接件，
详见 [倾斜安装、完整行程及物理复核](x5-tilted-installation-2026-11-14.md)。
以下旧实验的 0.65/0.75 比例结果保留原平行安装配置；统一入口已移除
`--mount-profile parallel_legacy`，这些旧运行只作为历史证据，不能混作新安装验证。

唯一入口是 `benchmarks/robodojo_task_soft.py`，机器人固定为原双 X5，不再提供 Franka 路径。
保留官方任务类、双 X5 基座、六轴机械臂、两侧滑块关节、布局和奖励管理器；隐藏原
`link7/link8` 的指体并关闭其碰撞，在两只机械臂上各安装一对 FEM 软指。
原生腕部相机的实体保留；采集使用执行臂的双指内相机和外部相机。
IsaacLab 无窗口模式的 `render()` 可能不刷新 Fabric；采样现在先显式调用 `forward()`，
再进行零时间 Replicator 渲染，避免“关节已运动、画面里的机械臂仍在初始位置”。

## 等比缩放

`--finger-scale` 控制软指尺寸，X5 当前默认 0.75。下面的尺寸/偏置说明来自旧平行安装。
源 STL 和上游机器人文件均不覆盖。

| 比例 | 软指包围盒，mm | 相对完整软指的体积 |
| --- | --- | --- |
| 1 | 40 × 40 × 88.18 | 1 |
| 0.8 | 32 × 32 × 70.544 | 0.512 |
| 0.65 | 26 × 26 × 57.317 | 0.274625 |
| 0.5 | 20 × 20 × 44.09 | 0.125 |

原 X5 的滑块根部在 `link6` 前方 86.57 mm；官方抓取 TCP 在前方 145 mm，差为
58.43 mm。原指体网格最前端相对滑块为 71 mm，所以 0.8 倍更接近原几何长度，
0.65 倍更接近原抓取 TCP 的距离，并留出更多桌面间隙。TCP 偏置不等于网格极限尺寸。

同步缩放四面体节点、源 STL 单位、显示表面、固定高度、相机安装位置、近裁剪面及
转接件/指内相机的碰撞代理；拓扑、材料密度、75A 对应的弹性参数及相机视场角保留。
质量因此按比例三次方变化。生成文件保存原始网格元数据和缩放比例。

模型以原生 `link8/link7` 实际位姿驱动两个软指底座，支持不对称开合。原机器人滑轨
和腕部尺寸不缩放。当前安装使用独立楔块，两侧各向内倾斜约 12°，由
`x5_mount_geometry.py` 统一生成；旧平行安装的 10 mm 内移只保留在历史记录中。
相机外壳也采用等比代理，不表示实际相机硬件可等比制造。

均匀缩小、材料不变时，线弹性和几何相似的条件下结构力/位移刚度约按比例变化，
因此不能沿用原尺寸的闭合量、夹持力或成功率。规划使用缩放后的真实接触截面与
完整软指外廓，重新搜索接触点、倾角和开口，再进行物理试提。

## 运行

当前安装与采集命令见 [根 README](../README.md) 和
[倾斜安装验证](x5-tilted-installation-2026-11-14.md)。基本形式：

```bash
envs/robodojo/.venv/bin/python scripts/download_benchmark_assets.py --prefix Assets/Robots/x5

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
envs/robodojo/.venv/bin/python benchmarks/robodojo_task_soft.py \
  --task general_pickup --seed 22 --mode collect --finger-scale .75 \
  --dt .001 --solver-iterations 80 --vertex-velocity-damping 20 \
  --texture-max-resolution 512 --fem-mesh data/generated/assets/mesh/soft_finger_tetgen_4mm.npz \
  --dataset-root data/datasets/robodojo_x5_75a_general_pickup22 \
  --output data/generated/x5_pickup22_NEW
```

任务引用的目标和杂物资产需预先下载。每次采集使用新的输出目录。
原生 X5 默认选择贴合 4 mm 参数网格和 20 s⁻¹ 节点阻尼；网格缺失会明确要求先运行
`envs/meshing/.venv/bin/python scripts/prepare_mesh.py --method tetgen --pitch .004 --output data/generated/assets/mesh/soft_finger_tetgen_4mm.npz`。
8GB GPU 的单环境默认使用较小 PhysX 缓冲区、限制纹理分辨率、只分配使用中的相机产品；
这些设置保存在 `scene.json`，原布局、物体网格、质量和摩擦不改变。

导出包含三路预览、`mount_validation.json`、`initial_fem.npz`、初始单元 Jacobian/固定误差
检查、物体几何和凸包障碍物。其他机械臂的各个连杆及变形软指也作为停放障碍物。
目标几何使用实际物体位姿修正 USD 变换；不把 Fabric 下未更新的 USD 当作实时状态。
房间墙体仍未加入规划障碍物。闭合过程分段检查自碰撞与环境碰撞，仅允许软指接触目标。

## 数据与验证范围

LeRobot v3 采集保存两臂合计 16 个关节的实际位置、速度和指令位置，名字按
`robot0/joint1…joint8, robot1/joint1…joint8` 排列。原机械臂关节为 rad、滑块为 m。
当前只驱动 `--active-arm`（默认右臂 1）执行任务，另一臂保持停放指令。
相机与 FEM 详细状态来自执行臂；`active_arm`、颜色、75A 材料和比例写入标定元数据。
公共数据集的来源元数据还嵌入校验过 SHA256 的原始布局，保留目标及杂物的原质量、摩擦和位姿。
应使用独立数据集，不能追加到旧 Franka 的 9 维动作数据中。

```bash
envs/planning/.venv/bin/python benchmarks/test_soft_grasp_frames.py
envs/dataset/.venv/bin/python -m unittest discover -s tests -p test_lerobot_recording.py
envs/dataset/.venv/bin/python benchmarks/validate_rgb_episode.py OUTPUT/recording
```

当前执行器实现 `general_pickup` 的接近、闭合、保持、12 mm 试提、抬升和保持，
并使用原任务抬高超过 10 cm 的成功条件。`--mode collect` 会从实时场景调用运动规划器；
`stack_blocks` 已新增两次抓取、携物转移、放置、松爪、撤离和回位，第二次抓取前重新导出
实际场景和规划，物理通过情况见后文。对齐、插入等其他任务仍只支持导出。FEM 采用规定底座运动，未反馈
反作用力矩；任务硬件已改变，结果不作为原装 benchmark 得分。

已实跑第 0 个原布局的双 X5、0.65 倍贴合 FEM 和三路相机；安装位置与独立 URDF FK
最大差约 0.0585 mm，旋转矩阵最大元素差约 0.000302，均通过预设检查。
该布局的薄剪刀在已试的三处接触高度（缩放前 55/73/80 mm）均无通过几何筛选的候选，
没有执行未通过规划的抓取，也未把它标成成功数据。

第 22 个原布局已通过四指初始单元检查：步长 0.25 ms、120 次迭代、节点速度阻尼
20 s⁻¹ 时，各指最小 Jacobian 为 0.757、0.787、0.412、0.724，最大固定误差约 0.0021 mm。
这是待标定的数值稳定参数；材料仍为 75A，未改变其杨氏模量。水平待机时存在明显重力
形变，最大节点位移约 68.8 mm，不能将刚性静态外廓规划等同于软体执行碰撞保证。
1 ms、80 次迭代的试验出现翻转单元，不能用来生成有效训练数据。

4 mm 参数贴合网格为 8491 节点、33417 四面体，源表面和材料相同，体积相对 STL 误差
约 +0.000745%。在 1 ms、80 次迭代、20 s⁻¹ 节点阻尼下，四指初始最小 Jacobian
分别为 0.946、0.926、0.946、0.924，固定误差均小于 0.0014 mm，最大重力位移约 2.86 mm。
网格间重力形变差异明显，尚未达到网格收敛，不能将该仿真当作已标定的实物材料响应。
采集还逐帧检查执行臂的单元翻转和固定区误差；不通过的顺序流保留为 incomplete，
不自动提交到 LeRobot 训练数据集。录制索引在首帧及每秒检查点原子写入，减少中断损失。

第 22 个布局的三组接触高度均得到碰撞规划；73 mm 接触带在缩放后为 47.45 mm，
掌部到 TCP 为 134.02 mm，规划使用 88 mm 滑块总位移和约 27.81 mm 闭合位移。
目标网格宽约 58.7 × 81.3 × 46.9 mm。物理抓取结果应查看运行的 `result.json`，
不能用计划中的 `status=planned` 宣称抓取成功。

接触规划已从凸包改为实际三角表面，并用两个接触点的中点作为 TCP。第 22 个布局
的一个实际接触截面宽约 48.83 mm；旧包围盒中心沿闭合轴偏了约 9.59 mm，会出现
一侧未接触、另一侧过度弯曲。新方案同时修正横向中心和闭合量，避免将其误判为
单纯的材料过软。凸包仍用于保守避碰；已有的旧计划文件不会自动获得这项修正。

## 已完成的采集与验收（2026-11-13）

`data/generated/x5_pickup/episodes/x5_pickup22_065_episode007` 使用原 `general_pickup_22` 布局、双 X5、
0.65 倍软指、75A 材料；目标仍为原 40 g、摩擦系数 0.32 的玩具。最终方案使用实际
接触中点 TCP、8 mm 单侧预压，闭合参数约 10.524 mm。12 mm 试提时物体跟随上升
约 10.84 mm；完整抬升后约 132.41 mm，最后 1 秒全部 20 帧的最低抬升约 132.42 mm。
原任务抬升超过 10 cm 的条件通过。这是一次实例验证，不是随机任务成功率或原装 benchmark 得分。

公共数据集为 `data/datasets/robodojo_x5_75a_scale065_4mm`，使用官方 LeRobot v3 写入器，
共 4 个完整 episode、1030 帧。`episode_index=3` 为成功实例，302 帧、15.1 秒；
索引 0、1、2 是保留的失败实例。成功/失败索引集中保存在 `meta/simulation/collection_summary.json`。
`meta/simulation` 的原始布局快照保留质量、摩擦、位姿及来源 SHA256。

成功实例的数据完整性、相机刚性安装、全部记录帧的固定区误差与正单元体积通过：
最大固定误差约 0.00294 mm，左右所有帧最小 Jacobian 约 0.592 / 0.625。
原运动验收的总体 `valid` 仍为 false：水平未载重力位移约 2.9 mm，超过旧 2 mm 阈值；
最终保持段左右最大帧 P95 节点投影运动约 0.630 / 0.737 px，超过旧 0.5 px 阈值。
没有修改阈值或过滤原始图像。任务成功、数据完整性通过，不等于全部运动/画面稳定验收通过。
原始报告位于该 episode 的 `recording/validation.json`、`finger_motion_validation.json`，
独立 FEM 复核位于 `data/generated/x5_pickup/episodes/x5_pickup22_065_episode007_audit/clamp_deformation.json`。

另已加载 `stack_blocks_0` 原布局及双 X5 替换软指，初始 FEM 和安装坐标检查通过：
`data/generated/x5_stack/scenes/x5_stack_blocks_065_fastshutdown`。该结果只验证装配与场景导出；后续完整动作实现与物理验证见下节。
批量入口在录制/转换完成后使用 Kit 的 `skip_cleanup` 退出方式，避免旧软体张量视图卡在场景清理；
上述堆叠场景导出及退出已实际返回 0。

## 原驱动执行、堆叠与指内视角复核（继续验证）

[当前目标](goal.md)记录：以运动规划收集数据，保留原双 X5、场景和任务判定。
尺寸比较基准是原装 X5，而不是把 0.65 当作原尺寸。原指体含后端安装区长约 86 mm，
从滑块向前伸约 71 mm。0.75 倍软指前伸约 66.14 mm，计入转接件后长约 78.59 mm；
0.8 倍分别约 70.54 / 83.82 mm。两种结构的截面不同：例如前伸 50 mm 处，原指约
7.45 × 21.66 mm，0.75 软指约 10.66 × 20.90 mm。长度接近不能当作工作区各方向都更小。
尺寸复核保存在 `data/generated/assets/mesh/x5_factory_soft_size_comparison.json`、`x5_factory_soft_sections.json`。
可用 `envs/planning/.venv/bin/python benchmarks/compare_x5_finger_sizes.py` 重建同一安装坐标下的
原爪／软指外廓对照；`data/generated/x5_mount/x5_finger_size_comparison/comparison.png` 中灰色为原指，
橙色为软指与转接件，坐标单位均为 mm。该图比较未加载静态几何，不代表运动中的避碰余量。

默认 `--joint-control position` 通过原驱动发送位置和速度目标，保留原关节增益及力矩限制。
`--joint-control kinematic` 仅用于诊断；旧 `episode007` 的每步关节改写属于该类回放。
新增 `x5_pickup22_065_position_drive002` 已通过原驱动完成 `general_pickup_22`：
原 RewardManager 返回 1，末端保持后目标抬升约 132.30 mm；已保存 302 帧、15.1 秒的
独立 LeRobot v3 数据，位于该运行的 `recording/dataset`。这仍是单布局实例，不是全任务成功率。

该实例四指逐帧检查未发现真实翻转或固定区滑移，执行臂独立复核的最小 Jacobian
约 0.585 / 0.588，最大固定误差约 0.00294 mm。保持段节点投影运动最坏约 1.33 / 1.18 px，
原始视频光流也超过旧 0.5 px 阈值；总体质量验收仍不通过，不能写成全部稳定性已解决。

`check_fem_camera_projection.py` 使用记录帧的体网格边界三角面，独立 CPU 射线检查指内
近场深度，避免把真实重力形变与未变形 STL 比较。上述实例全部 302 帧通过；报告为
`recording/fem_camera_projection_validation.json`。检查范围为光学深度 1–20 mm 的指体，
不替代 RGB 稳定性或物体/背景同步检查；旧 STL 报告保持不变，仍可查看其失败项。
另对接近、保持、试提与最终保持帧抽查实际物体表面深度，结果保存在
`object_depth_sync_probe.json`：试提及抬升后当前物体位姿的可见射线一致率为 100%，
冻结在初始物体位姿的对照为 0%，用于排除物体画面仍停在初始位置。

旧 PhysX `updateNodeInverseVolumes` 的报错条件包含正行列式小于等于 `1e-9` 的单元，
见 [SDK 源码](https://github.com/NVIDIA-Omniverse/PhysX/blob/107.3-physx-5.6.1/physx/source/physxextensions/src/ExtDeformableVolumeExt.cpp)。
`check_physx_mesh_diagnostics.py` 对旧成功运行核对得到每指 28,381 个小正体积单元，
四指合计与初始化的 113,524 条信息完全一致。它只解释这一类初始化质量诊断，不清除
其他 cooking 报错，也不替代实际单元检查。只读 tensor 逆静止矩阵报告还具有单位敏感性；
补偿局部坐标的试验未改善真实形变，未作为生产修复采用，默认仍以米建模。

0.75 倍初次完整装配曾出现单侧翻转；仅增加迭代数未可靠解决。现将未加载软指的初始化
放在任务、机器人和物体重置完成后，按实际安装位姿恢复无应变节点及零速度，再自由稳定。
该操作只在采集前执行，不在抓取中纠正物理节点。`x5_stack0_scale075_reset_export`
已验证四指最小 Jacobian >0.93、最大固定误差 <0.0005 mm。

`x5_stack0_scale075_position002` 已完成第一次堆叠抓取尝试，保留 264 帧、13.2 秒数据，
公共堆叠数据集 `robodojo_x5_75a_scale075_stack_blocks` 的 episode 0 按原任务记录为失败：
首块积木未随 12 mm 试提上升，原 reward / score 均为 0，尚未执行放置。
四指逐帧检查未发现真实单元翻转。`contact_audit/contact_diagnosis.json` 显示闭合保持时
两滑块实际位置与指令相差不到 0.001 mm，双指节点距原 35 mm、250 g 积木表面约
0.11–0.13 mm，最大形变约 7.9 / 8.1 mm；因此不能将失败简单归因为未闭合或完全未接触，
也不能仅凭节点距离推断夹持力足够。当前继续对比接触位置和闭合预压，未改变积木属性或成功条件。

`x5_stack0_scale075_position003` 将接触高度从 54.75 mm 改为 41.25 mm，单侧预压从
4 mm 增至 8 mm，并采用避碰通过的倾斜抓取。该尝试也失败：最终试提约 0.29 mm，
原 reward / score 均为 0；episode 1 保留 257 帧、12.85 秒。保持时最大形变约
13 mm，试提末约 15 mm，四指在线检查和执行臂独立逐帧体积复核均未发现真实翻转。
因此当前问题包括承载抓持未通过，不能归结为软指形变太小。

可选 `--static-camera-probe-frames 20` 在零物理步下保存连续原始 RGB 和诊断报告，
不写入训练数据。`x5_stack0_scale075_static_camera_probe` 已验证软指节点、目标位姿与
相机矩阵变化均为 0；19 对相邻指内画面的最大 P95 光流约 0.066 / 0.097 px。
外部相机约 0.530 px，说明其渲染本身也有变化；该初始姿态对照不代表接触或执行中的稳定性已通过。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
envs/robodojo/.venv/bin/python benchmarks/robodojo_task_soft.py \
  --task stack_blocks --seed 0 --mode collect --finger-scale .75 \
  --dt .001 --solver-iterations 80 --vertex-velocity-damping 20 \
  --texture-max-resolution 512 --fem-mesh data/generated/assets/mesh/soft_finger_tetgen_4mm.npz \
  --indent .004 --dataset-root data/datasets/robodojo_x5_75a_scale075_stack_blocks \
  --output data/generated/x5_stack0_scale075_NEW
```

堆叠使用原任务注册的叠放与回位条件，以及原分段评分；规划成功不是任务成功。
放开后，已放积木重新作为普通障碍物参与撤离与回位规划。记录增加 `operation_index`
和所有任务物体的位姿，验收只比较同一次保持段中的相邻帧。不同物体数对应不同特征形状，
应按任务分开数据集；最终写入器自动维护 `meta/simulation/collection_summary.json` 的成败索引。
实际物理、传感器或转换错误保留为 incomplete，并保存 `run_failure.json` 和非零退出码。
