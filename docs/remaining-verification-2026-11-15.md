# 剩余验证记录（2026-11-15 晚）

本轮使用当前工作区代码、TPU 90A、80 次求解迭代、1 ms 时间步、节点阻尼 20、固定区消元、
`measured_wedge_v3` 0.75 倍安装和原机械臂位置驱动。任务判定、布局、物体属性未修改。
所有结论只适用于列出的 seed 与运行，不代表全布局成功率。

## 输入与回归

官方资产目录检查确认 54 个任务均有 seed 0 保存布局。已下载全部 54 份布局与对应选中物体模型，并补充随机变体引用的房间、材质及背景。
随机变体使用自身官方布局，不以基础任务的运行代替变体验证。
输入准备记录保存在 `assets/downloads/task_<task>_s0.json`，包含来源与 SHA-256。
总表见 [`final_preflight.json`](../artifacts/cache/remaining_verification_20261115/final_preflight.json)。
这里的 `ready` 表示布局、所选模型和显式环境资源存在，不涵盖辅助轨迹、动作语义或初始化兼容性。

本地与 4090 原始回归均为 unit 126、planning 11、dataset 5，通过且无 skip。
本轮修正 `align_blocks` 的三角尺标签：官方标签为 `target`，原适配器错误使用 `set_square`。
修正后两端 unit 127 项通过；新增单对象碰撞标量兼容与房间/材质/背景输入检查后，最终本地与 4090 均为 unit 130、planning 11、dataset 5，共 146 项通过且无 skip。
保留修正前失败日志及修正后实际复验结果。
布局预检和下载器现在同时检查/准备显式房间、材质包与 HDR 背景。随机环境补充清单见
`assets/downloads/verification_random_rooms_s0.json`、`verification_random_environment_s0.json`；
传送带普通房间见 `verification_conveyor_room_s0.json`。
该修正只处理目标解析，不代表推动动作或原任务已成功。

## 抓取与堆叠对照

- `verify_pickup_curobo_20261115`：`general_pickup` seed 22，当前默认 cuRobo，原奖励 1，
  实际抬升 **133.761 mm**。208 帧提交到显式诊断集合
  `artifacts/datasets/lerobot/diagnostic_pickup_curobo_20261115/`；全部数值状态回读成功、时间间隔
  50 ms、记录的最小 Jacobian 0.844102、最大固定区误差 0.002907 mm。因为本次显式保留诊断，
  未向正式成功任务集合追加；完整独立质量复核已执行，结论见下节。
- `goal_stack_blocks_s0_ind8_4090` 为用户提供的上一轮证据；本轮另运行
  `verify_stack_contact_20261115`，保留 148 帧失败诊断。实际夹爪开度从 88 mm 闭合至
  **18.8 mm**；试抬末端上移约 **11.98 mm**，方块高度不变。闭合及保持阶段末端位置跟踪误差
  约 0.011 mm，最大臂关节跟踪误差约 2e-5 rad，未发现指令未执行或 IK 目标坐标错误。
- 堆叠闭合后两指最大局部形变约 8.4 mm。原接触高度附近 2 mm 节点带的两侧最近节点间距约
  35.16 mm，方块宽约 35 mm。此为节点带几何观测，不是连续表面间隙或接触力测量，
  不能据此断言没有接触，也不能把原因定为摩擦不足。接触受力与保持能力仍待专项排查。
- 成功参考 `goal_90a_pickup_host2_20261115` 使用 mplib；新 cuRobo 对照也成功，故不能把
  堆叠失败笼统归因于更换规划器。两种物体、姿态和接触几何不同，仍需逐项比较。

数值证据与图像：[`对照目录`](../artifacts/cache/remaining_verification_20261115/)、
[`关节与末端`](../artifacts/cache/remaining_verification_20261115/stack_contact.json)、
[`局部形变`](../artifacts/cache/remaining_verification_20261115/deformation_comparison.json)、
[`外部相机对照`](../artifacts/cache/remaining_verification_20261115/contact_comparison.jpg)。

诊断录制曾因本轮过早调用数据读取器创建空数据目录而首次转换失败；保留原始帧，移除空目录后
从暂存恢复成功，未重跑或伪造帧。失败 episode 只在独立诊断集合，未混入正式成功数据。

## 质量复核边界

既有 90A 成功参考的独立完整复核已经执行，结论是任务成功、数据完整，但质量未全部通过。
FEM 体积、固定区、形变相机投影和采样碰撞检查通过；静态 STL 投影、部分保持阶段画面/投影
稳定性、初始化 cooking 检查未通过。见原运行的
[`acceptance.json`](../artifacts/runs/goal_90a_pickup_host2_20261115/reports/20261115T080124Z/acceptance.json)。
不能将这类已执行且失败的复核继续写成“未检查”，也不能写成“全部质量验收通过”。

本轮新 cuRobo 成功对照也完成了全部独立复核：208 帧数据完整、相机刚性安装、FEM 投影、
固定区、正单元体积和采样碰撞通过（非预期表面交叉 0）；STL 投影、保持阶段的投影/RGB
稳定性与 cooking setup 未通过，原日志有 4 条 cooking setup 错误。质量结论仍为失败。
见 [`本轮 acceptance.json`](../artifacts/runs/verify_pickup_curobo_20261115/reports/independent_spot/acceptance.json)。
单对象有效性标记回读为标量曾触发碰撞检查器索引异常；已修复并回归，重跑碰撞检查通过，
未修改录制数据或放宽碰撞条件。

## 动作流程的实现缺口

源码复核还发现，54 项存在适配器分支不等于 54 项具备任务所需的完整动作。
`placement()` 对 `press`、`close_lid`、`fold` 返回物体原位姿，执行端仍调用通用
`pickup_phases()`；尚未构成沿按钮行程按压、转动盖板关节或连续折叠衣物的专用动作。
`push` 也未建立专用持续接触推动阶段。相关任务即使解决初始化和抓取问题，仍需补齐动作能力。
见 [`adapters.py`](../src/soft_sim/tasks/adapters.py) 与 [`execution.py`](../src/soft_sim/execution.py)。

`imitate_sorting_sequence` 配方使用不存在的 `target0..4`，官方标签是 `t0..4` 与 `aim0..4`；
且原任务需要辅助臂演示、观察顺序及辅助轨迹，不能只改名后按固定序列装篮来宣称完成。
`play_tic_tac_toe` 引用 `chessboard`，官方对象标签为 `checkerboard`，还需正确处理对手与局面。
本轮保留这些失败证据，不将只改允许列表或名称作为完整适配。

此外，当前 cuRobo 适配器复用了上游机器人/桌面配置，但未把导出场景的任意障碍网格与软指
装配碰撞体完整加入其世界碰撞模型。`plan.json` 的 `collision_checked` 不能独自证明对全部
真实场景避碰；全场景及携物碰撞约束的覆盖仍需补齐或用独立检查确认。

## 并行执行

用户已授权 4090 按资源容量并行。实际两路 Isaac 进程同时运行时，显存约 **14.9 GB**，
GPU 利用率 **99%**，无显存不足退出。本地 8GB 卡维持一路。每路批次内部串行，独立任务
使用独立 batch/run ID，同一任务数据集不并发写入。此观测证明可并发运行，未证明吞吐提升比例。

## 全量汇总与最终状态

本轮所有队列（含 `verify_random_{a,b,c}_environment_fixed_20261115` 12 个随机变体与 `verify_conveyor_environment_fixed_20261115` 2 个传送带任务）均已结束；本地与 `4090` 均无残留活跃仿真进程，亦无失效 `running` 冒充完成结果。由 [`summarize.py`](../artifacts/cache/remaining_verification_20261115/summarize.py) 生成的总表见 [`latest_results.json`](../artifacts/cache/remaining_verification_20261115/latest_results.json)：

- **总覆盖（`overall_attempted`）**：在 `artifacts/runs` 与 `artifacts/batches` 的 `verify*`、`goal*` 命名范围内，54 / 54（`not_attempted`: 0；由 53 项 seed 0 与 `general_pickup` seed 22 组成），含本轮 `verify*`（42 项）与本轮稍早同代码基线 `goal_*` cuRobo 运行（12 项）；`task_success` 保留三态（`true`: 1、`false`: 43、`null`: 10），`quality_valid` 保留三态（`false`: 1、`null`: 53）：
  - `success`：1 项（`general_pickup` seed 22，`task_success=true`、`quality_valid=false`）
  - `task_failed`：23 项（主要为 `probe_lift` 物体未随动、夹持滑移或专用动作未实现）
  - `planning_failed`：20 项（无满足几何/高度的抓取候选或 `transfer`/`lift` 阶段可达性失败）
  - `runtime_error`：10 项（布料/液体导出 `NoneType.get_world_pose` 5 项、`imitate_sorting_sequence`/`play_tic_tac_toe` 官方标签不匹配 2 项、`make_kong` 缺官方 `1.pkl` 1 项、`sort_nesting_dolls_by_size` 12.0s FEM 采样保护退出 1 项、`make_toast_random` 场景导出 `No triangle geometry: .../toaster_4_15/root_joint` 1 项）
- **本轮队列（`this_round_attempted`）**：42 / 54（`success` 1、`task_failed` 17、`planning_failed` 15、`runtime_error` 9；其余 12 项已在 `goal_curobo_*` / `goal_cover_blocks_s0_ind8` 完成尝试）。
- **关节 `dof_properties` CUDA→NumPy 兼容修复与复验（`verify_articulation_compat_20261115`）**：
  - 根因：`third_party/RoboDojo/env/scene_manager/objects/articulation.py:235` 读取 `SingleArticulation.dof_properties` 时，GPU pipeline 下 `_articulation_view.get_max_efforts()[0]` 等返回 CUDA `torch.Tensor`，直接赋给 NumPy 结构化数组触发 `TypeError`。
  - 修复：在 [`robodojo.py`](../src/soft_sim/sim/robodojo.py) 新增局部幂等兼容层 `ensure_articulation_dof_properties_compat()`（仅对 `dof_properties` 做 host CPU 拷贝，保留原 DOF 限位/力矩/刚度/阻尼与设备状态，不改 `site-packages`/`third_party`/全局 `torch`），并附单元测试 [`test_robodojo_compat.py`](../tests/unit/test_robodojo_compat.py)。
  - 复验结果：全部 7 项均已消除 `dof_properties` 初始化 `TypeError`：`fill_egg_holder`、`press_by_number`、`swap_blocks`、`store_laptop_and_headphones_random` 4 项进入物理执行（`task_failed`）；`make_toast`、`store_laptop_and_headphones` 2 项进入规划（`planning_failed`）；`make_toast_random` 通过 `articulation.initialize()` 后在场景导出阶段遇 `No triangle geometry: .../toaster_4_15/root_joint`（`runtime_error`）。
- **明确未解决项**：不能宣称全任务适配完成或质量全部通过；`general_pickup` 需解决 STL 投影、保持稳定性与 cooking setup 警告；其余任务需分别补齐专用动作（按压/关盖/折叠/推动/倾倒）、接触保持能力、双臂切换/工作空间规划及布料/流体/随机 `root_joint` 场景导出兼容。

各任务最新结果、证据路径与质量标签已同步至 [`task-adaptation-status.md`](task-adaptation-status.md) 与 [`latest_results.json`](../artifacts/cache/remaining_verification_20261115/latest_results.json)。
