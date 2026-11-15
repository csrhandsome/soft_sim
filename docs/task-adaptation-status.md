# RoboDojo 任务与软指适配清单

核对日期：2026-11-15。以下为本地固定版本快照与当前代码的状态，中文名称为便于阅读的译名，任务 ID 为准。

晚间补验与全量汇总以[剩余验证记录](remaining-verification-2026-11-15.md)和 [`latest_results.json`](../artifacts/cache/remaining_verification_20261115/latest_results.json) 为准：已核实官方 54 项均有 seed 0 布局，补齐全部 42 个基础任务 seed 0 模型及 12 个随机变体、2 个传送带任务的房间/材质/HDR 资源；在 `verify*` 与 `goal*` 命名范围内，54 项均已有至少一次实际运行记录（`overall_attempted` 54/54，由 53 项 seed 0 与 `general_pickup` seed 22 组成：`success` 1、`task_failed` 23、`planning_failed` 20、`runtime_error` 10；`task_success` 三态：`true` 1、`false` 43、`null` 10；`quality_valid` 三态：`false` 1、`null` 53）。代码复核发现按压、关盖、折叠、推动、倾倒等仍缺专用动作，不能把“动作代码已接入”或“已执行”理解为完整适配或质量通过。

基础仿真任务 42 个；额外随机变体 12 个，共 54 套任务类与配置。基础任务与随机变体均已在自有软指执行器中接入动作代码（42 + 12 = 54）；完整物理适配与成功证据仍分别记录。18 个真机任务不属于此仿真仓库的已适配范围。

“已接入”表示存在任务动作编排与原任务成功判定的调用，不代表已验证完成；“未适配”表示没有相应软指动作执行流程，不能通过修改允许任务列表冒充支持。后续 YAML 批量采集入口已通过 `configs/collection.yaml` 与 `python scripts/run.py collect-batch` 接入。

默认材料为 90A。规划器已优先接入 RoboDojo 同款 `CuroboPlanner`（[`curobo.py`](../src/soft_sim/planning/curobo.py)，保留 `mplib` 作为回退）与共享候选生成器（[`candidates.py`](../src/soft_sim/planning/candidates.py)）。本轮 `general_pickup` seed 22 新增 `cuRobo` 对照通过原任务判定（`task_success=true`，抬升 133.761 mm，208 帧诊断集合完整回读），但独立质量复核 `quality_valid=false`（STL 投影、保持稳定性与 cooking setup 未通过）；不能据此宣称其他任务或全部布局成功。

| 维度 | 任务 ID | 任务 | 软指适配 / 执行验证（最新证据） | 本地保存布局数 |
| --- | --- | --- | --- | ---: |
| 泛化 | `arrange_largest_number` | 排列最大数字 | `task_failed`：`cuRobo` seed 0 执行，`digit_1` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/goal_curobo_8tasks_s0_20261115_arrange_largest_number_s0_a0/manifest.json)） | 30 |
| 泛化 | `fold_clothes` | 叠衣服 | `runtime_error`：seed 0 场景导出 `NoneType.get_world_pose`；折叠专用动作缺失（[`manifest.json`](../artifacts/runs/verify_parallel_b_20261115_fold_clothes_s0_a0/manifest.json)） | 55 |
| 泛化 | `hang_mugs` | 挂杯子 | `planning_failed`：`cuRobo` seed 0 `pregrasp/approach/probe` 通过，`lift` 处规划失败（[`manifest.json`](../artifacts/runs/goal_curobo_5tasks_s0_20261115_hang_mugs_s0_a0/manifest.json)） | 30 |
| 泛化 | `make_toast` | 制作吐司 | `planning_failed`：已消除 `dof_properties` 初始化异常并完成场景导出，`cuRobo` seed 0 规划失败；按压动作仍缺（[`manifest.json`](../artifacts/runs/verify_articulation_compat_20261115_4090_a_make_toast_s0_a0/manifest.json)） | 70 |
| 泛化 | `pack_objects_into_box` | 物体装箱 | `task_failed`：`cuRobo` seed 0 执行，`box` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/goal_curobo_8tasks_s0_20261115_pack_objects_into_box_s0_a0/manifest.json)） | 1 |
| 泛化 | `pour_liquid_into_cup` | 倒液体入杯 | `runtime_error`：seed 0 场景导出 `NoneType.get_world_pose`；倾倒专用动作缺失（[`manifest.json`](../artifacts/runs/verify_parallel_f_20261115_pour_liquid_into_cup_s0_a0/manifest.json)） | 1 |
| 泛化 | `push_T` | 推动 T 形物体 | `task_failed`：`cuRobo` seed 0 执行，`t` `probe_lift` 未随动；持续推动专用动作缺失（[`manifest.json`](../artifacts/runs/verify_parallel_e_20261115_push_T_s0_a0/manifest.json)） | 1 |
| 泛化 | `sort_nesting_dolls_by_size` | 按大小排列套娃 | `runtime_error`：`cuRobo` seed 0 执行至 `transfer`，FEM 采样保护拒绝退出（[`manifest.json`](../artifacts/runs/goal_curobo_8tasks_s0_20261115_sort_nesting_dolls_by_size_s0_a0/manifest.json)） | 1 |
| 泛化 | `stack_blocks` | 堆叠积木 | `task_failed`：90A seed 0 `cuRobo` `indent=0.008` 闭合至 18.8 mm、末端上移 11.98 mm，方块未随动（[`manifest.json`](../artifacts/runs/verify_stack_contact_20261115/manifest.json)） | 1 |
| 泛化 | `stack_bowls` | 叠碗 | `planning_failed`：`cuRobo` seed 0 三个接触高度均无抓取候选；需补碗沿抓取（[`manifest.json`](../artifacts/runs/verify_bowls_curobo_20261115/manifest.json)） | 1 |
| 泛化 | `store_laptop_and_headphones` | 收纳电脑和耳机 | `planning_failed`：已消除 `dof_properties` 初始化异常并完成场景导出，`cuRobo` seed 0 规划失败（[`manifest.json`](../artifacts/runs/verify_articulation_compat_20261115_4090_b_store_laptop_and_headphones_s0_a0/manifest.json)） | 1 |
| 泛化 | `sweep_blocks` | 清扫积木 | `planning_failed`：`cuRobo` seed 0 抓取规划失败（[`manifest.json`](../artifacts/runs/verify_parallel_f_20261115_sweep_blocks_s0_a0/manifest.json)） | 1 |
| 记忆 | `cover_blocks` | 盖住积木 | `task_failed`：`cuRobo` seed 0 `indent=0.008` 完整执行但原任务失败（[`manifest.json`](../artifacts/runs/goal_cover_blocks_s0_ind8/manifest.json)） | 55 |
| 记忆 | `imitate_sorting_sequence` | 模仿分类顺序 | `runtime_error`：seed 0 配方解析失败（`target0..4` 与官方 `t0..4`/`aim0..4` 不符）；演示与记忆时序待接入（[`summary.json`](../artifacts/batches/verify_parallel_a_20261115/summary.json)） | 65 |
| 记忆 | `match_and_pick_from_conveyor` | 匹配并抓取传送带物体 | `planning_failed`：补齐普通房间后 `cuRobo` seed 0 规划失败（[`manifest.json`](../artifacts/runs/verify_conveyor_environment_fixed_20261115_match_and_pick_from_conveyor_s0_a0/manifest.json)） | 1 |
| 记忆 | `press_by_number` | 按数字按键 | `task_failed`：已消除 `dof_properties` 初始化异常，seed 0 执行但原任务失败；按压专用动作缺失（[`manifest.json`](../artifacts/runs/verify_articulation_compat_20261115_4090_a_press_by_number_s0_a0/manifest.json)） | 1 |
| 记忆 | `swap_T` | 交换 T 形物体 | `task_failed`：`cuRobo` seed 0 执行，`t0` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/goal_curobo_8tasks_s0_20261115_swap_T_s0_a0/manifest.json)） | 1 |
| 记忆 | `swap_blocks` | 交换积木 | `task_failed`：已消除 `dof_properties` 初始化异常，seed 0 执行但原任务失败（[`manifest.json`](../artifacts/runs/verify_articulation_compat_20261115_4090_b_swap_blocks_s0_a0/manifest.json)） | 1 |
| 长程 | `classify_objects` | 物体分类 | `planning_failed`：`cuRobo` seed 0 `pregrasp/approach/probe/lift` 通过，`transfer` 规划失败（[`manifest.json`](../artifacts/runs/goal_curobo_8tasks_s0_20261115_classify_objects_s0_a0/manifest.json)） | 55 |
| 长程 | `fill_egg_holder` | 装满蛋托 | `task_failed`：已消除 `dof_properties` 初始化异常，seed 0 执行但原任务失败；关盖动作仍缺（[`manifest.json`](../artifacts/runs/verify_articulation_compat_20261115_4090_a_fill_egg_holder_s0_a0/manifest.json)） | 65 |
| 长程 | `fill_pen_holder` | 装满笔筒 | `planning_failed`：`cuRobo` seed 0 目标位于远端极限，规划失败（[`manifest.json`](../artifacts/runs/goal_curobo_5tasks_s0_20261115_fill_pen_holder_s0_a0/manifest.json)） | 55 |
| 长程 | `make_kong` | 麻将杠牌 | `runtime_error`：seed 0 初始化奖励缺官方辅助臂轨迹 `Assets/Traj/RoboDojo/make_kong/1.pkl`（[`manifest.json`](../artifacts/runs/verify_remaining_4090_20261115_make_kong_s0_a0/manifest.json)） | 65 |
| 长程 | `organize_table` | 整理桌面 | `task_failed`：`cuRobo` seed 0 执行，`keyboard` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/verify_parallel_c_20261115_organize_table_s0_a0/manifest.json)） | 1 |
| 长程 | `play_stacking_toy` | 套叠玩具 | `planning_failed`：`cuRobo` seed 0 抓取规划失败（[`manifest.json`](../artifacts/runs/verify_parallel_c_20261115_play_stacking_toy_s0_a0/manifest.json)） | 1 |
| 长程 | `play_tic_tac_toe` | 井字棋 | `runtime_error`：seed 0 配方解析失败（缺 `chessboard`，官方为 `checkerboard`）；需处理对手与局面（[`summary.json`](../artifacts/batches/verify_parallel_e_20261115/summary.json)） | 1 |
| 长程 | `put_bottles_into_dustbin` | 将瓶子放入垃圾桶 | `planning_failed`：`cuRobo` seed 0 抓取/转移规划失败（[`manifest.json`](../artifacts/runs/verify_remaining_local_20261115_put_bottles_into_dustbin_s0_a0/manifest.json)） | 1 |
| 精细操作 | `build_tower` | 搭塔 | `planning_failed`：`cuRobo` seed 0 `pregrasp/approach/probe/lift` 通过，`transfer` 规划失败（[`manifest.json`](../artifacts/runs/goal_curobo_8tasks_s0_20261115_build_tower_s0_a0/manifest.json)） | 55 |
| 精细操作 | `deposit_coin` | 投币 | `planning_failed`：`cuRobo` seed 0 `pregrasp/approach/probe/lift` 通过，跨桌单臂 `transfer` 规划失败（[`manifest.json`](../artifacts/runs/goal_curobo_5tasks_s0_20261115_deposit_coin_s0_a0/manifest.json)） | 60 |
| 精细操作 | `fasten_screws` | 拧紧螺丝 | `task_failed`：`cuRobo` seed 0 执行，首个螺母 `nut0` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/verify_remaining_4090_20261115_fasten_screws_s0_a0/manifest.json)） | 55 |
| 精细操作 | `insert_key` | 插钥匙 | `planning_failed`：`cuRobo` seed 0 抓取规划失败；完整插接/交接动作仍未验证（[`manifest.json`](../artifacts/runs/verify_remaining_4090_20261115_insert_key_s0_a0/manifest.json)） | 55 |
| 精细操作 | `insert_tubes` | 插管 | `task_failed`：`cuRobo` seed 0 执行，`tube0` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/goal_curobo_5tasks_s0_20261115_insert_tubes_s0_a0/manifest.json)） | 55 |
| 精细操作 | `play_Xylophone` | 敲木琴 | `task_failed`：`cuRobo` seed 0 执行，`mallet` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/verify_parallel_f_20261115_play_Xylophone_s0_a0/manifest.json)） | 1 |
| 精细操作 | `plug_in_charger` | 插充电器 | `task_failed`：`cuRobo` seed 0 执行但原任务失败（[`manifest.json`](../artifacts/runs/verify_parallel_f_20261115_plug_in_charger_s0_a0/manifest.json)） | 1 |
| 精细操作 | `pour_balls_into_vase` | 将球倒入花瓶 | `task_failed`：`cuRobo` seed 0 执行，`cup` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/verify_parallel_f_20261115_pour_balls_into_vase_s0_a0/manifest.json)） | 1 |
| 开放能力 | `align_blocks` | 对齐积木 | `planning_failed`：官方三角尺标签已修正，但 `cuRobo` seed 0 三个接触高度均无抓取候选（[`manifest.json`](../artifacts/runs/verify_align_label_fixed_20261115/manifest.json)） | 55 |
| 开放能力 | `classify_objects_by_language` | 按语言分类物体 | `task_failed`：`cuRobo` seed 0 执行，`cat0_0` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/goal_curobo_8tasks_s0_20261115_classify_objects_by_language_s0_a0/manifest.json)） | 55 |
| 开放能力 | `general_pickup` | 通用抓取 | `success`（`task_success=true`、`quality_valid=false`）：90A seed 22 `cuRobo` 抬升 133.761 mm、208 帧诊断完整；FEM/固定区/体积/采样碰撞通过，STL 投影/保持稳定性/cooking setup 失败（[`manifest.json`](../artifacts/runs/verify_pickup_curobo_20261115/manifest.json)，[`acceptance.json`](../artifacts/runs/verify_pickup_curobo_20261115/reports/independent_spot/acceptance.json)） | 55 |
| 开放能力 | `pick_from_conveyor_by_image` | 按图像抓取传送带物体 | `planning_failed`：补齐普通房间后 `cuRobo` seed 0 规划失败（[`manifest.json`](../artifacts/runs/verify_conveyor_environment_fixed_20261115_pick_from_conveyor_by_image_s0_a0/manifest.json)） | 1 |
| 开放能力 | `pour_by_language` | 按语言倾倒 | `runtime_error`：seed 0 场景导出 `NoneType.get_world_pose`；倾倒专用动作缺失（[`manifest.json`](../artifacts/runs/verify_parallel_f_20261115_pour_by_language_s0_a0/manifest.json)） | 1 |
| 开放能力 | `solve_equation` | 解算式 | `planning_failed`：`cuRobo` seed 0 抓取规划失败（[`manifest.json`](../artifacts/runs/verify_parallel_c_20261115_solve_equation_s0_a0/manifest.json)） | 1 |
| 开放能力 | `stack_blocks_by_language` | 按语言堆叠积木 | `task_failed`：`cuRobo` seed 0 执行，`block_1` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/verify_remaining_local_20261115_stack_blocks_by_language_s0_a0/manifest.json)） | 1 |
| 开放能力 | `store_tools_in_toolbox` | 工具入箱 | `task_failed`：`cuRobo` seed 0 执行，`pliers` `probe_lift` 未随动（[`manifest.json`](../artifacts/runs/verify_parallel_d_20261115_store_tools_in_toolbox_s0_a0/manifest.json)） | 1 |

12 个随机变体（均已在补齐房间/材质/HDR 后完成 seed 0 实际尝试）：

- `arrange_largest_number_random`：`planning_failed`（`cuRobo` seed 0 规划失败，[`manifest.json`](../artifacts/runs/verify_random_a_environment_fixed_20261115_arrange_largest_number_random_s0_a0/manifest.json)）。
- `fold_clothes_random`：`runtime_error`（seed 0 场景导出 `NoneType.get_world_pose`；折叠专用动作缺失，[`manifest.json`](../artifacts/runs/verify_random_a_environment_fixed_20261115_fold_clothes_random_s0_a0/manifest.json)）。
- `hang_mugs_random`：`task_failed`（`cuRobo` seed 0 执行，`mug0` `probe_lift` 未随动，[`manifest.json`](../artifacts/runs/verify_random_a_environment_fixed_20261115_hang_mugs_random_s0_a0/manifest.json)）。
- `make_toast_random`：`runtime_error`（已通过 `articulation.initialize()`，但在场景导出阶段遇 `ValueError: No triangle geometry: .../toaster_4_15/root_joint`，[`manifest.json`](../artifacts/runs/verify_articulation_compat_20261115_local_make_toast_random_s0_a0/manifest.json)）。
- `pack_objects_into_box_random`：`planning_failed`（`cuRobo` seed 0 规划失败，[`manifest.json`](../artifacts/runs/verify_random_b_environment_fixed_20261115_pack_objects_into_box_random_s0_a0/manifest.json)）。
- `pour_liquid_into_cup_random`：`runtime_error`（seed 0 场景导出 `NoneType.get_world_pose`；倾倒专用动作缺失，[`manifest.json`](../artifacts/runs/verify_random_b_environment_fixed_20261115_pour_liquid_into_cup_random_s0_a0/manifest.json)）。
- `push_T_random`：`task_failed`（`cuRobo` seed 0 执行，`t` `probe_lift` 未随动；持续推动专用动作缺失，[`manifest.json`](../artifacts/runs/verify_random_b_environment_fixed_20261115_push_T_random_s0_a0/manifest.json)）。
- `sort_nesting_dolls_by_size_random`：`task_failed`（`cuRobo` seed 0 执行，`doll1` `probe_lift` 未随动，[`manifest.json`](../artifacts/runs/verify_random_b_environment_fixed_20261115_sort_nesting_dolls_by_size_random_s0_a0/manifest.json)）。
- `stack_blocks_random`：`task_failed`（`cuRobo` seed 0 执行，`block_1` `probe_lift` 未随动，[`manifest.json`](../artifacts/runs/verify_random_c_environment_fixed_20261115_stack_blocks_random_s0_a0/manifest.json)）。
- `stack_bowls_random`：`planning_failed`（`cuRobo` seed 0 规划失败；需补碗沿抓取，[`manifest.json`](../artifacts/runs/verify_random_c_environment_fixed_20261115_stack_bowls_random_s0_a0/manifest.json)）。
- `store_laptop_and_headphones_random`：`task_failed`（已消除 `dof_properties` 初始化异常，`cuRobo` seed 0 执行但原任务失败，[`manifest.json`](../artifacts/runs/verify_articulation_compat_20261115_local_store_laptop_and_headphones_random_s0_a0/manifest.json)）。
- `sweep_blocks_random`：`planning_failed`（`cuRobo` seed 0 规划失败，[`manifest.json`](../artifacts/runs/verify_random_c_environment_fixed_20261115_sweep_blocks_random_s0_a0/manifest.json)）。

本轮已从官方资产仓库补齐上述全部 54 项 seed 0 的选中模型、布局及环境资源，并保留下载来源与哈希。表中布局数为当前本地快照；布局存在不等于已执行成功。

证据：

- 54 项最新汇总总表：[`latest_results.json`](../artifacts/cache/remaining_verification_20261115/latest_results.json) 与 [`summarize.py`](../artifacts/cache/remaining_verification_20261115/summarize.py)。
- 上游完整任务清单与维度：[`task_inventory.py`](../third_party/RoboDojo/scripts/internal/task_inventory.py)。
- 当前入口允许的任务：[`config.py`](../src/soft_sim/config.py) 中 `TaskConfig.supported_tasks`，[`collect.py`](../src/soft_sim/cli/collect.py) 中 `--task`。
- 已接入的动作流程：[`tasks/collect.py`](../src/soft_sim/tasks/collect.py) 与 [`tasks/adapters.py`](../src/soft_sim/tasks/adapters.py) 的任务序列及目标位姿解析；详见[第一阶段实现说明](task-adaptation-phase1.md)。
- 本轮 90A 实际执行：[`task.json`](../artifacts/runs/goal_90a_pickup_host2_20261115/results/task.json)，[逐帧采集索引](../artifacts/runs/goal_90a_pickup_host2_20261115/capture/episode.json)。该运行在本轮正式目录修改前已启动，数据保留原 artifacts 位置作为独立证据；后续正式采集使用 `data/datasets/<task>/`。
- 早期抓取成功记录：[`verification.json`](../artifacts/cache/benchmarks/default80_20261115/verification.json)。
- 堆叠验证仍未完成：[`goal.md`](goal.md)。
