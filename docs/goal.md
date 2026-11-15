# 换装软指后的 RoboDojo 运动规划采集目标

当前目标是使用 **motion planning（运动规划）**，让换装新软指的机器人执行 RoboDojo 任务并收集仿真数据。当前阶段不以接入或训练策略模型为前提。

2026-11-15 晚剩余验证更新见[验证记录](remaining-verification-2026-11-15.md)与 [`latest_results.json`](../artifacts/cache/remaining_verification_20261115/latest_results.json)：全部 54 项任务 seed 0 的模型、布局及房间/材质/HDR 输入已补齐，54 项均已有最新实际运行记录（`overall_attempted` 54/54，由 53 项 seed 0 与 `general_pickup` seed 22 组成：`success` 1、`task_failed` 23、`planning_failed` 20、`runtime_error` 10；`task_success` 三态 `true: 1 / false: 43 / null: 10`，`quality_valid` 三态 `false: 1 / null: 53`）；默认 cuRobo 的 `general_pickup` seed 22 新增原任务成功对照（`task_success=true`，抬升 133.761 mm，208 帧诊断 LeRobot 数据完整回读，但独立质量复核 `quality_valid=false`）。堆叠 `indent=0.008` 已确认原驱动闭合与试抬指令实际执行，但物体未随动；详细质量检查与动作实现缺口另列，仍不能宣称全任务适配完成或质量全部通过。
4090 经用户授权改为两路独立任务队列，已实测双进程约 14.9GB 显存，见[双机说明](remote-4090-sync.md)。

2026-11-15 更新：优先尽可能完成更多任务的软指适配，再建设覆盖已适配任务的全量运行与采集入口。
换装夹爪会改变可抓取范围，不要求所有任务、物体、位置和 seed 都成功。不同布局与 seed
本身已经提供大量抓取和操作变化，应接受个别布局无法完成，把投入优先用于扩大可执行任务的覆盖面。
采集哪些任务、使用哪些 seed 和保存多少条数据，最终由 `configs/` 下的一份 YAML 采集清单决定。

## 必须保留

- RoboDojo 原双 X5 机械臂、原关节和驱动；仅替换末端手指，并增加软指所需的安装结构及指内相机。
- 原任务场景、布局、目标物与杂物，以及物体的质量、摩擦等物理属性。
- 原任务的成功／失败判定、评分条件和必要动作要求；不得通过降低难度或修改判定规则制造成功。

## 必须实现

- 针对新软指的实际尺寸、安装位置、接触几何和碰撞模型进行运动规划，执行任务所需的完整动作流程。
- 默认通过原机械臂驱动执行规划指令。直接改写关节状态的回放仅用于诊断，不能替代任务执行验收。
- 使用原任务规则判断成功或失败；默认只把成功且录制完整的 episode 保存到正式数据集，失败保留简要结果、原因与日志。
  场景加载、装配检查或规划成功均不能当作任务成功。
- 采集执行臂两根手指内相机的真实视角，与关节状态、动作指令、物体状态和软指形变按同一仿真时刻同步保存；保留外部相机便于复核。
- 保存可读取、可验证的 LeRobot v3 数据，并明确区分成功、失败和未完成的采集。
- 默认材料为 **TPU 90A**；默认软指求解采用 80 次迭代、1 ms 时间步、节点阻尼 20 和固定区消元，保留完整碰撞网格。
  已有 85A 成功记录仅作为实现参考，不能代替 90A 的实际执行记录。

## 验收依据

- 换指后，运动规划能够驱动机器人完成实际任务，并通过该任务原有的成功条件。
- 指内相机画面随机器人和软指的实际运动正确更新，图像与记录状态同步，数据完整且可回读。
- 检查真实的单元翻转、固定区滑移、碰撞和数值异常；查明引擎诊断与独立检查不一致的原因。
- 不随意改变材料、结构、物体属性或放宽阈值来通过验收，不通过图像过滤掩盖仿真问题。
- 按任务、布局和运行分别记录验证范围；单个抓取实例成功不能代表堆叠等其他任务已完成，也不能代表全任务成功率。

上述检查需要分别记录，但不要求每条成功操作先通过全部离线训练质量检查才保存。
保存以原任务成功和基本录制完整性为依据：例如 `general_pickup` 沿用原任务抬升超过 10 cm
的判定；堆叠、插接等任务使用各自原规则，不能统一降成“抓起来就成功”。相机／FEM 等详细
复核结果可以单独保留为质量标签；损坏、缺帧、无法写入的数据不冒充完整 episode。

“适配完成”表示任务有可执行的软指动作流程、已接入原任务判定，并有实际完整运行证据；
是否出现成功另行记录。存在失败 seed 不撤销已完成的适配，也不要求在进入下一个任务前解决
当前任务的所有失败。当前夹爪无法可靠操作的任务可以记录限制后继续推进其他任务。

## 推进顺序

当前安装方案已有验证基础。先用默认 90A 复验 `general_pickup`，补完 `stack_blocks` 的完整
物理执行，再把已有能力逐步扩展到更多任务。**先扩大任务适配覆盖，再实现统一全量入口与批量采集**；
不先围绕少数任务的全部 seed 反复调参，也不把批量循环跑通当作全任务已适配。

## 全任务适配范围与现状

逐项状态以 [任务适配清单](task-adaptation-status.md) 为准。当前上游快照有 42 个基础仿真任务和
12 个随机变体；18 个真机任务不在本轮仿真适配范围内。自有软指执行器原先只接入
`general_pickup` 和 `stack_blocks`；本轮已补充阶段 1 的 7 个任务动作流程，实际验证状态
见[第一阶段实现说明](task-adaptation-phase1.md)和任务清单。目前仍只有通用抓取有当前安装的成功采集记录。
55 个抓取布局仍然只是一个任务的不同输入，不计作 55 个已适配任务。

本轮目标是尽可能覆盖这 42 个基础任务；12 个随机变体在对应基础任务有可用流程后扩展。
任务定义、资产和布局是否存在，动作流程是否已接入，以及实际运行是否成功，分别记录。
本地暂存布局缺失不等于上游无法生成场景；需要补齐官方布局输入或接入上游生成流程，并保存
seed、布局快照和哈希，不能通过删掉难抓物体或换成自制简单场景来适配。

## 如何借鉴已有适配

复用已有的安装、规划、控制、判定和采集实现，把差异集中在任务的目标解析与动作编排。
不为每个任务复制一套启动和录制脚本，也不能只修改 `supported_tasks` 或 CLI 允许列表就宣称支持新任务。

| 已有实现 | 可以直接借鉴的部分 | 扩展时需要补齐的部分 |
| --- | --- | --- |
| [`sim/robodojo.py`](../src/soft_sim/sim/robodojo.py) | 加载固定版本的原任务类、配置、机器人和布局；调用原 RewardManager | 新任务的资产、布局来源、场景初始化和任务特有状态更新 |
| [`robots/fingers.py`](../src/soft_sim/robots/fingers.py)、[`sim/soft_tool.py`](../src/soft_sim/sim/soft_tool.py) | 实测软指安装、楔块、夹具、FEM、碰撞网格及指内相机随动 | 检查新动作的工作空间、抓握位置与夹具间隙；必要时增加双臂协调能力 |
| [`sim/scene.py`](../src/soft_sim/sim/scene.py) | 导出实时物体几何、障碍物、测量安装位姿和规划输入 | 从当前单目标／三积木逻辑推广到任意任务物体、容器、功能点及关节物体，保留任务语义 |
| [`planning/grasp.py`](../src/soft_sim/planning/grasp.py) | 软指尺寸约束下的候选抓取、接近、试抬、带物避碰及支撑面放置 | 任意目标位置／朝向放置、容器内部间隙、工具接触、插接、旋转及交接；不能全用“放到支撑物中心”替代 |
| [`execution.py`](../src/soft_sim/execution.py) | `pickup_phases` 的接近、闭合、保持、试抬、抬升，以及放置、松爪、撤离、回位；`phase_commands` 的原驱动指令 | 推动、按压、敲击、倾倒、连续接触和双臂交接等新动作阶段 |
| [`tasks/collect.py`](../src/soft_sim/tasks/collect.py) | 当前场景中规划并执行；逐操作重新导出；同步采样；调用原任务判定 | 将写死的 `operations` 和目标标签改为任务适配接口，支持动态动作顺序、两臂角色及任务进度 |
| [`recording/`](../src/soft_sim/recording/) | 仿真端暂存、LeRobot v3 提交、成功筛选与转换重试 | 为新增双臂／任务状态扩展稳定的数据字段；不同 schema 分集合，避免强行混写 |

`general_pickup` 是抓握原语的起点：新任务可复用从真实几何生成抓取、执行试抬、确认物体随动的流程。
`stack_blocks` 是多步骤任务的起点：复用“抓取 → 转移 → 放置 → 松爪 → 撤离 → 重新导出下一步场景”的编排。
堆叠当前完整执行仍未验证，复用它的代码不等于继承了成功结论。

上游任务代码主要提供场景、指令、成功条件和评分，不应假设每个任务都自带可直接调用的专家轨迹。
新增适配要逐项阅读 `third_party/RoboDojo/task/RoboDojo/tasks/<task>.py`、对应 YAML、资产功能点和
RewardManager 的实现，再编写必要的动作流程。

## 每个任务的适配步骤

1. **读懂任务要求。** 列出原指令、物体标签、目标位置／朝向、动作顺序、交接要求、触发事件和成功条件。
   同时检查 `run_reward`、`get_score`、`query` 等逻辑；例如对齐积木要用三角尺推动，插钥匙涉及交接，
   装蛋托还有关盖，不能截取一个抓取子动作当作完成整个任务。
2. **准备可复现输入。** 检查原任务资产、材质、布局与种子；缺少保存布局时使用上游对应生成流程。
   保留布局快照、来源和哈希，先选少量有代表性的输入调通，不要求逐个 seed 全成功。
3. **分解动作并选择可复用能力。** 优先组合已有抓取、转移、放置、松爪、撤离、回位。
   缺少的推动、倾倒、插接、关盖或交接能力作为公共动作补齐；目标位姿来自当前场景及任务规则。
4. **接入任务编排。** 任务适配层提供当前操作目标、活动机械臂、操作前后条件、下一动作和停止条件；
   共用仿真、规划与记录器。每个改变环境的步骤后更新场景，不能继续使用过时障碍物和目标位姿。
5. **执行完整任务并调用原判定。** 通过原驱动运行，用原任务管理器检查状态变化和最终结果。
   记忆、语言、交互与传送带任务还要处理指令、观察历史、对手／环境变化与时序，不能省略这些流程，
   也不能读取尚未展示的信息绕过任务要求。
6. **记录结果后继续扩展覆盖。** 每个任务先做少量完整试运行，记录成功、任务失败、规划失败或运行异常及原因。
   成功且录制完整的操作入库；失败不入正式集合。更新适配清单中的实现状态、运行证据和已知限制，
   再推进其他任务；成功率优化与更多 seed 验证留到后续批量阶段。

## 分阶段适配计划

下表按需要复用或新增的能力安排，不等同于官方五个评估维度。每组先选择容易复用的任务推进，
遇到夹爪几何、物理模型或动作能力的明显限制时记录原因，继续其他任务；不要求一组全部成功才能进入下一组。
下列新增工作均为计划，不表示已经实现。

| 阶段 | 任务范围 | 复用方式与主要新增能力 |
| --- | --- | --- |
| 0：确认现有基础 | `general_pickup`、`stack_blocks` | 用 90A 重新规划和完整执行；确认抓取、抓放及原任务判定能作为后续扩展起点 |
| 1：优先扩大抓放覆盖 | `stack_bowls`、`arrange_largest_number`、`sort_nesting_dolls_by_size`、`pack_objects_into_box`、`classify_objects`、`swap_T`、`build_tower` | 从已有抓放流程扩展任意目标位姿、多目标顺序、容器间隙和多层支撑；保留各任务朝向与顺序要求 |
| 2：双臂与复合长程操作 | `fill_pen_holder`、`fill_egg_holder`、`put_bottles_into_dustbin`、`organize_table`、`store_laptop_and_headphones`、`store_tools_in_toolbox`、`hang_mugs`、`play_stacking_toy` | 在抓放基础上增加一臂扶持、交接、挂放、插放、推动和开合关节物体；不同任务按实际要求选择能力 |
| 3：规则、记忆和动态场景 | `cover_blocks`、`swap_blocks`、`imitate_sorting_sequence`、`press_by_number`、`make_kong`、`play_tic_tac_toe`、`solve_equation`、`stack_blocks_by_language`、`classify_objects_by_language`、`match_and_pick_from_conveyor`、`pick_from_conveyor_by_image` | 复用操作原语，新增任务状态机、指令目标解析、观察记忆、交互事件和传送带时序；`make_kong` 是麻将杠牌任务 |
| 4：接触与精细操作 | `push_T`、`align_blocks`、`sweep_blocks`、`play_Xylophone`、`deposit_coin`、`plug_in_charger`、`insert_tubes`、`insert_key`、`fasten_screws`、`make_toast` | 新增工具抓持与使用、推动、敲击、按压、位姿对准、插入和旋转；为任务需要增加双臂交接与接触后调整 |
| 5：特殊物理与倾倒 | `fold_clothes`、`pour_liquid_into_cup`、`pour_balls_into_vase`、`pour_by_language` | 新增衣物接触与折叠、容器抓持、连续倾倒及内容物状态检查；先核对本地引擎和资产能力，明确暂时无法完成的项 |
| 6：扩展随机变体 | 对应 12 个 `_random` 任务 | 基础任务已有可执行流程后，接入各变体原有随机化；依据实际几何重新规划，不复用固定布局轨迹冒充泛化 |
| 7：统一全量入口 | 当大部分基础任务已有适配后开展 | 从 YAML 选择任务与输入，调用统一任务适配接口；实现批量执行、成功保存、失败跳过、断点续跑和覆盖汇总 |

本轮实施：默认 90A 的 `general_pickup` seed 22 已通过原驱动完整执行（原奖励 1，抬升约 136.814 mm，268 帧 LeRobot v3）；独立质量复核单独记录。42 个基础任务与 12 个随机变体的动作编排已接入 `tasks/adapters.py`，并建立 `configs/collection.yaml` 与 `collect-batch` 统一批量入口；各任务仍需通过实际 GPU 运行逐项验证，代码接入不等于适配成功。

### 规划与双臂适配排查项（2026-11-15）

相对原版 RoboDojo，本项目实质上**只改变了两件事**：
1. **末端夹爪的软体物理求解**（TPU 90A FEM 软指求解、80 次迭代、1 ms 步长、节点阻尼 20、固定区消元）；
2. **末端夹爪的构形与安装**（`measured_wedge_v3` 0.75 倍倾斜安装，完整外廓尺寸已与原装 X5 手指基本对齐）。

以上两项的基础装配与物理求解已经解决。因此，**除软指物理与夹爪构形外，其余部分（规划器、双臂选择、功能点抓取与任务执行逻辑）原则上应当尽量沿用原版 RoboDojo 的做法**。当前阶段只接受因软指构形／物理接触导致的抓取不到、打滑或碰撞，不接受机械臂可达却在规划阶段就报 `planning_failed`。针对首批任务试运行中的 `planning_failed`，日志与 `scene.json` 核对出以下排查方向：

1. **评估采用原版同款 `cuRobo` 规划器（对照当前 `mplib`）**：核对 [`third_party/RoboDojo/env/planner_manager/curobo_planner.py`](../third_party/RoboDojo/env/planner_manager/curobo_planner.py) 的批量 IK 与轨迹优化配置，对比 `mplib`（Pinocchio + FCL）在姿态采样、碰撞体包络与 IK 求解上的差异，必要时接入同款 `cuRobo` 规划路径以消除规划器本身导致的可达性损失。
2. **左／右臂未按目标位置选择**：此前 `collect.py` 默认固定使用右臂（`active_arm = 1`，基座 `x = +0.3`），导致 `cover_blocks`（`x = -0.20`）、`hang_mugs`（`x = -0.42`）、`fill_pen_holder`（`x = -0.34`）、`deposit_coin`（`x = -0.30`）等左臂区物体全部 IK 失败。已改为未显式指定 `--active-arm` 时按首目标 X 坐标自动选择左／右臂（`x < 0` 选左臂 0，`x >= 0` 选右臂 1）；后续进一步扩展为单步按需切换双臂。
3. **扁平物体桌面间隙过滤过严（`GEOMETRIC_CANDIDATES 0`）**：硬币、数字牌、平放笔高度仅 1～2 cm，原 `grasp.py` 只在质心 `zoffset <= 9 mm` 水平夹取，导致张开手指低于 `table_z + 2 mm` 而无候选。已在 `grasp.py` 增加贴近物体上沿的 `zoffset` 采样。
4. **`place` 阶段笛卡尔直线插值失败**：`insert_tubes` 在 `pregrasp/approach/lift` 通过后于 `place` 报笛卡尔 IK 失败，已增加 `plan_pose` 碰撞安全 RRT 回退。
5. **待后续排查的专项几何／自碰问题**：
   - `stack_bowls`（直径 120 mm > 最大张开 88 mm）：对称中心抓取不可行，需补充**容器壁／碗沿单侧抓取**候选。
   - `sort_nesting_dolls_by_size`：`link4` 与 `camera_base` 发生自碰，需核查相机底座碰撞体包络与接近姿态。

### 规划器切换与双机验证进展（2026-11-15 晚）

1. **已将 RoboDojo 同款 `cuRobo` 设为默认规划器**：
   - 新增 [`src/soft_sim/planning/curobo.py`](../src/soft_sim/planning/curobo.py)，在 `envs/robodojo/.venv` 进程内缓存复用 `CuroboPlanner`（基座相对桌面高度 `table_height = table_z - base_z - 0.025`）。
   - `scripts/run.py collect`、`collect-batch` 与 `plan` 均默认使用 `cuRobo`，保留 `--planner auto/mplib` 与 `plan-mplib` 作为显式回退。
   - 抽取共用候选生成器 [`src/soft_sim/planning/candidates.py`](../src/soft_sim/planning/candidates.py)（薄/扁物体加密 yaw 采样并自适应预压 `indent`），并修复 `grasp.py` 使用真实三角网格而非实心凸包。
2. **13 个任务 `seed 0` 的 `cuRobo` 实测表现**：
   - **8 个任务已通过规划并进入物理仿真执行**：`cover_blocks`、`insert_tubes`、`stack_blocks`、`arrange_largest_number`、`swap_T`、`pack_objects_into_box`、`classify_objects_by_language`、`sort_nesting_dolls_by_size`（其中套娃已执行至 `transfer` 阶段 12.0s；详细状态见 [任务适配清单](task-adaptation-status.md)）。
   - **5 个任务仍为 `planning_failed`**：`deposit_coin`、`classify_objects`、`build_tower`（抓起后 `transfer` 跨桌到另一侧臂区超出单臂范围）、`hang_mugs`（`lift` 0.14 m 边缘可达性）、`fill_pen_holder`（目标 `y=+0.066` 位于臂展极限）。
3. **当前排查重点**：
   - 多个进入仿真的放置任务在 `probe_lift` 阶段物体未抬起（`verify_stack_contact_20261115` 已验证 `--indent 0.008` 闭合至 18.8 mm、末端上移 11.98 mm 但方块仍未随动）；
   - 单臂跨桌 `transfer` 需进一步接入按操作目标自动切换左／右臂或交接。
4. **双机并行验证**：已打通 `ssh 4090`（RTX 4090 24GB，`/home/sustechdl/code/threecheng/soft_sim`）代码同步与回归测试流程，详见 [双机并行验证与 4090 代码同步说明](remote-4090-sync.md)。

阶段 1～5 合计覆盖余下 40 个基础任务。当前阶段的进展主要看新增可执行任务数和取得成功的任务数，
并附实际运行输入与失败原因；不以 42／42 全成功或某个任务全部 seed 成功作为前置条件。

## 最终数据保存目录

最终正式数据统一保存到项目根目录下的 `data/datasets/`，其下按任务名称建立目录：

```text
data/datasets/
├── general_pickup/   # 独立、完整的 LeRobot v3 数据集根目录
├── stack_blocks/     # 独立、完整的 LeRobot v3 数据集根目录
└── <task_name>/      # 其他任务使用原任务名称
```

进入任意 `data/datasets/<task_name>/` 后，该目录本身就是可直接读取和验证的 LeRobot v3
数据集根目录，包含该格式要求的元数据、数据文件及对应媒体文件；不再额外套一层 `dataset_id`
或运行目录。同一任务的成功且录制完整的 episode 追加到对应任务的数据集，失败日志、运行记录和
转换暂存另行保存，不作为正式 episode 混入其中。

## 后续全量入口与 YAML 采集清单

先完成大部分任务适配，再建设统一入口。全量入口的职责是编排已经存在的任务流程，不能代替任务适配本身。
“全量”表示遍历 YAML 选中的任务与输入；没有适配的任务明确记为未支持，缺资产／布局记为输入缺失，
实际操作失败与运行异常分开统计，不能把跳过项或仅加载场景算作已验证。

`configs/collection.yaml` 与 `python scripts/run.py collect-batch --config configs/collection.yaml` 已实现，统一决定采集任务、seed、默认材料与保存数量；交付验收命令 `validate-dataset` 与首条正式样本 `data/datasets/general_pickup` 验收记录见 [正式数据集交付与验收说明](dataset-delivery.md)。现有 `configs/*.json` 仍作为单任务默认参数；配置结构如下：

```yaml
defaults:
  material: bambu_tpu_90a
  success_only: true
  attempts_per_seed: 1

tasks:
  - task: general_pickup
    enabled: true
    seeds: [0, 1, 2, 22]
    dataset_id: x5_90a_general_pickup_success_v1
    max_saved_episodes: 100

  - task: stack_blocks
    enabled: true
    seeds: [0]
    dataset_id: x5_90a_stack_blocks_success_v1
    max_saved_episodes: 100

  - task: stack_bowls
    enabled: false  # 完成适配后再启用
    seeds: [0, 1, 2]
    dataset_id: x5_90a_stack_bowls_success_v1
    max_saved_episodes: 100
```

`max_saved_episodes` 是保存上限，不是必须凑齐的数量；选中 seed 的预算耗尽就汇总结束，
不能为达到数量而无限重试或降低成功标准。相同 seed 的重试不算新布局。
后续可增加“全部可用布局”的选择方式，但必须在启动前展开成明确任务与输入清单并保存快照，
不默默替换缺失布局。所有采集任务从 YAML 读取，批量脚本中不再写死任务名单。

统一入口需要做到：启动前检查配置与输入；每条运行独立记录；失败后继续下一条；成功且完整才提交；
已提交 episode 在恢复时不重复追加；转换失败但暂存完整的成功操作优先恢复提交；限制重试和超时；
同一集合串行写入。最终报告至少列出任务适配覆盖、尝试次数、成功保存数、失败数、跳过原因和数据集位置。
是否通过全部离线质量复核作为独立信息，不作为本轮成功数据入库的统一前置门槛。

## 小目标：软指安装倾角与连接件

新软指与原装 X5 手指的构形不同，不能只等比缩放后沿用平行安装。以软指的实际抓取曲面／斜面为依据，验证是否需要两侧对称向内倾斜，并比较楔形连接件与直接调整安装角度的方案；最终角度由网格与装配验证确定，不能凭示意图直接定值。

- 保留原双 X5 的手腕、滑块、驱动和关节限位；通过手指安装姿态及必要的连接件适配新软指。
- 在原滑块允许的最闭合位置，两侧抓取面应正确相向并基本贴合，检查接触区域、面法向及沿曲面的间隙；允许接触后的真实软体形变，不以初始穿模或修改关节限位实现贴合。
- 检查整个开合过程：指体、连接件与相机不得发生非预期碰撞；张开时应有任务所需的抓取空间，并检查接近、放置、松爪和撤离时与桌面及积木的间隙。
- 软指网格、固定区、指内相机、连接件碰撞模型、运动规划与数据中记录的安装变换必须一致。
- 先保存新旧构形及不同安装倾角的装配对照、闭合间隙和碰撞检查结果，再进行原场景中的物理抓取与堆叠验证；旧平行安装的成败数据保留其原安装配置，不混作新安装姿态的验证。

尺寸比较的基准是**替换前的原装 X5 夹爪**。新软指应整体略小于原爪，形状和内部结构保持等比缩放；不是相对当前 0.65 再缩小一点，也未确定采用 0.6。按同一安装坐标比较完整外廓、前伸长度和实际工作区截面，重点检查堆叠放置、松爪和撤离时对积木的推碰。原装手指前伸约 71 mm，0.75／0.8 倍软指分别约 66.1／70.5 mm，可作为尺寸对照；每个比例均重新规划并做物理执行验证，不能仅凭尺寸较小推断任务必然成功。

### 安装小目标进展（2026-11-14）

- [x] 从网格外侧抓取面测得约 12°，比较 0/6/10/12/14/16°；选择对称内倾与独立楔块。
- [x] 最终 `measured_wedge_v3` 采用 0.75 倍，完整装配前伸 70.203 mm，闭合方向宽 36.800 mm，略小于原爪对应 71.000/37.350 mm；0.8 倍倾斜安装前伸 74.750 mm，未作为默认。
- [x] 原滑块零位下外侧抓取面间隙约 0.20 mm、法向相向，无初始穿插；保留原限位。
- [x] 完整原滑块行程静态避碰及非对称开合几何检查通过；四指原驱动全闭合再张开完成，50 个形变采样中无真实翻转、固定区滑移或非预期相机/楔块/腕部/桌面/积木表面穿插。
- [x] FEM 固定区、指内相机、Isaac 连接件实体、规划 URDF 及数据共用实际安装变换；新旧配置不匹配时拒绝执行。
- [x] 从本次真实场景导出完成 `stack_blocks_0` 首块接近、放置、松爪、撤离和回位的碰撞规划；该项不等于完整物理堆叠成功。
- [x] v2 安装曾按原驱动通过 `general_pickup_22`：最终抬升约 136.996 mm，原任务奖励 1，268 帧 LeRobot v3；但相机外壳遮挡/跟随检查失败，不算最终 v3 相机验收。
- [ ] 完整 `stack_blocks` 物理执行及新安装采集的全部质量验收；旧平行安装成败不迁移。

证据、适用范围和复现命令见 [倾斜安装验证](x5-tilted-installation-2026-11-14.md)。真实形变避碰为 0.1 秒采样复核，连接件及相机外壳仍是待实物 CAD 标定的仿真包络。

### 实现收敛（2026-11-14）

本节保留当时的实现记录；当前入口已改为 `python scripts/run.py <command>`，命令与目录以
[README](../README.md) 为准。后续任务扩展沿用当前 `src/soft_sim/` 路径，不恢复下述旧脚本。

- 当前目标只保留一条实现路径：RoboDojo 原任务 + 原双 X5 + `measured_wedge_v3` 软指安装，
  唯一入口 `benchmarks/robodojo_task_soft.py`。
- RoboDojo 源码快照移至 `third_party/RoboDojo/`；任务、布局、机器人、物体、材质和成功判定仍全部取自该快照。
- 删除 RoboTwin 源码与 Franka 规划路径，删除 MuJoCo 后端、Isaac Sim 6.2 后端、
  旧资产演示入口 `robodojo_soft_pick.py`、旧 NPZ 录制器及相关测试。
- 任务入口不再提供 `--robot franka` 或 `--mount-profile parallel_legacy`；旧平行安装与旧比例结果
  仅作为历史证据保留在 `docs/archive/`，不能复现为当前安装姿态的验收数据。
- 调用方式、数据复核和边界见 [README](../README.md) 与
  [倾斜安装验证](x5-tilted-installation-2026-11-14.md)。
