# RoboDojo 任务与软指适配清单

核对日期：2026-11-15。以下为本地固定版本快照与当前代码的状态，中文名称为便于阅读的译名，任务 ID 为准。

基础仿真任务 42 个；额外随机变体 12 个，共 54 套任务类与配置。基础任务均有上游定义，但自有软指执行器只接入了 2 个；其中仅通用抓取有当前安装的成功采集记录。18 个真机任务不属于此仿真仓库的已适配范围。

“已接入”表示存在任务动作编排与原任务成功判定的调用，不代表已验证完成；“未适配”表示没有相应软指动作执行流程，不能通过修改允许任务列表冒充支持。

默认材料为 90A。最近的 80 次迭代完整成功回放使用的是 85A，不能当作 90A 全任务验证。抓取成功记录来自少量布局/同一既有计划的多次运行，不代表全布局成功率。

| 维度 | 任务 ID | 任务 | 软指适配 / 执行验证 | 本地保存布局数 |
| --- | --- | --- | --- | ---: |
| 泛化 | `arrange_largest_number` | 排列最大数字 | 未适配；未执行验证 | 0 |
| 泛化 | `fold_clothes` | 叠衣服 | 未适配；未执行验证 | 0 |
| 泛化 | `hang_mugs` | 挂杯子 | 未适配；未执行验证 | 0 |
| 泛化 | `make_toast` | 制作吐司 | 未适配；未执行验证 | 0 |
| 泛化 | `pack_objects_into_box` | 物体装箱 | 未适配；未执行验证 | 0 |
| 泛化 | `pour_liquid_into_cup` | 倒液体入杯 | 未适配；未执行验证 | 0 |
| 泛化 | `push_T` | 推动 T 形物体 | 未适配；未执行验证 | 0 |
| 泛化 | `sort_nesting_dolls_by_size` | 按大小排列套娃 | 未适配；未执行验证 | 0 |
| 泛化 | `stack_blocks` | 堆叠积木 | 已接入；首块规划有证据；完整任务成功未验证 | 1 |
| 泛化 | `stack_bowls` | 叠碗 | 未适配；未执行验证 | 0 |
| 泛化 | `store_laptop_and_headphones` | 收纳电脑和耳机 | 未适配；未执行验证 | 0 |
| 泛化 | `sweep_blocks` | 清扫积木 | 未适配；未执行验证 | 0 |
| 记忆 | `cover_blocks` | 盖住积木 | 未适配；未执行验证 | 0 |
| 记忆 | `imitate_sorting_sequence` | 模仿分类顺序 | 未适配；未执行验证 | 0 |
| 记忆 | `match_and_pick_from_conveyor` | 匹配并抓取传送带物体 | 未适配；未执行验证 | 0 |
| 记忆 | `press_by_number` | 按数字按键 | 未适配；未执行验证 | 0 |
| 记忆 | `swap_T` | 交换 T 形物体 | 未适配；未执行验证 | 0 |
| 记忆 | `swap_blocks` | 交换积木 | 未适配；未执行验证 | 0 |
| 长程 | `classify_objects` | 物体分类 | 未适配；未执行验证 | 0 |
| 长程 | `fill_egg_holder` | 装满蛋托 | 未适配；未执行验证 | 0 |
| 长程 | `fill_pen_holder` | 装满笔筒 | 未适配；未执行验证 | 0 |
| 长程 | `make_kong` | 麻将杠牌 | 未适配；未执行验证 | 0 |
| 长程 | `organize_table` | 整理桌面 | 未适配；未执行验证 | 0 |
| 长程 | `play_stacking_toy` | 套叠玩具 | 未适配；未执行验证 | 0 |
| 长程 | `play_tic_tac_toe` | 井字棋 | 未适配；未执行验证 | 0 |
| 长程 | `put_bottles_into_dustbin` | 将瓶子放入垃圾桶 | 未适配；未执行验证 | 0 |
| 精细操作 | `build_tower` | 搭塔 | 未适配；未执行验证 | 0 |
| 精细操作 | `deposit_coin` | 投币 | 未适配；未执行验证 | 0 |
| 精细操作 | `fasten_screws` | 拧紧螺丝 | 未适配；未执行验证 | 0 |
| 精细操作 | `insert_key` | 插钥匙 | 未适配；未执行验证 | 0 |
| 精细操作 | `insert_tubes` | 插管 | 未适配；未执行验证 | 0 |
| 精细操作 | `play_Xylophone` | 敲木琴 | 未适配；未执行验证 | 0 |
| 精细操作 | `plug_in_charger` | 插充电器 | 未适配；未执行验证 | 0 |
| 精细操作 | `pour_balls_into_vase` | 将球倒入花瓶 | 未适配；未执行验证 | 0 |
| 开放能力 | `align_blocks` | 对齐积木 | 未适配；未执行验证 | 1 |
| 开放能力 | `classify_objects_by_language` | 按语言分类物体 | 未适配；未执行验证 | 0 |
| 开放能力 | `general_pickup` | 通用抓取 | 已接入；有成功采集记录；90A 全量未验证 | 55 |
| 开放能力 | `pick_from_conveyor_by_image` | 按图像抓取传送带物体 | 未适配；未执行验证 | 0 |
| 开放能力 | `pour_by_language` | 按语言倾倒 | 未适配；未执行验证 | 0 |
| 开放能力 | `solve_equation` | 解算式 | 未适配；未执行验证 | 0 |
| 开放能力 | `stack_blocks_by_language` | 按语言堆叠积木 | 未适配；未执行验证 | 0 |
| 开放能力 | `store_tools_in_toolbox` | 工具入箱 | 未适配；未执行验证 | 0 |

12 个随机变体：

- `arrange_largest_number_random`：未适配、未执行验证。
- `fold_clothes_random`：未适配、未执行验证。
- `hang_mugs_random`：未适配、未执行验证。
- `make_toast_random`：未适配、未执行验证。
- `pack_objects_into_box_random`：未适配、未执行验证。
- `pour_liquid_into_cup_random`：未适配、未执行验证。
- `push_T_random`：未适配、未执行验证。
- `sort_nesting_dolls_by_size_random`：未适配、未执行验证。
- `stack_blocks_random`：未适配、未执行验证。
- `stack_bowls_random`：未适配、未执行验证。
- `store_laptop_and_headphones_random`：未适配、未执行验证。
- `sweep_blocks_random`：未适配、未执行验证。

本地保存布局只覆盖通用抓取 55 个、堆叠积木 1 个和对齐积木 1 个。未发现保存布局不表示上游不能生成场景；它表示当前依赖保存布局的采集入口缺少相应输入。

证据：

- 上游完整任务清单与维度：[`task_inventory.py`](../third_party/RoboDojo/scripts/internal/task_inventory.py)。
- 当前入口允许的任务：[`config.py`](../src/soft_sim/config.py) 中 `TaskConfig.supported_tasks`，[`collect.py`](../src/soft_sim/cli/collect.py) 中 `--task`。
- 已接入的动作流程：[`tasks/collect.py`](../src/soft_sim/tasks/collect.py) 中 `operations` 与动作阶段。
- 抓取成功记录：[`verification.json`](../artifacts/cache/benchmarks/default80_20261115/verification.json)。
- 堆叠验证仍未完成：[`goal.md`](goal.md)。
