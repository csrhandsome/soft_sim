# Soft Sim 重构方案

本文基于当前工作区，而非 Git 中尚未删除的旧文件编写。本文是实施方案，本次只新增文档，不移动代码、修改环境或迁移历史数据。

目标是：**入口在 `scripts/`，实现放在 `src/soft_sim/`，配置只有一套规则，测试集中分类，所有生成物都有固定出口。** 保留现有 RoboDojo 原任务、双 X5、软指安装、官方 RewardManager 判定及独立质量复核的语义。

## 1. 当前情况与结论

### 1.1 不是只有两个环境

当前有根目录环境和四个 `envs/` 子项目，共五套环境定义：

| 环境 | 当前 Python | 关键依赖 | 当前职责 | 本轮建议 |
| --- | --- | --- | --- | --- |
| 根 `.venv` | 3.12 | NumPy 2.3.4、SciPy 1.16.3 | 公共逻辑与开发测试 | 保留为开发环境 |
| `envs/robodojo` | 3.11 | Isaac Sim 5.1、IsaacLab、CuRobo、NumPy 1.26.0 | 任务场景、物理仿真、采集 | 保留 |
| `envs/planning` | 3.10 | mplib、SAPIEN、NumPy 1.26.4 | 碰撞规划、安装检查、部分物理复核 | 保留 |
| `envs/dataset` | 3.11 | LeRobot 0.4、NumPy 2.x | LeRobot 写入、读取、相机与数据复核 | 保留 |
| `envs/meshing` | 3.11 | TetGen、PyVista、NumPy 2.3.4 | 可选的贴合四面体网格生成 | 按需安装 |

用户理解的 RoboDojo / LeRobot 隔离方向是对的：当前依赖声明分别要求 NumPy 1.26 和 2.x，不能放进同一解释器。`EpisodeRecorder` 已经通过 `capture.partial.npy` 暂存，再启动 dataset 环境的收尾脚本。这是已有的进程边界，不需要重做成服务。

不过，当前采集还会启动 Python 3.10 的规划子进程，因此主流程实际是 **仿真、规划、数据集三个运行环境协作**。重构先统一它们的调用方式，不以强行合并环境为目标。这里依据本地依赖声明和调用代码判断，未重新尝试依赖合并。

### 1.2 混乱来自职责与路径交叉

| 现状 | 具体例子 | 重构方向 |
| --- | --- | --- |
| `benchmarks/` 实际承载生产代码 | 任务入口、规划器、录制器、读取器、检查器都在里面 | 按职责迁入 `src/soft_sim/` |
| 入口与实现混在一起 | `robodojo_task_soft.py` 约 726 行，负责参数、建场景、规划、执行、采集与收尾 | 薄 CLI + 流程编排 + 引擎适配 |
| `shared/` 同时有活跃工具和旧框架 | 几何、网格仍在用；`BaseGripperEnv`、旧统一 CLI 对应已移除后端 | 按调用关系拆分，不整体改名搬家 |
| 项目导入依赖文件位置 | `sys.path.insert`、`from isaac51_tool import ...`、双路导入 fallback | 安装本地包，统一绝对导入 |
| 环境调用散落 | 任务、录制器、审计脚本分别拼 `.venv/bin/python` 和脚本路径 | 一个环境注册表与进程调用模块 |
| 配置体系脱节 | 当前任务入口直接构造 `ExperimentConfig()`，未使用 `configs/` 五个旧 JSON | 建立当前主流程真正使用的配置 |
| 测试未完全归位 | `benchmarks/test_soft_grasp_frames.py` 在测试目录外 | 全部自动测试进 `tests/` |
| 输出位置和命名不统一 | `data/generated/` 混放网格、计划和运行；根目录存在 PNG、MP4 | 统一存储根与产物归属 |
| 源码信息分散 | 根 TOML、`upstream_versions.json`、`third_party/sources.json` | 区分安装清单、参考清单与运行快照 |

## 2. 目标目录

目录按现有职责拆分；小模块先保持单文件，只有确实需要时再细拆。以下为目标结构，不表示这些文件已存在。

```text
soft_sim/
├── README.md
├── pyproject.toml                  # soft_sim 包、公共依赖范围、开发配置
├── uv.lock                         # 根开发环境锁
├── .python-version                 # 根开发/准备脚本用的 Python
├── scripts/                        # 用户可执行入口，不放业务算法
│   ├── run.py                      # 日常操作唯一调度入口
│   ├── setup_sources.py            # 环境尚未安装时的源码准备入口
│   └── sync_envs.sh                # 环境同步入口
├── configs/
│   ├── general_pickup.json         # 相对代码默认值的必要覆盖
│   └── stack_blocks.json
├── src/
│   └── soft_sim/
│       ├── __init__.py             # 无重型导入、无启动副作用
│       ├── cli/                    # 参数解析和命令分发
│       ├── config.py               # 类型定义、加载、覆盖、校验
│       ├── contracts.py            # 场景、计划、采集、数据集引用契约
│       ├── geometry.py             # 坐标变换、投影等公共数学
│       ├── mesh.py                 # 通用网格读写与计算
│       ├── storage.py              # 所有输出路径及 manifest 更新
│       ├── runtime.py              # 环境注册表、子进程、日志与退出码
│       ├── tasks/                  # collect/export/replay/assembly 编排
│       ├── robots/
│       │   └── x5/                 # 安装几何、运动学、URDF/TCP 生成
│       ├── sim/
│       │   ├── launcher.py
│       │   ├── robodojo.py         # 上游任务加载和适配
│       │   ├── scene.py            # USD 场景与几何导出
│       │   ├── soft_tool.py        # FEM 软指及安装
│       │   └── cameras.py          # 引擎相机与渲染表面支持
│       ├── planning/               # mplib 规划与候选生成
│       ├── execution.py            # 不依赖引擎的动作阶段、轨迹命令
│       ├── recording/
│       │   ├── capture.py          # 仿真端暂存，不导入 LeRobot
│       │   ├── stream.py           # 暂存格式读写，共享轻依赖
│       │   ├── lerobot_io.py       # dataset 环境专用写入/读取
│       │   └── finalize.py         # 暂存到正式数据集的收尾
│       ├── validation/             # 质量指标、碰撞与验收实现
│       ├── meshing/                # TetGen 等可选生成器
│       └── sources.py              # 源码快照准备与补丁逻辑
├── tests/
│   ├── unit/
│   ├── integration/
│   │   ├── planning/
│   │   ├── dataset/
│   │   ├── meshing/
│   │   └── sim/
│   ├── e2e/
│   └── fixtures/                   # 少量固定输入，不放运行产物
├── envs/
│   ├── robodojo/{pyproject.toml,uv.lock}
│   ├── planning/{pyproject.toml,uv.lock}
│   ├── dataset/{pyproject.toml,uv.lock}
│   └── meshing/{pyproject.toml,uv.lock}
├── assets/
│   ├── meshes/soft_finger.stl      # 版本控制中的源网格
│   ├── materials/                 # 材料参考资料
│   └── downloads/                 # 外部下载的输入资产，不提交 Git
├── artifacts/                     # 统一生成物出口，不提交 Git
│   ├── runs/
│   ├── datasets/lerobot/
│   └── cache/
├── third_party/                   # RoboDojo 快照、参考代码及其清单
├── .deps/                         # IsaacLab / CuRobo 安装依赖快照
└── docs/
    ├── goal.md                    # 当前目标与验收依据
    ├── refactor.md                # 本方案与实施状态
    └── archive/                   # 历史实验和旧方案
```

`benchmarks/`、`shared/`、顶层 `backends/` 在迁移完成后退出当前结构。以后确实新增基准评测时再建立 `benchmarks/`，不要继续把它用作生产代码总目录。

## 3. 代码边界与公共代码

公共代码按职责命名，不再新建 `common/`、`utils/` 或另一个杂物式 `shared/`。

依赖方向为：

```text
scripts/run.py → 环境调度 → cli → tasks / finalize / validation
                                      ↓
                         sim / planning / recording
                                      ↓
              config / contracts / geometry / mesh / storage
```

- 基础模块不得反向导入 `sim`、`planning`、`lerobot` 或 CLI。
- `recording/capture.py` 与 `lerobot_io.py` 通过文件契约协作，不通过继承共享运行时状态。当前 `EpisodeRecorder(LeRobotRecorder)` 应改为各自独立、复用轻量校验函数。
- 纯数组算法从引擎文件抽出。例如 `base_pin_mask`、渲染表面的纯几何绑定计算，单元测试不应为调用它们而导入仿真实现。
- `import soft_sim` 不加载 Torch、mplib、SAPIEN、Isaac Sim、LeRobot，也不创建目录或修改环境变量。
- Isaac Sim 启动前后的导入顺序必须保留：`launcher` 启动完成后才导入需要 Kit 的 `omni`、USD 和上游任务模块。
- 对 RoboDojo 非标准包结构的路径适配集中在 `sim/robodojo.py`；去掉自有代码的 `sys.path` 注入。上游本身暂时需要的路径适配是明确例外。

### 3.1 主要文件迁移映射

| 当前文件/目录 | 目标 | 处理方式 |
| --- | --- | --- |
| `benchmarks/robodojo_task_soft.py` | `cli/`、`tasks/`、`sim/robodojo.py`、`sim/scene.py` | 分离参数、编排、上游适配与场景操作 |
| `benchmarks/plan_soft_grasp.py` | `planning/` | 函数可调用，CLI 不参与算法 |
| `benchmarks/x5_mount_geometry.py`、`x5_soft_tool.py` | `robots/x5/` | 安装、运动学、URDF 使用同一安装参数 |
| `benchmarks/isaac51_tool.py` | `sim/soft_tool.py` | 纯几何函数下沉到基础模块 |
| `backends/isaac_sim/` | `sim/` 及基础几何模块 | 保留当前仍使用的三类能力 |
| `benchmarks/motion_execution.py` | `execution.py` | 保持无引擎依赖 |
| `benchmarks/x5_assembly_motion.py` | `tasks/assembly.py` | 诊断执行归任务流程 |
| `episode_recording.py`、`episode_reader.py`、`finalize_lerobot.py` | `recording/` | 拆开仿真端与数据集端依赖 |
| `benchmarks/check_*.py`、`validate_*.py`、`audit_task_episode.py` | `validation/` | 计算返回报告；路径由调用者提供 |
| `compare_*.py`、`probe_x5_fem_initialization.py` | `validation/` 或 `sim/diagnostics.py` | 离线比较和运行时探针分开 |
| `benchmarks/test_soft_grasp_frames.py` | `tests/integration/planning/` | 从生产代码目录移出 |
| `shared/config.py`、`geometry.py`、`mesh.py` | 对应基础模块 | 逐项去除旧后端参数与路径假设 |
| `shared/tool_mount.py` | `geometry.py` / `robots/x5/` | 按是否专属 X5 分配 |
| `scripts/prepare_mesh.py` | `meshing/` + CLI | 生成逻辑进入包内 |
| `prepare_sources.py`、`clone_benchmarks.py/.sh` | `sources.py` + `setup_sources.py` | 合并重复的公开准备入口 |
| `scripts/download_benchmark_assets.py` | `sources.py` 或资产下载模块 + CLI | 下载到明确的输入资产目录 |

旧 `shared/cli.py`、`environment.py`、`dataset.py`、`experiment.py` 中部分逻辑属于已移除的多后端框架。`train_force.py`、`predict_force.py` 目前读取旧 NPZ episode，不能因为入口存在就当作当前 LeRobot 训练链路。

处理顺序：先检查生产调用、测试调用和历史数据使用者；复用的函数迁入对应模块，无生产消费者的旧框架退出主包。旧力估计/学习功能如需保留，明确作为独立能力并验证数据接口；不能提供空壳的 LeRobot 训练入口，也不能仅凭旧测试仍引用就把整套旧配置保留。

## 4. 入口与环境管理

### 4.1 对外入口

日常统一使用根开发环境执行 `scripts/run.py`。它只用标准库完成初始解析和环境选择，再启动指定解释器；具体命令通过 `python -m soft_sim.cli.<module>` 执行。这样入口脚本本身不要求提前导入重型包。

建议保留以下命令，命名与帮助信息集中管理：

| 命令 | 用途 | 使用环境 |
| --- | --- | --- |
| `collect` | 当前官方场景规划、执行、暂存、转换与复核 | RoboDojo → planning / dataset |
| `export-scene` | 只导出场景 | RoboDojo |
| `replay` | 执行匹配场景的已保存计划 | RoboDojo |
| `assembly` | 原驱动安装与行程诊断 | RoboDojo |
| `plan` | 对已有场景离线规划 | planning |
| `prepare-mesh` | 生成网格 | 按方法选择 core / meshing |
| `check-installation` | 安装几何与碰撞检查 | planning |
| `finalize` | 将已完整暂存的采集写入 LeRobot | dataset |
| `validate` | 组织数据、相机、FEM、碰撞与任务验收 | core 编排 → dataset / planning |
| `preview` | 从已有运行或数据集生成预览 | dataset |
| `download-assets` | 下载必要输入资产 | 准备环境 |
| `test` | 按组选择环境执行测试 | 按测试组分发 |
| `doctor` | 检查解释器、包、源码、资产与路径 | 各环境的最小探针 |

以下是重构后的使用示例，目前还不能直接执行：

```bash
python scripts/run.py collect --config configs/general_pickup.json --seed 22 \
  --dataset-id x5_pickup_v1

python scripts/run.py assembly --config configs/general_pickup.json
python scripts/run.py validate --run-id <run_id>
python scripts/run.py preview --run-id <run_id>
python scripts/run.py finalize --run-id <run_id>
python scripts/run.py test --group unit
python scripts/run.py test --group dataset
```

每条命令结束时打印 `run_id`、run 目录、正式数据集目录与 episode 索引、报告位置和执行状态。无数据集产出的命令不打印虚构的数据集路径。

### 4.2 本地包与依赖

根 `pyproject.toml` 改成可安装的 `src` 布局包，子环境通过 uv 的本地 editable dependency 安装同一份 `soft-sim`。自有模块统一写 `from soft_sim...`，不再依赖当前工作目录或 `PYTHONPATH`。

这里有一个必须处理的细节：**根包当前 `requires-python = >=3.12,<3.13` 且公共依赖固定为 NumPy 2.3.4，不能原样安装到 planning / RoboDojo。**

- 根包的 Python 兼容范围调整为覆盖现有 3.10–3.12，并实际检查代码语法和标准库兼容性。
- 包的公共运行依赖声明兼容范围；NumPy、SciPy 等具体版本由各环境 TOML 和 lock 固定。不要把根开发环境的 NumPy 2.3.4 强加给仿真环境。
- 根开发环境通过开发依赖或 constraints 保留已验证的版本，根 `.python-version` 仍可指定 3.12。
- 准备源码使用的 `tomllib` 等 3.11+ 功能不在 Python 3.10 运行路径中顶层导入；准备脚本使用指定的开发 Python。
- 不把五套环境改成需要统一求解的 uv workspace；继续各自维护锁文件。
- 第一轮保持已验证的重型依赖版本，不同时升级 Isaac Sim、LeRobot 或规划器。

`runtime.py` 统一登记环境路径，禁止其他模块拼接 `envs/.../.venv/bin/python`。默认用已有目录，可通过明确的命令选项指定外部解释器；不静默退回系统 Python。

子进程统一记录命令、环境名、开始结束时间、退出码和日志路径，使用参数列表调用。规划或转换失败必须传递到调用方。`sync_envs.sh` 的默认安装集合与当前一致，meshing 作为明确可选目标补齐；说明文档不再让人误以为 `all` 已包含可选环境。

源码准备保留根 TOML 中的固定提交为安装事实来源；第三方参考项目清单仍放 `third_party/sources.json`。旧 `upstream_versions.json` 中的历史验证结论进入归档，不再承担重复的依赖版本配置。

## 5. 配置：少量、有效、可追溯

### 5.1 格式与优先级

建议继续用 **JSON** 管理自有运行配置：现有加载代码已经使用 JSON，没有必要为目录整理引入新的 YAML/Hydra 体系。

只采用这一条规则：

```text
代码中的类型化默认值 < 一个任务 JSON 的局部覆盖 < 显式 CLI 参数
```

嵌套对象按字段合并，数组整体替换；未知字段、类型错误、非法范围立即报错。CLI 未指定的参数不得用 argparse 默认值覆盖配置文件。配置校验在启动 Isaac Sim 之前完成。

不增加多层 `extends`、不同环境各自一份任务配置、每个 seed 一份 JSON，或完整复制默认值的 `default.json`。

建议配置分区：`task`、`tool`、`material`、`mesh`、`simulation`、`camera`、`planning`、`recording`。输出位置由统一存储模块决定；只开放 `--storage-root`、`--dataset-id` 等必要选项，不让各分区自由填写输出目录。

安装参数以当前 `measured_wedge_v3` 为中心，去掉为了复用旧配置而把 X5 设置成 `profile="panda"` 的过渡做法。规划、仿真、相机与验收使用同一个解析后的安装模型。

配置内相对输入路径按配置文件所在目录解析；CLI 显式相对路径按调用目录解析；默认存储根定位到项目 `artifacts/`。解析后生成绝对运行路径用于执行，同时在快照中保存存储根内的相对引用，保证整体移动产物目录后仍能读取。

每次运行自动写 `config.resolved.json`，记录完整有效参数、CLI 覆盖及配置格式版本。这是运行快照，不是第二份供用户编辑的配置。

### 5.2 现有配置处理清单

| 当前配置 | 建议 |
| --- | --- |
| `configs/default.json` | 退出当前配置目录；有效的材料/相机默认值回到类型定义 |
| `configs/panda.json` | 旧机器人 profile，引用清理后删除；历史说明留归档 |
| `configs/angular.json` | 旧夹爪模式，引用清理后删除 |
| `configs/isaac_ycb.json` | 旧独立场景路径，引用清理后删除 |
| `configs/isaac_webrtc_preview.json` | 旧预览配置，引用清理后删除 |
| RoboDojo `env_cfg/**/*.yml`、上游任务配置 | 继续由适配层加载，不复制、不批量删除 |
| CuRobo 必需 YAML、生成的规划配置 | 作为上游输入或规划产物，放在对应位置，不算自有任务配置 |

`general_pickup.json` 和 `stack_blocks.json` 只保留当前任务的必要差异。具体 seed、单次输出 ID 从 CLI 指定。独立预设只有在确有稳定用途时才新增。

## 6. 存储出口：所有生成物有固定归属

这是与代码目录同等优先的重构项。采用一个可配置存储根，默认 `<项目>/artifacts/`；大磁盘场景通过 `--storage-root /path/to/storage` 整体切换，内部结构一致。

### 6.1 三种顶层归属

| 位置 | 存什么 | 生命周期 |
| --- | --- | --- |
| `artifacts/runs/<run_id>/` | 单次执行的参数、场景、计划、暂存、诊断、日志和预览 | 实验记录；不能当缓存随意删 |
| `artifacts/datasets/lerobot/<dataset_id>/` | 正式 LeRobot 数据集 | 长期保存、独立复制与训练 |
| `artifacts/cache/` | 可重建、可复用的网格或资产加工缓存 | 可按规则清理 |

输入资产位于 `assets/` 或上游 RoboDojo 的既定资产目录；下载器不再把输出型 manifest 随手写进 `data/generated/`。上游必须依赖固定资产位置时由适配层显式处理，不在这一轮强行重排第三方内部目录。

根目录不再出现 `frame_verify_*.png`、`pickup22_*.mp4`；也不继续增加 `data/generated/x5_*_NEW` 之类人工命名目录。

### 6.2 一次运行的标准结构

`run_id` 自动生成，例如 `20261114T080000Z_general-pickup_s22_a1b2c3`，含 UTC 时间、任务、seed 和短随机后缀；标识创建时就确定，不用成功/失败后改目录名。同 seed 重跑会产生新 run。

```text
artifacts/runs/<run_id>/
├── manifest.json                    # 本次运行唯一的执行/存储状态索引
├── config.resolved.json             # 完整参数快照
├── provenance.json                  # 代码版本、上游版本、锁文件与资产摘要
├── inputs/
│   └── layout.json                  # 本次实际使用布局的快照
├── scene/
│   ├── scene.json
│   └── *.npz                       # 物体、障碍物等导出几何
├── plans/
│   └── operation_000/
│       └── attempt_000/             # plan、轨迹、URDF、碰撞代理
├── capture/
│   ├── episode.json                 # 采集索引，不负责声称任务验收通过
│   ├── finger_reference.npz
│   └── capture.partial.npy          # 转换确认前的跨环境暂存
├── results/
│   └── task.json                    # RewardManager 结果、任务阶段与检查点
├── reports/
│   └── <validation_id>/             # 每次复核单独一份，避免覆盖证据
│       ├── acceptance.json
│       ├── camera.json
│       ├── fem.json
│       └── collisions.json
├── diagnostics/                    # 无效 FEM 样本、静态探针、装配扫描等
├── previews/
│   └── <preview_id>/                # 拼接视频、裁剪、关键帧
└── logs/                            # 仿真、规划、转换、复核进程日志
```

按需创建子目录。只做网格准备、安装检查或跨运行比较时，也创建带命令类型的 run，报告和图进入本次 run，不写到被读取的历史 run 中。检查器输出统一由路径对象传入，不再从 `recording.parent` 猜位置或把报告写回输入目录。

未来若保留训练功能，模型和训练指标放在训练 run 的 `models/`、`reports/`，其 manifest 引用输入 dataset ID 和版本；不另建命名随意的全局模型堆。

### 6.3 LeRobot 数据集的唯一正式出口

```text
artifacts/datasets/lerobot/<dataset_id>/
├── meta/                           # LeRobot 官方写入器管理
│   ├── info.json
│   └── simulation/                 # 本项目已有的仿真来源扩展
├── data/                           # 官方 Parquet 分片
└── videos/                         # 官方视频分片
```

此处仅示意顶层归属；具体分片与文件名继续由当前锁定版本的官方写入器决定，不能按自己的 episode 目录重排。

约定如下：

1. `dataset_id` 是明确指定的集合名，例如 `x5_pickup_v1`，一个集合可包含多个 seed / episode；不要每次重跑都产生一套默认数据集。
2. 首次创建集合时保存 schema、机器人安装、网格摘要、FPS、相机和状态维度等兼容签名。同一集合的新增 episode 必须匹配；签名不符就拒绝追加，要求新 dataset ID。首版按任务分集合。
3. `run_id` 标识一次执行，`dataset_id + episode_index` 标识正式数据所在位置。run 的 manifest 保存二者的对应关系，避免复制正式数据到 run 目录。
4. LeRobot 原始相机视频只保留在官方 `videos/`；三视图拼接、放大和标注视频是派生产物，放在 run 的 `previews/`，不能混入训练帧。
5. `meta/simulation/` 保留离线读取所需的布局快照、校准、FEM 参考和来源信息。它不能只保存某台机器的绝对 run 路径；数据集单独复制后仍能读取。
6. 重跑复核不往正式数据集中持续堆 PNG、诊断 NPZ、审计日志和多版 acceptance。它们归 run；数据集只保存必要来源扩展。
7. 不直接重命名、移动或手工删除官方分片。复制或迁移以整个数据集根为单位；筛选成功样本通过索引/官方导出流程生成新集合。
8. 一个集合同一时间只允许一个收尾写入器。并发采集可各自暂存，转换时按集合串行加锁，不能让多个进程争抢 episode 索引。

### 6.4 状态、失败恢复和清理

必须区分以下几类状态，不能把“文件写完”等同于“任务成功”：

| 维度 | 示例状态/字段 | 含义 |
| --- | --- | --- |
| 执行 | `created / running / completed / failed` | 流程是否正常结束 |
| 采集/转换 | `none / capturing / staged / converting / committed / failed` | 暂存和正式写入进度 |
| 任务 | `task_success: true / false / null` | 官方 RewardManager 结论，未执行为 null |
| 质量复核 | `pending / passed / failed` | 相机、数据、FEM、碰撞检查 |
| 最终验收 | `accepted` | 任务成功且必要质量复核全部通过 |

完整、质量合格的失败任务 episode 可以保留并明确标记，不能伪装成成功数据。采集损坏或未完成的记录保留在 run 的暂存区，不作为完整正式 episode 提交。训练样本选择显式依据验收结果。

暂存格式保留 NumPy 数组流和 JSON，增加协议版本、字段顺序、shape、dtype、帧数、单位、坐标约定与必要摘要；两端只交换文件，不传 pickle/Python 对象。场景与计划同样带版本、布局及安装摘要。

转换成功的判据包括：官方写入器保存并完成收尾、目标 episode 可重新读取、来源元数据与 run 引用落盘。达到这些条件后才能删除 `capture.partial.npy`，避免同时保存两份大体积原始数据。失败则保留流、索引和错误日志供显式重试。

`finalize --run-id ...` 必须具备重试语义：已提交的 run 返回原 episode 引用，不能重复追加。如果在官方写入完成后、run 状态更新前崩溃，重试要先在数据集来源记录中核对 run ID；无法确定写入状态时停止追加并报告待恢复状态。文件系统 rename 不能让多个 LeRobot 分片和 run manifest 成为一个原子事务，不能只靠改状态字段假装解决。

`manifest.json` 以临时文件加替换方式更新；数据集追加由集合锁保护。已存在的 run 目录禁止覆盖，报告和预览重跑生成新 ID。

缓存以源网格摘要、生成方法、参数和工具版本生成 key；缓存不是 run 的唯一实验依据，run 仍保存引用和摘要。缓存被清理后，历史数据读取不能依赖它才能打开。

不自动递归清理 runs / datasets。后续清理入口必须区分可重建 cache、已确认提交的暂存文件和实验/正式数据；默认先列出路径、大小、原因。保留期限不是散落在各脚本里的自动删除规则。

## 7. 测试集中且按环境分组

保留当前 `unittest`，无需为目录整理同时切换测试框架。子目录补齐发现所需的 `__init__.py`，`test` 入口按显式组运行，记录通过、失败和跳过数量。

| 位置 | 内容 | 环境 |
| --- | --- | --- |
| `tests/unit/` | 配置、几何、网格数组计算、动作阶段、指标公式、验收规则、存储路径 | 根开发环境；公共契约在相关运行环境补充检查 |
| `tests/integration/planning/` | URDF/TCP/安装一致性、真实 mplib 碰撞 | planning |
| `tests/integration/dataset/` | 官方 LeRobot 写入/读回、schema、追加与收尾恢复 | dataset |
| `tests/integration/meshing/` | TetGen 真实网格生成与基本质量 | meshing，显式可选 |
| `tests/integration/sim/` | Isaac 启动、相机采样、软指安装、原驱动最小运行 | RoboDojo + GPU + 必要资产 |
| `tests/e2e/` | 场景→规划→执行→暂存→LeRobot→独立读取/复核 | 多环境，显式启动 |
| `tests/fixtures/` | 小型网格、布局片段、合成帧、manifest 样例 | 不依赖现有 `data/generated/` |

当前测试按实际行为迁移，而非按文件名猜测：

- `test_soft_grasp_frames.py` 归 planning integration。
- `test_lerobot_recording.py` 归 dataset integration；指定 dataset 组运行时，缺 LeRobot 是环境错误，不能全 skip 后报绿。
- `test_mounted_capture.py` 使用 mock，当前属于 unit；以后真正启动 Isaac 的测试才放 sim integration。
- 相机光流、FEM、碰撞检测中的纯函数测试仍是 unit；需要 OpenCV 的单元组必须声明依赖，不能静默漏测。
- `test_shared.py` 按配置、几何、网格、保留的学习能力拆开；测试不再导入 CLI 文件拿函数。

新增测试聚焦重构风险：跨环境协议、路径解析、数据集往返、重复收尾、失败保留暂存与存储归属。不要为每个薄包装入口写只验证转发参数的测试。

检查真实历史运行数据的 `check_*` 是产品工具，实现放 `validation/`；验证这些工具算法的自动测试放 `tests/`。两者都集中管理，但职责不同。

## 8. 分阶段实施与验收

### 阶段 0：冻结基线和数据清单

记录当前工作区的已有改动，不覆盖正在进行的删除或清理。选取当前安装的 pickup / stack 场景、一个已有录制和一组小型测试输入作为对照，记录命令、有效参数、环境锁、布局/网格摘要及当前通过/失败情况。

盘点 `data/`、根目录媒体、已跟踪的生成物和历史数据集；输出迁移清单。注意 `.gitignore` 忽略 `data/generated/` 不会移除已经跟踪的文件，必须单独核对。

**验收：** 能说明哪些行为和数据需要保持，哪些是旧实现，且基线失败项没有被当成重构引入的新失败。

### 阶段 1：建立 `src` 包和统一入口

先打通 root / planning / RoboDojo / dataset 中的 editable 安装与绝对导入，集中环境调用。准备脚本保持可在干净环境启动；旧命令需要短期兼容时仅留转发壳，并写明移除节点。

**验收：** 各环境能导入所属模块；`--help` 不启动 Isaac；从项目外使用入口绝对路径也能运行；原有纯逻辑回归保持基线结果。

### 阶段 2：优先收拢存储出口

实现 `storage.py`、run manifest、run/dataset ID 与标准目录；所有新输出先接入此模块。改造录制与收尾，维持暂存协议兼容，并补足提交确认、重复收尾和失败恢复。

**验收：** 新运行只写到指定存储根及明确声明的输入资产/第三方缓存位置；根目录不新增媒体；LeRobot 正式输出只有一个位置；转换失败保留暂存，重试不会重复追加 episode。

### 阶段 3：分解主流程、配置和旧代码

按映射拆分任务脚本，统一 X5 安装模型。接入一个任务 JSON + CLI 覆盖，生成完整快照；清除无生产用途的旧配置、抽象框架与重复准备入口。

**验收：** pickup 与 stack 都能解析到可复现的有效配置；规划与仿真安装一致；仍使用原任务与 RewardManager；不通过修改阈值或判定消除基线失败。

### 阶段 4：测试归位与完整流程验证

迁移所有自动测试，分类运行；保留旧报告对照，完成一条跨环境的真实录制、转换、独立读取和复核。复核报告是否通过取决于实际证据，不能把“重构完成”写成“任务一定成功”。

**验收：** unit / planning / dataset 各组明确报告执行情况；可选 GPU 和 meshing 测试未运行时如实标明；正式数据集脱离仓库位置后仍能读取。

### 阶段 5：历史产物迁移与最终清理

提供显式的迁移工具，先 dry-run，再按清单复制和验证：

| 当前位置 | 目标 |
| --- | --- |
| `data/assetes/soft_finger.stl` | `assets/meshes/soft_finger.stl`，修正目录拼写 |
| `data/materials/` | `assets/materials/` |
| `data/generated/` 中的网格 | 经核验后进入 `artifacts/cache/meshes/` |
| `data/generated/` 中的场景、计划、诊断 | 对应的新建历史 run |
| `data/runs/` | `artifacts/runs/legacy_<稳定标识>/` |
| `data/datasets/` | 整体迁入 `artifacts/datasets/lerobot/<dataset_id>/` |
| 根目录 PNG / MP4 | 根据来源进入对应历史 run 的 previews；来源不明则进入标记为 unclassified 的历史 run |

老数据中的 `dataset_root`、布局、URDF/mesh 路径等可能包含旧相对/绝对路径，不能只 `mv`。迁移器生成旧路径→新 ID 的映射并调整项目自有引用，不改写官方视频/Parquet 分片；读取校验通过前保留原件。无法自动分类的内容列入清单，不猜测归属或删除。

最后更新 README，只保留当前入口、环境表、配置规则和存储规则；历史命令注明当时目录和版本。清除已跟踪的生成物时只调整版本管理，不能顺带删掉本地实验数据。移除过渡转发壳，完成顶层 `benchmarks/`、`shared/`、`backends/` 的退出。

**验收：** 新用户从 README 能完成准备、采集、定位 LeRobot 输出和复核；历史数据迁移可追踪；源码目录不包含运行输出。

## 9. 完成标准

- 用户只需记住 `scripts/run.py`，无需手动选择三套解释器或知道包内文件位置。
- 当前主实现全部在 `src/soft_sim/`；公共数学与文件协议不导入重型运行库。
- `configs/` 只含当前入口实际使用的少量配置；上游必要 YAML 与自有配置边界明确。
- 所有自动测试在 `tests/` 分类执行，缺依赖与未运行不能被全 skip 掩盖。
- 所有新生成物进入统一存储根；每个文件都有 run、正式 dataset 或可重建 cache 的归属。
- 给定 run ID 能找到参数、代码和依赖来源、场景、计划、日志、结果、报告，以及 dataset ID / episode 索引。
- LeRobot 数据集能独立移动和读取；无重复正式输出、并发追加冲突或重复收尾生成的新 episode。
- 任务成功、录制提交与质量验收分别记录，保持当前实验结论的证据边界。

---

## 10. 本次实施状态（2026-11-14）

本轮已按下述边界落地，验证只在本机可执行的部分完成。

### 已完成

- `src/soft_sim/` 包建立，`benchmarks/`、`shared/`、`backends/` 作为代码目录退出；历史说明归档到 `docs/archive/`。
- 唯一入口 `scripts/run.py`（仅标准库做初始解析）与命令表 `soft_sim/commands.py`；14 个命令的 `--help` 均不启动 Isaac Sim。
- 环境注册表与子进程调用集中在 `runtime.py`；四个环境已用 `uv pip install -e --no-deps` 装入同一份包，`soft-sim` 也可在 3.10–3.12 解析（`requires-python = ">=3.10"`）。
- `storage.py`：单存储根、run/dataset/cache 三类归属、`run_id`、原子 manifest、storage 相对引用。
- `contracts.py`：采集协议版本、暂存索引校验、数据集兼容签名。
- 录制拆分：`recording/capture.py`（仿真端暂存，不导入 LeRobot）与 `recording/lerobot_io.py`（数据集端写入）通过文件契约协作；`finalize.py` 具备重复收尾与失败保留暂存的语义。
- 配置收敛为“类型化默认值 < 任务 JSON 局部覆盖 < 显式 CLI”；`configs/` 只留 `general_pickup.json`、`stack_blocks.json`；`native_x5` 取代 `profile="panda"` 过渡写法；每次运行写 `config.resolved.json`。
- 主流程拆分为 `tasks/collect.py`（编排）、`sim/robodojo.py`（上游适配与唯一路径例外）、`sim/scene.py`（场景与几何导出）、`planning/grasp.py`（可调用 `plan_scene`）。
- 测试分组：`tests/unit`（73）、`tests/integration/planning`（8）、`tests/integration/dataset`（4）全部通过且无跳过；缺依赖不再静默 skip。
- 新增测试覆盖重构风险：配置优先级、存储归属与 run 不可覆盖、跨进程契约、数据集签名、重复收尾、导入边界（不加载 Torch/mplib/Isaac/LeRobot、不建目录）。
- `migrate` 命令提供 dry-run 的历史产物迁移清单；`data/generated/` 已从版本控制移除（本地文件保留）。
- `preview` 命令除关键帧拼图外，可输出三视角（外部相机 + 左右指内相机）视频，代码使用 H.264（PyAV），标注与页脚取自采集索引和真实任务结果。
- 源网格与材料说明迁到 `assets/`；`import soft_sim` 无重型导入与启动副作用；3.10/3.11 语法编译通过。

### 未在本机验证（需要 GPU / 资产）

- `collect` / `export-scene` / `replay` / `assembly` 的真实 Isaac 场景执行与结果判定。
- `tests/integration/sim`、`tests/e2e`，以及一条完整的“场景→规划→执行→暂存→LeRobot→独立复核”链路。
- 历史数据 `migrate --apply` 的实际复制与独立读取校验（dry-run 已通过）。
