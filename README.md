# Soft Sim：RoboDojo 双 X5 软指仿真

本项目只有一条实现路径：**RoboDojo 原任务场景 + Isaac Sim 5.1 + 原双 X5，仅替换末端软指**。
任务类、官方布局、机器人资产、物体与材质、成功／评分条件全部来自固定提交的 RoboDojo 快照
（`third_party/RoboDojo/`）。软指网格、安装倾角与楔块、指内相机、运动规划、LeRobot v3
采集和独立质量复核是本项目新增的部分。第二套滑块（Franka）的换装路径见
[Franka 说明](docs/franka-softfinger.md)。

软指默认材料为 **拓竹 TPU 90A 西瓜红（51901 / Blaze）**，密度 1240 kg/m³、
杨氏模量 5.3 MPa（官方 TDS XY 值的各向同性近似），配色为深红端 `#D21B3C`。
75A、85A、90A 采用同一套资产格式，来源和假设见 [资产说明](assets/README.md)。
85A 沿用西瓜红自定义外观，可用 `--material bambu_tpu_85a` 切换；默认仍为 90A。

- 用户入口只有一个：`scripts/run.py`。实现全部在 `src/soft_sim/`。
- 默认机器人是原双 X5，安装固定为 `measured_wedge_v3`（0.75 倍、双侧内倾约 12°、楔块）。
- 换装层按 `FingerMechanism` 表组织（`src/soft_sim/robots/mechanism.py`）：机械臂、关节顺序、
  滑块站位、工具坐标系与座面宽度都在表里，安装与规划数学与机器人无关。选哪套滑块是配置里的
  一个字段（`task.robot`），缺省 `x5`；`configs/general_pickup_franka.json` 就是 Franka 的可选项。
  上游 Panda 尚未通过驱动验证，见 [Franka 说明](docs/franka-softfinger.md)。
- 默认通过原机械臂驱动执行；直接改写关节状态的回放只用于诊断。
- 任务成功由 RoboDojo 原 `RewardManager` 判定，场景加载或规划成功都不算成功。
- 任务成败、录制提交和质量验收是三个独立字段，不互相冒充。

## 快速开始

```bash
bash scripts/sync_envs.sh all          # 四个必需环境
bash scripts/sync_envs.sh meshing      # 可选：TetGen 网格
python scripts/run.py doctor           # 解释器、包、源码与资产状态
python scripts/run.py test             # unit + planning + dataset 三组回归
```

`sync_envs.sh` 按各环境自己的锁文件同步，然后把同一份 `soft-sim` 以 editable 方式装入
每个解释器；环境之间不共享 NumPy 版本，也不做统一求解。

```bash
python scripts/run.py collect --config configs/general_pickup.json --seed 22 \
  --dataset-id x5_pickup_v1
python scripts/run.py validate --run-id <run_id>
python scripts/run.py preview  --run-id <run_id>
# 三视角派生视频（桌面相机 + 左右指内相机），写到指定路径
python scripts/run.py preview --run-id <run_id> --video data/generated/videos/three_views.mp4
python scripts/run.py finalize --run-id <run_id>
```

每条命令结束时打印 `run_id`、run 目录、正式数据集位置与 episode 索引（如果产生了数据集）。
只导出场景或做安装检查的命令不会打印虚构的数据集路径。
三视角视频及同名 JSON 统一保存在 `data/generated/videos/`（HIKSEMI 硬盘），生成方法见 [简短说明](docs/three-view-video.md)。

软指默认采用 **80 次迭代＋固定区消元**（1 ms、节点阻尼 20），原完整碰撞网格保留。
普通采集会自动启用；对照时可加 `--no-fixed-cell-elimination`。配置与验证见
[默认软指方案](docs/default-softfinger.md)。

## 命令

| 命令 | 用途 | 环境 |
|---|---|---|
| `collect` | 原场景规划、执行、暂存、转换 | RoboDojo → planning / dataset |
| `export-scene` | 只导出场景几何 | RoboDojo |
| `replay` | 在匹配场景执行已保存计划 | RoboDojo |
| `assembly` | 原驱动全行程安装诊断 | RoboDojo |
| `plan` | 对已导出场景做 mplib 规划 | planning |
| `check-installation` | 安装几何、间隙与碰撞检查 | planning |
| `prepare-mesh` | 生成贴合四面体网格 | core / meshing |
| `finalize` | 已暂存采集写入 LeRobot | dataset |
| `validate` | 组织数据、相机、FEM、碰撞与任务验收 | core 编排 |
| `preview` | 从运行生成派生预览图或三视角视频 | dataset |
| `download-assets` | 下载固定输入资产 | core |
| `setup-sources` | 准备固定版本的第三方源码 | core |
| `migrate` | 列出／复制重构前生成物（默认 dry-run） | core |
| `test` | 按组执行测试 | 按组分发 |
| `doctor` | 检查解释器、包、源码与路径 | core |

## 环境

| 环境 | Python | 关键依赖 | 职责 |
|---|---|---|---|
| `.venv`（core） | 3.12 | NumPy 2.3.4、SciPy 1.16.3 | 配置、存储、调度、单元测试 |
| `envs/planning` | 3.10 | mplib、SAPIEN、NumPy 1.26.4 | 碰撞规划与安装检查 |
| `envs/robodojo` | 3.11 | Isaac Sim 5.1、IsaacLab、CuRobo | 任务场景、物理仿真、采集 |
| `envs/dataset` | 3.11 | LeRobot 0.4、NumPy 2.x | 数据集读写与复核 |
| `envs/meshing` | 3.11 | TetGen、PyVista | 可选网格生成；不在 `all` 中 |

解释器路径只由 `src/soft_sim/runtime.py` 提供，其他模块不得拼接 `.venv/bin/python`。
需要外部解释器时用 `SOFT_SIM_<ENV>_PYTHON` 显式覆盖，不会静默退回系统 Python。

## 配置只有一套规则

```text
代码中的类型化默认值  <  一个任务 JSON 的局部覆盖  <  显式 CLI 参数
```

- 嵌套对象按字段合并，数组整体替换；未知字段、类型错误、非法范围立即报错。
- CLI 未指定的参数不会用 argparse 默认值覆盖配置文件。
- 校验在启动 Isaac Sim 之前完成。
- 每次运行写 `config.resolved.json`，记录完整有效参数与 CLI 覆盖。

`configs/` 只保留当前任务：`general_pickup.json`、`stack_blocks.json`，以及把被换装机械臂
换成 Franka 的 `general_pickup_franka.json`（只多 `task.robot` 与数据集名）。上游 RoboDojo 的
`env_cfg/**/*.yml` 与 CuRobo YAML 由适配层读取，不复制成自有配置。

## 生成物只有一个出口

默认存储根是 `artifacts/`，可用 `--storage-root /path/to/storage` 整体切换。

```text
artifacts/
├── runs/<run_id>/                  单次执行：配置、场景、计划、暂存、报告、日志、预览
├── datasets/lerobot/<dataset_id>/  正式 LeRobot 数据集
└── cache/                          可重建的网格／资产缓存
```

`run_id` 形如 `20261114T080000Z_general-pickup_s22_a1b2c3`，创建时确定，不因成败改名；
同名 run 目录禁止覆盖，重跑产生新 run。一次 run 的结构：

```text
manifest.json            执行/采集/任务/验收状态与产物索引
config.resolved.json     完整参数快照
scene/                   场景与几何（每个 operation 一份）
plans/operation_000/     候选、plan、轨迹、碰撞代理
capture/                 episode.json、finger_reference.npz、capture.partial.npy
results/task.json        RewardManager 结论与检查点
reports/<validation_id>/ 每次复核单独一份，不覆盖旧证据
diagnostics/             无效 FEM 样本、静态探针
previews/<preview_id>/   派生预览，不进入训练帧
logs/                    子进程日志
```

`dataset_id` 是显式集合名（例如 `x5_pickup_v1`），seed 是集合内的 episode。首次创建集合时
记录 schema／安装／相机／FPS 兼容签名；签名不符时拒绝追加，要求新的 dataset ID。
同一个集合同一时间只允许一个收尾写入器。

## 状态与失败恢复

| 维度 | 取值 | 含义 |
|---|---|---|
| 执行 | created / running / completed / failed | 流程是否正常结束 |
| 采集 | none / capturing / incomplete / staged / converting / committed / failed | 暂存与正式写入进度 |
| 任务 | `task_success: true / false / null` | 官方 RewardManager 结论 |
| 复核 | pending / passed / failed | 相机、数据、FEM、碰撞检查 |
| 验收 | `accepted` | 任务成功且必要复核全部通过 |

- 默认仅提交原任务判定成功的 episode（`recording.success_only: true`）。任务失败时删除暂存
  帧和 FEM 参考文件，保留结果与日志，采集状态为 `discarded`；不进入正式数据集。
- 诊断时可显式使用 `--no-success-only` 保留完整失败 episode，不应与仅成功的集合混用。
- 未完成或损坏的采集保留在 run 的暂存区，不提交为正式 episode。
- `finalize` 可重试：已提交的 run 返回原 episode 引用，不会重复追加；写入确认前不删除
  `capture.partial.npy`。

## 测试

```bash
python scripts/run.py test --group unit       # core
python scripts/run.py test --group performance # core，采集代码优化前后计时及等价性检查
python scripts/run.py test --group gpu-performance # RoboDojo，需指定成功 run，完整回放 A/B
python scripts/run.py test --group planning   # planning
python scripts/run.py test --group dataset    # dataset
python scripts/run.py test --group sim        # RoboDojo + GPU + 资产
python scripts/run.py test --group meshing    # 可选
```

缺依赖或未运行时如实报告，不会被全 skip 掩盖成通过。

采集优化的范围、复现方式和计时报告见 [采集性能](docs/collection-performance.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/run.py` | 唯一调度入口（仅标准库做初始解析） |
| `scripts/sync_envs.sh` | 环境同步入口 |
| `src/soft_sim/cli/` | 命令实现 |
| `src/soft_sim/tasks/` | collect / assembly / 预览编排 |
| `src/soft_sim/sim/` | Isaac 启动、相机、FEM 软指、场景导出、上游适配 |
| `src/soft_sim/robots/x5/` | X5 安装几何、运动学、URDF/TCP 生成 |
| `src/soft_sim/planning/` | mplib 规划与候选生成 |
| `src/soft_sim/recording/` | 仿真端暂存、数据集端写入、收尾与读取 |
| `src/soft_sim/validation/` | 质量指标、碰撞与验收实现 |
| `src/soft_sim/{config,contracts,geometry,mesh,fem_reduction,storage,runtime,execution}.py` | 公共基础模块 |
| `assets/meshes/` | 受版本控制的源网格 |
| `third_party/RoboDojo/` | 固定提交的上游快照 |
| `docs/` | 当前说明、[目标](docs/goal.md)、[重构方案](docs/refactor.md)；历史记录在 `docs/archive/` |

`import soft_sim` 不加载 Torch、mplib、SAPIEN、Isaac Sim 或 LeRobot，也不创建目录。

## 已移除的实现

为避免同一目标存在多套入口：`benchmarks/`、`shared/`、`backends/` 作为代码目录退出
（历史说明归档到 `docs/archive/`）；RoboTwin、旧 Franka 规划路径、MuJoCo 后端、Isaac Sim 6.2
后端均已删除，`--mount-profile parallel_legacy` 不再存在。旧平行安装的成败数据保留其原安装
配置，不混作当前安装姿态的验证。

`--robot` 现在只表示"被换装的滑块属于哪台机械臂"：`x5` 是验证过的默认路径，
`franka` 走同一套安装与规划数学，**但上游 Panda 的关节驱动尚未通过验证**，
当前跑不出可用的采集结果，原因与证据见 [Franka 说明](docs/franka-softfinger.md)。
