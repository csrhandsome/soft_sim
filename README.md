# Soft Sim：RoboDojo 软夹爪

主仿真使用 RoboDojo / Isaac Sim 5.1；软手指、双相机、规划 TCP 与原有测试保持可用。
根目录不再有 `isaacsim/`、`mujoco/` 两个独立项目。保留的后端代码位于 `backends/isaac_sim/`、`backends/mujoco_sim/`，测试没有删减。

## 安装

在 Linux x86_64 上安装 uv，准备 curl，并使用系统 Python 3.12 运行准备脚本。
当前软夹爪运行与回归流程的 Python 依赖由 TOML 和各项目的 `uv.lock` 管理：

| 配置 | Python | 环境 | 用途 |
|---|---|---|---|
| `pyproject.toml` | 3.12 | `.venv` | 共享逻辑、MuJoCo 4.0 回归 |
| `envs/planning/pyproject.toml` | 3.10 | `envs/planning/.venv` | mplib、SAPIEN、软夹爪规划 |
| `envs/robodojo/pyproject.toml` | 3.11 | `envs/robodojo/.venv` | Isaac Sim 5.1、IsaacLab、CuRobo |
| `envs/meshing/pyproject.toml` | 3.11 | `envs/meshing/.venv` | 可选 TetGen 贴合四面体网格生成 |

```bash
bash scripts/clone_benchmarks.sh all
bash scripts/sync_envs.sh all --locked
```

同步脚本默认检查锁文件并精确同步，不保留锁文件以外的包。
两个 benchmark 环境建在 `envs/` 下，与源码 checkout 分离；旧的 `RoboTwin/.venv`、`RoboDojo/.venv` 保留供回退，不再作为默认运行环境。
规划环境只包含本项目实际使用的 SAPIEN / mplib 流程；上游策略评测、语言描述生成、训练及全量任务未纳入这套环境。

RoboDojo / RoboTwin 也使用根 TOML 固定提交的源码快照，目录内没有 `.git` 或 `.gitmodules`。
兼容入口 `scripts/clone_benchmarks.sh` 现在下载提交归档，不执行 Git 克隆或子模块命令；已登记快照重复执行时直接复用，不覆盖未知目录。
IsaacLab 与 CuRobo 由 `scripts/prepare_sources.py` 按固定提交准备到 `.deps/`，再通过 TOML 中的 editable sources 安装。
这里使用源码快照，以保留 IsaacLab 的 config/data 与 CuRobo 的 YAML/CUDA 资源；`.deps/` 内没有 Git 仓库。
IsaacLab 保留已验证的 Starlette 兼容约束；CuRobo 仅补充无 Git 元数据时的版本号。
Isaac Sim 5.1 的跨平台 wheel 元数据有差异，Linux 专用 TOML 显式使用当前 Linux wheel 的依赖元数据，避免错误引入 pywin32。
`patches/` 保留历史子模块移除补丁供查阅；当前归档准备脚本直接移除子模块登记，无需执行 Git 补丁命令；兼容路径 `RoboDojo/third_party/{IsaacLab,curobo}` 是到 `.deps/` 的链接。
策略评测 XPolicyLab 不在当前软夹爪流程内，已取消登记。历史源码备份在 `data/backups/toml-migration-*/`；其中 Git 元数据已按要求清理。后来按用户新要求在 `third_party/` 克隆的软体仿真参考项目保留各自 `.git`，见 [源码清单与阅读记录](third_party/README.md)。

仿真资产另外下载，克隆源码不会自动下载全量数据：

```bash
python3 scripts/download_benchmark_assets.py --prefix Assets/Object/RoboDojo/Clutter/ball/00000
```

已有的机器人、木纹材质、物体资产和抓取计划保持原位置。完整资产说明见 [benchmark 说明](benchmarks/README.md)。

## 验证与运行

三路连续 RGB 与 MP4 采集见 [新软夹爪采集说明](docs/robodojo-rgb-collection.md)，支持两个夹爪相机和外部固定相机同步录制。
STL 表面过疏造成的弯曲显示误差、材料初值与体网格对照见 [形变排查](docs/fem-deformation-2026-11-12.md)。
参考实现、固定区与横梁分区检查，以及贴合体网格实验见 [贴合网格迭代](docs/conforming-fem-2026-11-12.md)。

```bash
# 全部单元测试、MuJoCo 真接触形变、双相机 RGB/depth 渲染
SOFT_SIM_TEST_RENDER=1 MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tests -v

# 规划 URDF 的软指安装坐标和 TCP 一致性
envs/planning/.venv/bin/python benchmarks/test_soft_grasp_frames.py

# 历史成功结果独立复核
envs/robodojo/.venv/bin/python benchmarks/validate_grasp_run.py

# 实际 GPU CuRobo 轨迹规划
envs/robodojo/.venv/bin/python benchmarks/check_robodojo_planner.py

# 新一轮真实 FEM 抓取，不覆盖历史结果
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/soft_grasp_plan_55/plan.json \
  --dt .001 --solver-iterations 80 --output data/runs/soft_planned_grasp_exact
envs/robodojo/.venv/bin/python benchmarks/validate_grasp_run.py data/runs/soft_planned_grasp_exact
```

规划流程、原始结果和限制见 [软夹爪规划](benchmarks/SOFT_GRASP_PLANNING.md)。
仿真物体保持自由刚体，没有绑定或吸附；单物体回归通过不代表跨物体成功率或官方 benchmark 分数。
旧 Isaac Sim 6.2 后端源码保留供回归和参考；当前 RoboDojo 环境固定为 5.1，主入口是 `benchmarks/robodojo_soft_pick.py`。

本次迁移的命令、结果和限制见 [2026-11-11 验证记录](docs/verification-2026-11-11.md)。

精确锁定环境的首次安装与回归结果见 [干净环境验证](docs/clean-environments-2026-11-11.md)。

两个 benchmark 目录去除 Git 元数据的迁移、后续清理与验证见 [普通源码快照迁移](docs/plain-sources-2026-11-11.md)。
