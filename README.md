# Soft Sim：RoboDojo 软夹爪

主仿真使用 RoboDojo / Isaac Sim 5.1；软手指、双相机、规划 TCP 与原有测试保持可用。
根目录不再有 `isaacsim/`、`mujoco/` 两个独立项目。保留的后端代码位于 `backends/isaac_sim/`、`backends/mujoco_sim/`，测试没有删减。

## 安装

在 Linux x86_64 上安装 uv，准备 git、curl，并使用系统 Python 3.12 运行准备脚本。
所有 Python 依赖由 TOML 和各项目的 `uv.lock` 管理：

| 配置 | Python | 环境 | 用途 |
|---|---|---|---|
| `pyproject.toml` | 3.12 | `.venv` | 共享逻辑、MuJoCo 4.0 回归 |
| `envs/planning/pyproject.toml` | 3.10 | `RoboTwin/.venv` | mplib、SAPIEN、软夹爪规划 |
| `envs/robodojo/pyproject.toml` | 3.11 | `RoboDojo/.venv` | Isaac Sim 5.1、IsaacLab、CuRobo |

```bash
bash scripts/clone_benchmarks.sh all
bash scripts/sync_envs.sh all --locked
```

克隆脚本按根 TOML 的提交固定 RoboDojo / RoboTwin，从不初始化子模块，也不覆盖已有 checkout 的提交或本地改动。
IsaacLab 与 CuRobo 由 `scripts/prepare_sources.py` 按固定提交准备到 `.deps/`，再通过 TOML 中的 editable sources 安装。
这里使用源码快照，以保留 IsaacLab 的 config/data 与 CuRobo 的 YAML/CUDA 资源；`.deps/` 内没有 Git 仓库。
IsaacLab 保留已验证的 Starlette 兼容约束；CuRobo 仅补充无 Git 元数据时的版本号。
Isaac Sim 5.1 的跨平台 wheel 元数据有差异，Linux 专用 TOML 显式使用当前 Linux wheel 的依赖元数据，避免错误引入 pywin32。
`patches/` 记录上游 checkout 的子模块移除改动；兼容路径 `RoboDojo/third_party/{IsaacLab,curobo}` 是到 `.deps/` 的链接。
策略评测 XPolicyLab 不在当前软夹爪流程内，已取消登记。历史 Git 对象与原工作目录在 `data/backups/toml-migration-*/`。

仿真资产另外下载，克隆源码不会自动下载全量数据：

```bash
python3 scripts/download_benchmark_assets.py --prefix Assets/Object/RoboDojo/Clutter/ball/00000
```

已有的机器人、木纹材质、物体资产和抓取计划保持原位置。完整资产说明见 [benchmark 说明](benchmarks/README.md)。

## 验证与运行

```bash
# 全部单元测试、MuJoCo 真接触形变、双相机 RGB/depth 渲染
SOFT_SIM_TEST_RENDER=1 MUJOCO_GL=egl .venv/bin/python -m unittest discover -s tests -v

# 规划 URDF 的软指安装坐标和 TCP 一致性
RoboTwin/.venv/bin/python benchmarks/test_soft_grasp_frames.py

# 历史成功结果独立复核
RoboDojo/.venv/bin/python benchmarks/validate_grasp_run.py

# 实际 GPU CuRobo 轨迹规划
RoboDojo/.venv/bin/python benchmarks/check_robodojo_planner.py

# 新一轮真实 FEM 抓取，不覆盖历史结果
OMNI_KIT_ACCEPT_EULA=YES RoboDojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/soft_grasp_plan_55/plan.json \
  --dt .001 --solver-iterations 80 --output data/runs/soft_planned_grasp_toml
RoboDojo/.venv/bin/python benchmarks/validate_grasp_run.py data/runs/soft_planned_grasp_toml
```

规划流程、原始结果和限制见 [软夹爪规划](benchmarks/SOFT_GRASP_PLANNING.md)。
仿真物体保持自由刚体，没有绑定或吸附；单物体回归通过不代表跨物体成功率或官方 benchmark 分数。
旧 Isaac Sim 6.2 后端源码保留供回归和参考；当前 RoboDojo 环境固定为 5.1，主入口是 `benchmarks/robodojo_soft_pick.py`。

本次迁移的命令、结果和限制见 [2026-11-11 验证记录](docs/verification-2026-11-11.md)。
