> **归档说明（2026-11-14）**：本文记录的是已移除的实现（RoboTwin / Franka /
> Isaac Sim 6.2 / MuJoCo / 旧 `robodojo_soft_pick.py` 入口）。当前唯一实现路径见
> [根 README](../../README.md)。本文的命令与路径只作为历史证据保留，不能直接运行。

# 精确锁定环境验证：2026-11-11

执行主机：ASUS，项目 `~/DataFromRoot/codes/grade_two/my_projects/robot/soft_sim`。
uv 版本：0.7.20；平台：Linux x86_64。

## 变更范围

- 默认环境为根 `.venv`、`envs/planning/.venv`、`envs/robodojo/.venv`。
- 同步脚本始终使用 `--locked`，默认精确同步，删除了隐式保留额外包的 `--inexact`。
- 保留 checkout 内原来的两个 `.venv` 供回退，验证时没有复用其中的 site-packages。
- RoboDojo 显式声明 `transforms3d==0.4.2` 和 `omegaconf==2.4.0`；版本与原环境相同。
- SAPIEN 3.0.0b1 未在依赖元数据中声明运行时导入的 `pkg_resources`。首次干净安装复现了缺失错误，随后在规划 TOML 中补充 `setuptools==69.5.1`（原环境已验证版本）。
- 两项 benchmark 检查脚本增加 `--output`，将本轮结果和 planner 配置保存到独立目录。
- RoboDojo / RoboTwin 固定提交、IsaacLab / CuRobo 源码快照和兼容补丁保持原配置；资产沿用已下载数据。

本轮验证的是当前软夹爪流程，未安装或验证上游策略评测、语言描述生成、训练、PyTorch3D 或全部官方任务。原 RoboTwin 环境中的 Torch 2.4 / CuRobo 0.7 不是当前 mplib 规划流程的依赖，因此没有复制整套旧环境。

## 首次安装

两个新 benchmark 环境创建前不存在，均由锁文件安装；没有开启系统 site-packages。
根项目另建 `data/runs/clean_env_validation/core` 以验证从空环境安装，默认根 `.venv` 保留。

```bash
bash scripts/clone_benchmarks.sh all --dry-run
bash scripts/sync_envs.sh planning
bash scripts/sync_envs.sh robodojo
UV_PROJECT_ENVIRONMENT="$PWD/data/runs/clean_env_validation/core" \
  uv sync --project . --locked
```

本轮联网下载并安装 Python 包，包括 Torch / CUDA / Isaac Sim 及 extension cache wheel；IsaacLab / CuRobo 使用 `.deps/` 中已有的固定提交快照。未重新下载全部源码和资产，因此不将其称为从空机器完成全量联网部署。驱动、EGL/Vulkan、系统库和 GPU 硬件由服务器提供，未由 uv 管理。

## 回归命令

```bash
SOFT_SIM_TEST_RENDER=1 MUJOCO_GL=egl \
  data/runs/clean_env_validation/core/bin/python -m unittest discover -s tests -v
envs/planning/.venv/bin/python benchmarks/test_soft_grasp_frames.py
envs/planning/.venv/bin/python benchmarks/check_robotwin.py \
  --output data/runs/clean_env_validation/robotwin
envs/planning/.venv/bin/python benchmarks/plan_soft_grasp.py \
  --contact-height .055 --output data/runs/clean_env_validation/plan
envs/robodojo/.venv/bin/python benchmarks/check_robodojo_planner.py \
  --output data/runs/clean_env_validation/robodojo_planner
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python \
  benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/runs/clean_env_validation/plan/plan.json \
  --dt .001 --solver-iterations 80 \
  --output data/runs/clean_env_validation/grasp
envs/robodojo/.venv/bin/python benchmarks/validate_grasp_run.py \
  data/runs/clean_env_validation/grasp
bash scripts/sync_envs.sh all --offline --dry-run
uv pip check --python data/runs/clean_env_validation/core/bin/python
uv pip check --python envs/planning/.venv/bin/python
uv pip check --python envs/robodojo/.venv/bin/python
```

## 实测结果

| 检查 | 结果 |
|---|---|
| 根项目干净安装 | 10 个包，18 项单元/集成测试全部通过，无跳过，包含 MuJoCo 真接触和双相机 EGL 渲染 |
| 规划环境干净安装 | 31 个包，已补齐 SAPIEN 的 setuptools 运行依赖 |
| 规划/仿真坐标回归 | 2 项通过 |
| SAPIEN / mplib | 69 步成功轨迹，480×640×3 RGB 有效 |
| 软夹爪完整规划 | 48 个几何候选，碰撞检查后的接近、试提、携带物体提起轨迹生成成功 |
| RoboDojo 干净安装 | 238 个包，IsaacLab / Assets / CuRobo editable 构建成功 |
| GPU CuRobo | Success，41×7 关节轨迹 |
| 锁文件包清单核验 | 三套新环境无额外包；有固定版本的包逐项相同，editable 包路径匹配 |
| `uv pip check` | 三套新环境全部通过 |
| 精确同步离线预演 | 三个默认项目均无需改动；使用空 XDG_CONFIG_HOME，未依赖用户全局 uv 配置 |

新生成的计划在新 RoboDojo 环境中完成真实 PhysX FEM 抓取，随后独立校验通过：

| 指标 | 本轮结果 |
|---|---|
| 成功标记 | `true` |
| 最大提起 | 82.2877 mm |
| 最终提起 | 77.8616 mm |
| 保持段最低提起 | 77.3136 mm |
| 最终保持时长 | 1.0 s |
| XY 漂移 | 23.0938 mm |
| 双相机 RGB | hold / verify_hold 各两路，标准差均超过 35 |
| 双相机深度 | 每路 65,536 个有限像素（256×256） |

本轮完整仿真约 289 秒。物体保持自由刚体；未绑定、吸附或调整通过阈值。结果覆盖这个单物体回归场景，不代表官方 benchmark 分数或跨物体成功率。
Isaac Sim 日志中保留了上游弃用、材质、Kit 用户目录和 Requests 版本支持等告警；本轮实际规划、物理和相机校验均通过。

环境逐包清单在 `data/runs/clean_env_validation/environment-report.json`；本轮输出在同目录，安装和回归日志在 `data/logs/clean_env_validation/`。

CuRobo 使用动态 editable 版本，uv.lock 不记录其发布版本号；源码提交仍由根 TOML 固定为 `d17b54ce32cba095c0b000c4c58777075d11de0e`，归档构建回退版本为 `0.0.post1.dev100`。这与 PyPI wheel 的版本锁定方式不同。
