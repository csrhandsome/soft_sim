# Soft Gripper：共享任务层与 Isaac Sim / MuJoCo 后端

> 2026-11-10：当前开发环境已迁移到 `RoboTwin/.venv`（Python 3.10 / SAPIEN）和 `RoboDojo/.venv`（Python 3.11 / Isaac Sim 5.1）。旧的 `mujoco/.venv`、`isaacsim/.venv` 及其 uv 缓存已按要求删除；源码、网格、历史数据保留。新的资产、软指双相机接入与运行方式见 [benchmark 接入说明](benchmarks/README.md)。下文旧后端命令需要重建对应环境后才能使用。

已加入基于 `data/assetes/soft_finger.stl` 的双软手指仿真、指内双相机、形变/力数据采集和图像估力基线。使用与限制见 [软夹爪说明](docs/soft_gripper.md)。

快速运行，以下命令在仓库根目录执行：

```bash
# 已附带默认体网格；修改 STL 或单位后重新生成
uv run --project mujoco --no-sync python scripts/prepare_mesh.py

# MuJoCo：两台相机分别位于软手指底部，跟随各自手指，随机化焦距
MUJOCO_GL=egl uv run --project mujoco --no-sync python mujoco/run.py --episodes 3

# Isaac Sim：需要在能访问 NVIDIA GPU 的宿主机终端执行
uv run --project isaacsim --no-sync python isaacsim/run.py --episodes 3

# SSH 远程可视化：官方 WebRTC 串流，无需 DISPLAY / X11 转发
# 把 SERVER_IP 换成客户端可直达的服务器局域网、VPN 或公网 IPv4
uv run --project isaacsim --no-sync python isaacsim/run.py \
  --livestream --livestream-ip SERVER_IP --episodes 3

# 查看 Isaac 官方资产简称，以及打开现成 Robotiq 模型并检查真实连接点
uv run --project isaacsim --no-sync python isaacsim/inspect_assets.py --list
uv run --project isaacsim --no-sync python isaacsim/inspect_assets.py robotiq_2f85 --gui
```

远程客户端使用 **Isaac Sim WebRTC Streaming Client**，连接上面的 `SERVER_IP`；服务器和网络需允许 TCP `49100`、UDP `47998`。采集结束后保留场景，SSH 终端按 `Ctrl+C` 退出。浏览器 Web viewer 需另行部署，详见[远程可视化说明](docs/soft_gripper.md#ssh-远程可视化webrtc)。

当前夹爪运动采用共享的简化构形预设；官方刚性夹爪资产的检查入口已提供，尚未把完整 Robotiq 闭链机构装配进软指采集场景。Isaac 的 YCB 物体加载已接入，可用 `--config configs/isaac_ycb.json` 选择。默认物体固定，用于控制条件下的挤压力/形变采集，没有执行提起物体验证。

MuJoCo 的体变形、三种物体接触、双相机 RGB/depth 和训练/推理流程已实际运行。Isaac 后端已按照本地 6.2 API 实现并做语法/接口核对；当前执行沙箱无法访问 NVIDIA 驱动，GPU 仿真与官方远程资产加载尚未实测。Isaac 的形变力估计和 MuJoCo 原生接触力在数据中严格区分。

## 原有环境管理

两个项目直接放在根目录下，分别解析依赖、生成锁文件并使用各自的 `.venv`，不使用 uv workspace。

| 目录 | 目标版本（2026-11-08 最新正式版） | Python | 虚拟环境 |
| --- | --- | --- | --- |
| `isaacsim/` | Isaac Sim 6.2.0.0，`all,extscache` | 3.12 | `isaacsim/.venv` |
| `mujoco/` | MuJoCo 4.0.0 | 3.12 | `mujoco/.venv` |

Isaac Sim 自身要求的 MuJoCo 3.12.0 留在 Isaac Sim 环境中；独立 MuJoCo 环境使用 4.0.0。根目录旧的 `.venv` 与 `.uv-cache` 保留，但这两个新项目不使用它们。

## 不使用代理安装

在根目录执行：

```bash
bash scripts/sync_envs.sh
```

也可以分别安装：

```bash
bash scripts/sync_envs.sh isaacsim
bash scripts/sync_envs.sh mujoco
```

脚本先清除大小写的 HTTP、HTTPS、ALL、FTP 代理变量，再设置 `NO_PROXY=*` 和 `no_proxy=*`，然后进入各自目录运行 `uv sync`。同时清除 `VIRTUAL_ENV` 和 `UV_PROJECT_ENVIRONMENT`，确保不会误用当前激活的其他环境。这些设置仅作用于脚本进程，不修改当前终端或系统配置。

Isaac Sim 只声明 `isaacsim[all,extscache]`，由 uv 自动选择版本并解析包括 PyTorch 在内的全部依赖；不逐个固定间接依赖或指定包的来源。解析时同时考虑阿里云镜像和 NVIDIA 源。每个项目的缓存都在自己的 `.uv-cache` 中。两个项目首次成功安装后分别生成 `uv.lock`，后续可使用：

```bash
bash scripts/sync_envs.sh all --locked
```

脚本默认设置 `UV_HTTP_TIMEOUT=600`、`UV_CONCURRENT_DOWNLOADS=2` 和 `UV_HTTP_RETRIES=5`，以减少大型 wheel 下载超时；外部设置的同名变量优先。`UV_HTTP_RETRIES` 需要 uv 0.7.21 或更新版本，旧版使用自身默认重试策略。

`isaacsim/pyproject.toml` 设置 `prerelease = "allow"` 和 `index-strategy = "unsafe-best-match"`，允许 Isaac Sim 所需的 rc 依赖，并在镜像缺少目标版本或平台 wheel 时从 NVIDIA 源选择。平台配置同时要求 Linux x86_64 wheel 可用。

也可以直接让 uv 添加并安装最新版本（与同步脚本不同，直接运行 uv 会继承当前终端的代理设置）：

```bash
uv add --project isaacsim --upgrade-package isaacsim 'isaacsim[all,extscache]'
```

首次解析后由 `uv.lock` 记录具体版本；以后再次执行上面的命令可更新 Isaac Sim 及所需依赖。

如果再次超时，单独重试 Isaac Sim 即可，已完成的下载缓存保留：

```bash
bash scripts/sync_envs.sh isaacsim
```

## 使用

以下命令都在根目录执行。安装完成后使用 `--no-sync`，避免运行时自动触发网络同步：

```bash
# MuJoCo：版本检查与 CPU 自由落体仿真
uv run --project mujoco --no-sync python mujoco/check_env.py

# Isaac Sim：无窗口启动验证，第一次运行按提示确认 NVIDIA EULA
uv run --project isaacsim --no-sync python isaacsim/smoke.py

# Isaac Sim：图形界面
uv run --project isaacsim --no-sync isaacsim
```

也可以分别激活：

```bash
source isaacsim/.venv/bin/activate
# 使用完成后退出，再切换到另一个环境
deactivate
source mujoco/.venv/bin/activate
```

运行 Isaac Sim 前需在宿主机确认 `nvidia-smi` 正常，并满足官方 GPU 与驱动要求。

## 验证状态

MuJoCo 4.0.0 和 Isaac Sim 6.2.0.0 的独立环境与锁文件已存在。MuJoCo CPU 物理与离屏双相机渲染已验证。当前受限执行环境无法访问宿主机 NVIDIA 驱动；Isaac Sim 启动/GPU 仿真仍需在宿主机验证。此限制不意味着宿主机驱动异常。

## 官方来源

- [Isaac Sim 发布版本](https://pypi.org/project/isaacsim/)
- [Isaac Sim 核心依赖](https://pypi.org/pypi/isaacsim-core/6.2.0.0/json)
- [MuJoCo 发布版本](https://pypi.org/project/mujoco/)
- [Isaac Sim Python 安装指南](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html)
- [Isaac Sim 系统要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
