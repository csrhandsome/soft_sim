# Docker 构建与跨机复现计划（4070 → RTX 5090 D）

## 1. 背景与决策：为什么从头 build，而不是打包本地 `.venv`

在本地开发机（RTX 4070，Driver `580.178.04`）与目标训练/仿真机（`5090`：RTX 5090 D，Driver `595.58.03`）之间迁移运行 `soft_sim` 时，采用 **Dockerfile + `uv.lock` 从头构建镜像**，而不是直接打包本地 `.venv`：

1. **Docker 不能打包或替换内核显卡驱动**：容器始终共用宿主机的 NVIDIA 驱动（`5090` 上为 `595.58.03`，向下兼容 CUDA 12.x）。
2. **GPU 架构差异（Ada `sm_89` vs Blackwell `sm_120`）**：
   - RTX 5090 D 为 Blackwell 架构（`sm_120`），需要 **CUDA 12.8+** 编译的 PyTorch（`cu128`）及针对 `sm_120` 编译的 `curobo` 算子；本地 `envs/robodojo/uv.lock` 默认锁定的 PyPI `torch==2.7.0` 无法直接覆盖 `sm_120`。
3. **本地 `.venv` 不可直接移植**：
   - 本项目包含多个独立环境（`.venv`、`envs/planning`、`envs/robodojo`、`envs/dataset`、`envs/meshing`），内部含硬编码绝对路径与 `.deps/IsaacLab`、`.deps/curobo` editable 引用。
   - Isaac Sim 5.1 依赖宿主级 Vulkan/OpenGL/X11 动态库，直接拷贝 `.venv` 到裸容器仍会缺失 `.so`。

---

## 2. `5090` 宿主机现状与前置条件

通过 `ssh 5090` 实测确认的目标机环境：

| 项目 | 实测状态 | 说明 |
|---|---|---|
| GPU | `NVIDIA GeForce RTX 5090 D` (32GB) | Blackwell `sm_120`；当前有其他训练进程占用 ~21.4GB |
| 驱动 / CUDA | Driver `595.58.03` / Host `CUDA 12.8` | 驱动支持最高 CUDA 13.2，满足 CUDA 12.8 容器需求 |
| OS / Docker | Ubuntu 24.04 / Docker `29.4.1` | 已配置镜像加速 `https://docker.1ms.run` |
| 磁盘空间 | `/` 剩余 **183GB**（已用 90%） | `robodojo` 完整镜像体积较大（20GB+），需注意清理无用镜像 |
| **阻塞项** | **未安装 `nvidia-container-toolkit`** | `docker run --gpus all` 报错 `failed to discover GPU vendor from CDI` |

### 在 `5090` 上启用 Docker GPU 支持（必须先执行）

```bash
# 1. 安装 nvidia-container-toolkit（Ubuntu 24.04）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 2. 配置 Docker runtime 并重启
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. 验证 GPU 透传
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

---

## 3. `envs/docker/` 文件结构与构建目标

```text
.dockerignore                # 排除本地 .venv、.git、artifacts、大视频与缓存，控制构建上下文在 ~10MB 内
envs/docker/
├── Dockerfile               # 多目标构建：cpu 与 robodojo
└── compose.yml              # 本地与 5090 一键构建/测试/运行编排
```

### 构建目标说明

1. **`target: cpu`**（基础镜像 `ubuntu:24.04`）
   - 安装 `uv`、基础编译/图形库，按 `uv.lock` 同步 `core`（3.12）、`planning`（3.10）、`dataset`（3.11）。
   - 用于快速验证项目核心逻辑与默认回归测试套件：`python scripts/run.py test`（`unit` + `planning` + `dataset`）。
2. **`target: robodojo`**（基础镜像 `nvidia/cuda:12.8.0-devel-ubuntu24.04`）
   - 安装 Vulkan / OpenGL / X11 无头渲染依赖、拉取固定版本第三方源码（`python3 -m soft_sim.sources`：`IsaacLab`、`curobo`、`RoboDojo`），同步全部 4 个必需环境（`bash scripts/sync_envs.sh all`）。
   - 预留 `TORCH_CUDA_ARCH_LIST="8.9;12.0"` 与 `PYTORCH_CUDA_INDEX` 构建参数；针对 **RTX 5090 D (`sm_120`)**，构建时可传入：
     `--build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128`
     将 `envs/robodojo` 中的 `torch==2.7.0` / `torchvision==0.22.0` / `torchaudio==2.7.0` 替换为 CUDA 12.8 wheel。

---

## 4. 构建与测试步骤

### 阶段 A：CPU 回归测试（`unit` + `planning` + `dataset`）

```bash
# 构建 cpu 镜像
docker build -f envs/docker/Dockerfile --target cpu -t soft-sim:cpu .

# 在容器内运行默认测试套件
docker run --rm soft-sim:cpu python scripts/run.py test --verbose
```

或使用 Compose：

```bash
docker compose -f envs/docker/compose.yml run --rm cpu-test
```

### 阶段 B：完整 GPU 仿真环境（`robodojo`，本地 4070 / 目标 5090）

```bash
# 本地 4070 构建
docker build -f envs/docker/Dockerfile --target robodojo -t soft-sim:robodojo .

# 目标 5090 D (sm_120) 构建（替换 torch 为 cu128）
docker build -f envs/docker/Dockerfile --target robodojo \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128 \
  -t soft-sim:robodojo .

# 运行环境自检与 GPU 测试
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$(pwd)/artifacts:/workspace/soft_sim/artifacts" \
  soft-sim:robodojo python scripts/run.py doctor

docker run --rm --gpus all --ipc=host --shm-size=16g \
  soft-sim:robodojo python scripts/run.py test --group gpu-solver --group sim --verbose
```

---

## 5. 验证记录

- [x] `soft-sim:cpu` 镜像构建并在容器内通过 `python scripts/run.py test`（`unit` 105 passed、`planning` 11 passed、`dataset` 4 passed，0 skipped）
- [ ] `soft-sim:robodojo` GPU 镜像构建并在本地 4070 上通过 `doctor` 与 GPU 测试
- [ ] `5090` 宿主机安装 `nvidia-container-toolkit` 并验证 `docker run --gpus all`
- [ ] `soft-sim:robodojo` 在 `5090` 上通过 `doctor` 与 GPU 仿真测试
