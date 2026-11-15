# syntax=docker/dockerfile:1.7
# Multi-target Dockerfile for soft_sim:
#   - target=cpu      : Lightweight Ubuntu 24.04 image with core, planning, dataset envs
#                       (runs `python scripts/run.py test` -> unit + planning + dataset)
#   - target=robodojo : Full CUDA 12.8 Ubuntu 24.04 image with Isaac Sim 5.1, IsaacLab,
#                       CuRobo, and all 4 envs (supports RTX 40xx sm_89 and 5090 sm_120)

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu24.04
ARG UBUNTU_IMAGE=ubuntu:24.04

# ==============================================================================
# Target 1: cpu (core + planning + dataset)
# ==============================================================================
FROM ${UBUNTU_IMAGE} AS cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=5 \
    MPLCONFIGDIR=/tmp/soft-sim-mpl

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    python3 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /uvx /usr/local/bin/

WORKDIR /workspace/soft_sim

# Copy dependency manifests first for better layer caching
COPY pyproject.toml uv.lock .python-version ./
COPY envs/planning/pyproject.toml envs/planning/uv.lock ./envs/planning/
COPY envs/dataset/pyproject.toml envs/dataset/uv.lock ./envs/dataset/
COPY envs/meshing/pyproject.toml envs/meshing/uv.lock ./envs/meshing/
COPY envs/robodojo/pyproject.toml envs/robodojo/uv.lock ./envs/robodojo/

# Copy source, scripts, configs, tests, and lightweight assets
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY assets ./assets

# Prepare pinned third-party sources (e.g., third_party/RoboDojo URDFs & configs)
RUN PYTHONPATH=/workspace/soft_sim/src python3 -m soft_sim.sources

# Sync core, planning, and dataset environments from locked files
RUN bash scripts/sync_envs.sh core \
    && bash scripts/sync_envs.sh planning \
    && bash scripts/sync_envs.sh dataset

# Download robot URDF/mesh assets needed by unit and planning tests
RUN /workspace/soft_sim/.venv/bin/python scripts/run.py download-assets --prefix Assets/Robots

ENV PATH="/workspace/soft_sim/.venv/bin:${PATH}"

CMD ["python", "scripts/run.py", "test"]

# ==============================================================================
# Target 2: robodojo (CUDA 12.8 + Isaac Sim 5.1 + IsaacLab + CuRobo + all envs)
# ==============================================================================
FROM ${CUDA_IMAGE} AS robodojo

ARG TORCH_CUDA_ARCH_LIST="8.9;12.0"
ARG PYTORCH_CUDA_INDEX=""

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=Y \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=5 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    MPLCONFIGDIR=/tmp/soft-sim-mpl

# System libraries required by Isaac Sim 5.1, Vulkan/OpenGL headless rendering, and CuRobo builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    python3 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxi6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxxf86vm1 \
    libglu1-mesa \
    libvulkan1 \
    vulkan-tools \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /uvx /usr/local/bin/

WORKDIR /workspace/soft_sim

COPY pyproject.toml uv.lock .python-version ./
COPY envs/planning/pyproject.toml envs/planning/uv.lock ./envs/planning/
COPY envs/dataset/pyproject.toml envs/dataset/uv.lock ./envs/dataset/
COPY envs/meshing/pyproject.toml envs/meshing/uv.lock ./envs/meshing/
COPY envs/robodojo/pyproject.toml envs/robodojo/uv.lock ./envs/robodojo/

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY assets ./assets

# Prepare pinned third-party sources (.deps/IsaacLab, .deps/curobo, third_party/RoboDojo)
RUN PYTHONPATH=/workspace/soft_sim/src python3 -m soft_sim.sources

# Sync all 4 required environments (core, planning, robodojo, dataset)
RUN bash scripts/sync_envs.sh all

# Download RoboDojo assets required for simulation and tests
RUN /workspace/soft_sim/.venv/bin/python scripts/run.py download-assets

# Optional: for RTX 5090 (Blackwell sm_120), pass --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128
# to upgrade torch/torchvision/torchaudio in robodojo to CUDA 12.8 wheels.
RUN if [ -n "${PYTORCH_CUDA_INDEX}" ]; then \
      uv pip install --python /workspace/soft_sim/envs/robodojo/.venv/bin/python \
        --reinstall --index-url "${PYTORCH_CUDA_INDEX}" \
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0; \
    fi

ENV PATH="/workspace/soft_sim/.venv/bin:${PATH}"

CMD ["python", "scripts/run.py", "doctor"]
