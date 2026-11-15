# 双机并行验证与 4090 代码同步说明

本地开发机（RTX 4070）负责所有代码修改与版本管理；远程工作站（`ssh 4090`，RTX 4090 24GB）作为独立 GPU 执行节点。
2026-11-15 晚用户授权按显存容量并行验证：本地 8GB GPU 保持单路，4090 先使用 **2 路独立任务队列**。
每个 `collect-batch` 内部仍串行；两队列使用不同 batch ID、run ID 和任务数据集，禁止同时追加同一任务的数据集。
单任务实测显存约 7.6GB、GPU 利用率可达 91%；有剩余显存不代表增加进程必然加速，三路以上需另测峰值和吞吐。

## 1. 远程连接与路径

- **SSH 别名**：`4090`（免密测试：`ssh -o BatchMode=yes 4090 "nvidia-smi"`）
- **远程代码路径**：`/home/sustechdl/code/threecheng/soft_sim`
- **远程 Python 入口**：`/home/sustechdl/code/threecheng/soft_sim/.venv/bin/python scripts/run.py ...`

## 2. 每次改动后的同步方法

所有修改先在本地完成并运行本地单元测试，再增量同步到 `4090`（排除 `__pycache__` 和 `.pyc`）：

```bash
REMOTE=4090:/home/sustechdl/code/threecheng/soft_sim
for dir in src tests scripts configs docs; do
  rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' "${dir}/" "${REMOTE}/${dir}/"
done
```

若 `4090` 尚未生成四面体软指网格（`artifacts/cache/meshes/soft_finger_tetgen_4mm.npz`），可直接同步本地网格缓存：

```bash
ssh -o BatchMode=yes 4090 "mkdir -p /home/sustechdl/code/threecheng/soft_sim/artifacts/cache/meshes"
rsync -az artifacts/cache/meshes/soft_finger_tetgen_4mm.npz \
  4090:/home/sustechdl/code/threecheng/soft_sim/artifacts/cache/meshes/
```

新增任务输入后还需同步官方布局、物体和环境资源。随机变体不仅依赖物体模型，还引用具体
房间、桌面/地面材质和 HDR 背景；只同步代码或 Object 目录仍会在初始化时失败。
下载清单位于 `assets/downloads/`，保留来源与哈希。增量同步这些资源时不使用 `--delete`，
避免删除另一端已有的官方资产，并排除尚未完成的 `*.partial` 下载文件。

## 3. 远程回归测试与任务/批量采集

同步后先跑三组回归测试（`unit` / `planning` / `dataset`）：

```bash
ssh -o BatchMode=yes 4090 "cd /home/sustechdl/code/threecheng/soft_sim && .venv/bin/python scripts/run.py test"
```

在本地和 `4090` 上分别执行不同任务或不同 `collect-batch` YAML 配置即可双机并行加速：

```bash
# 本地（4070）
python scripts/run.py collect-batch --config configs/collection.yaml --batch-id local_batch

# 远程（4090）
ssh -o BatchMode=yes 4090 "cd /home/sustechdl/code/threecheng/soft_sim && .venv/bin/python scripts/run.py collect-batch --config configs/collection.yaml --batch-id remote_4090_batch"
```

4090 两路并发时，分别在两个终端运行不同清单，清单中的任务集合不能重叠。例如本轮的
`artifacts/cache/remaining_verification_20261115/parallel_c.yaml` 和 `parallel_d.yaml`。
不能在两个进程里直接使用同一份启用相同任务的 `collection.yaml`。运行中检查
`nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv`，同时记录任务耗时和失败原因。

若需回收 `4090` 上新生成并提交成功的正式数据集，可按任务目录拉回本地：

```bash
rsync -az 4090:/home/sustechdl/code/threecheng/soft_sim/data/datasets/<task>/ data/datasets/<task>/
```
