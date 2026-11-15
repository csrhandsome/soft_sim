# 正式数据集交付与验收说明

本文档说明符合 [目标定义](goal.md) 的正式 LeRobot v3 数据集目录结构、真实采集与验收命令、当前已入库正式样本验收结果，以及完整性与质量边界。

> [!NOTE]
> 截至 2026-11-16 00:21 CST，`data/datasets/general_pickup/` 已通过 4090 默认 TPU 90A + cuRobo 原驱动实测入库 **4 条**正式样本（`episode_index=0..3`：`s22`、`s6`、`s43`、`s17`，每条 208 帧，共 **832 帧**，20 fps，`task_success=true`），并通过 `validate-dataset --task general_pickup --require-fem` 全帧 MP4 与数值同步验收。历史 75A 集合保留在各自原目录名下，不迁移冒充当前样本；`artifacts/datasets/lerobot/diagnostic_pickup_curobo_20261115` 为诊断集合。

## 1. 当前存储布局与已入库正式样本

- **逻辑路径**：`data/datasets/<task_name>/`（项目内保持不变）
- **本机物理存储（软链接）**：`data/datasets` -> `/media/three/HIKSEMI/soft_sim/datasets`（由本机标记 `data/.datasets_storage_target.json` 记录挂载点 `/media/three/HIKSEMI` 与目标路径；未配置外盘标记的环境如 4090 仍直接使用本地实体目录，互不影响）
- **挂载与断链防护**：`collect` preflight（`dataset_root_for`）与 `finalize` 写入前均调用 `ensure_formal_datasets_ready()`，并使用 `formal_datasets_lock()` 串行化提交；若 HIK 移动固态未挂载或 `data/datasets` 为悬空软链接，将直接拒绝正式数据写入并报错，不会在系统盘伪造目录。
- **验收报告**：[general_pickup_report.json](../artifacts/cache/dataset_delivery_20261115/general_pickup_report.json)（`valid: true`，`total_episodes: 4`，`total_frames: 832`，`fps: 20`，`fem_check_mode: schema_only`）
- **已入库样本明细**：
  1. **`episode_index=0`（seed 22，`active_arm=1`）**：来源 `artifacts/runs/goal_dataset_delivery_pickup_s22_20261115`（`object_lift=0.133339703 m`，208 帧，`sim_time_alignment: matched_provenance`，`task_success=true`）。
     - 独立离线质量复核（`validation_id: 20261115T150839Z`）：原始报告 [acceptance.json](../artifacts/runs/goal_dataset_delivery_pickup_s22_20261115/reports/20261115T150839Z/acceptance.json)，正式侧car [acceptance.json](../data/datasets/general_pickup/meta/simulation/quality/episode_000000/acceptance.json)；`quality_valid=false`（`sensor_quality`、`cooking_setup` 未通过，其余 7/9 通过）。
     - MP4 分片：左 [file-000.mp4](../data/datasets/general_pickup/videos/observation.images.left_rgb/chunk-000/file-000.mp4)、右 [file-000.mp4](../data/datasets/general_pickup/videos/observation.images.right_rgb/chunk-000/file-000.mp4)、外 [file-000.mp4](../data/datasets/general_pickup/videos/observation.images.external_rgb/chunk-000/file-000.mp4)（各 208 帧，`max_pts_error_s=0.0`）。
  2. **`episode_index=1`（seed 6，`active_arm=1`）**：来源 `artifacts/runs/gpu_campaign_pickup_s6_20261115`（`official_reward_success=true`，`object_lift=0.139496922 m`，208 帧，`sim_time_alignment: matched_provenance`，`task_success=true`；独立离线质量尚未复核，`quality_valid=null`）。
     - MP4 分片：左 [file-001.mp4](../data/datasets/general_pickup/videos/observation.images.left_rgb/chunk-000/file-001.mp4)、右 [file-001.mp4](../data/datasets/general_pickup/videos/observation.images.right_rgb/chunk-000/file-001.mp4)、外 [file-001.mp4](../data/datasets/general_pickup/videos/observation.images.external_rgb/chunk-000/file-001.mp4)（各 208 帧，`max_pts_error_s=0.0`）。
  3. **`episode_index=2`（seed 43，`active_arm=0`）**：来源 `artifacts/runs/gpu_campaign_pickup_s43_20261115`（`official_reward_success=true`，`object_lift=0.133744419 m`，208 帧，`sim_time_alignment: matched_provenance`，`task_success=true`；独立离线质量尚未复核，`quality_valid=null`）。
     - MP4 分片：`observation.images.{left_rgb,right_rgb,external_rgb}/chunk-000/file-002.mp4`（各 208 帧，`max_pts_error_s=0.0`）。
  4. **`episode_index=3`（seed 17，`active_arm=0`）**：来源 `artifacts/runs/gpu_campaign_pickup_s17_20261115`（`official_reward_success=true`，`lift=0.037284 m` / `verify_hold=5.45e-05 m`，208 帧，`sim_time_alignment: matched_provenance`，`task_success=true`；独立离线质量尚未复核，`quality_valid=null`）。
     - MP4 分片：`observation.images.{left_rgb,right_rgb,external_rgb}/chunk-000/file-003.mp4`（各 208 帧，`max_pts_error_s=0.0`）。

## 2. 正式目录结构与分片规则

所有默认（`--success-only` 且 `--joint-control position`）采集的成功且完整 episode 直接保存到：

```text
data/datasets/<task_name>/                  # 每个任务目录本身即独立完整的 LeRobot v3 根目录
├── meta/
│   ├── info.json                           # LeRobot v3.0 元数据、fps 与 features 定义
│   ├── tasks.parquet
│   ├── stats.json
│   ├── episodes/chunk-000/file-000.parquet # 每条 episode 的帧区间与视频时间范围索引
│   └── simulation/
│       ├── dataset_signature.json          # 兼容性签名（材料、安装、分辨率、FPS、任务对象）
│       ├── collection_summary.json         # 成功/失败 episode 索引与 active_arm 汇总
│       └── episode_000000.json             # 单条 episode 仿真溯源与任务结果
├── data/
│   └── chunk-000/file-000.parquet          # 同步的关节状态、动作、物体状态、FEM 等数值列
└── videos/
    ├── observation.images.left_rgb/
    │   └── chunk-000/file-000.mp4          # 左指内相机 MP4 分片
    ├── observation.images.right_rgb/
    │   └── chunk-000/file-000.mp4          # 右指内相机 MP4 分片
    └── observation.images.external_rgb/
        └── chunk-000/file-000.mp4          # 外部相机 MP4 分片
```

- `dataset_id` 仅作身份标签记录在元数据中，**不会**在 `data/datasets/<task_name>/` 下额外嵌套子目录。
- 双臂机器人（`dual_x5_softfinger`，完整 16 关节顺序与逐帧 `finger_camera_to_world` 一致）跨 seed 切换左/右活动臂（`active_arm` 0 或 1）属于单条 episode 差异，不阻碍追加到同一 `data/datasets/<task_name>/`；每条 episode 的 `active_arm` 保存在 `meta/simulation/episode_XXXXXX.json` 与 `collection_summary.json`。
- LeRobot v3 的视频按分片（shard，如 `chunk-000/file-000.mp4`）存储，多条 episode 可能共享同一个 MP4 分片；每条 episode 在该分片内的时间范围由 `meta/episodes/` 中的 `videos/observation.images.<cam>/from_timestamp` 与 `to_timestamp` 精确索引。

## 3. 可执行采集、恢复与验收命令

### 单任务采集（GPU / Isaac 环境）

```bash
python scripts/run.py collect --task general_pickup --seed 22 --material bambu_tpu_90a
```

### 批量采集（YAML 清单）

```bash
# 先干跑检查输入与布局是否就绪（已实测输出 {"ready": 2}）
python scripts/run.py collect-batch --config configs/collection.yaml --dry-run

# 正式串行执行（仅启用已适配且准备好的任务）
python scripts/run.py collect-batch --config configs/collection.yaml
```

### 断点恢复提交（针对已成功暂存或中断转换的 run）

```bash
# 对指定 run_id 执行安全提交/恢复（幂等，不重复追加已写入的 episode）
python scripts/run.py finalize --run-id <run_id>

# 若远端 4090 上出现 task_success=true 但因旧版签名/转换失败保留的 run（仅在远端无写者空档拉取 run 后恢复）：
rsync -av 4090:/home/sustechdl/code/threecheng/soft_sim/artifacts/runs/<run_id>/ artifacts/runs/<run_id>/
python scripts/run.py finalize --run-id <run_id>
python scripts/run.py validate-dataset --task <task_name> --require-fem
```

### 数据集交付验收（CPU / `dataset` 环境）

```bash
# 对正式任务根目录执行全量验收（含元数据、episodes 索引、轻量列 shape/有限值与逐帧 PTS/MP4 解码）
python scripts/run.py validate-dataset --task general_pickup --require-fem --output artifacts/cache/dataset_delivery_20261115/general_pickup_report.json

# 或直接指定任意 LeRobot v3 根目录
python scripts/run.py validate-dataset artifacts/datasets/lerobot/diagnostic_pickup_curobo_20261115 --require-fem
```

### 单次运行详细质量复核（可选独立标签）

```bash
python scripts/run.py validate --run-id <run_id>
```

## 4. 成功、完整性与质量三者的区分

1. **`task_success`（任务成功）**：仅由上游 RoboDojo `RewardManager` 判定。`success_only=True` 时，失败任务在 `EpisodeRecorder.close` 阶段直接标记 `discarded` 并清理暂存流，不进入 `data/datasets/<task_name>/`。
2. **数据集完整性（`validate-dataset` / `finalize` 提交前校验）**：
   - 正式采集（`success_only=True`）强制要求 `observation.state`、`action`、`left_rgb`、`right_rgb`（及声明的 `external_rgb`）全部存在；仅显式 `success_only=False` 的最小诊断采集才允许省略 `action`。
   - `validate-dataset` 逐帧核对视频 PTS 与 `from_timestamp + frame_index / fps` 对齐（拒绝缺 PTS、重复或非单调 PTS、帧间隔漂移），核对 parquet `timestamp` 与 `frame_index / fps` 对齐、`observation.sim_time_s` 与 `provenance.frames.time_s` 对齐（无溯源时标记 `unverified_no_provenance`），并核对状态/动作列的 shape 与有限值。
   - `finalize` 在写入 LeRobot v3 后先执行上述完整校验，全部通过后才标记 `committed` 并删除 `capture.partial.npy`；失败则保留暂存流与 `finalize_error`。
3. **`quality_valid`（详细离线质量标签）**：
   - `validate-dataset` 只读读取数据集自身溯源或关联运行目录 `artifacts/runs/<run_id>/reports/*/acceptance.json` 中的 `quality_valid` 标签（独立质量结论以对应 `acceptance.json` 为准，不修改原证据）。
   - 例如对 `diagnostic_pickup_curobo_20261115` 执行 `validate-dataset` 返回 `valid: true` 仅代表基础 LeRobot v3 完整性通过；其关联的 [`acceptance.json`](../artifacts/runs/verify_pickup_curobo_20261115/reports/independent_spot/acceptance.json) 明确记录 `quality_valid: false`（`sensor_quality`、`cooking_setup` 未通过）。详细质量检查不作为“原任务成功且基本数据完整入库”的统一前置门槛。

## 5. 已知限制与崩溃恢复边界

- **跨进程与跨机断点恢复**：`finalize` 在调用 `save_episode()` 前先在 `episode.json` 与 `meta/simulation/.pending_episode_XXXXXX.json` 中持久化预留 `episode_index`、`run_id` 与 `frame_count`。若在 `save_episode()` 完成后、`committed` 前进程崩溃（或从 4090 拉取现场到本机恢复），重试 `finalize` 会凭 `run_id` 与 `frame_count`（而非仅凭 `frame_count` 或绝对 `capture_dir`）识别已写入的 episode 并完成校验与提交，避免重复追加；`_resolve_layout()` 按 `third_party/`（或首个 `RoboDojo/`）映射远端 layout 路径并强制校验 `layout_sha256`，缺失官方布局直接拒绝。
- **正式存储守卫作用域与防回落**：`ensure_formal_datasets_ready()` 仅约束正式 `data/datasets` 目标（`artifacts/` 诊断集合与 `/tmp` 测试不受 HIK 挂载约束）；当 `data/.datasets_storage_target.json` 存在但 `data/datasets` 为悬空软链接或被误建为普通实体目录时，直接拒绝写入，不会回落系统盘。`EpisodeRecorder` 保留逻辑 `data/datasets/<task>` 相对路径，仅在 `finalize` 写入时解析软链接。
- **并发读写锁**：`finalize` 与 CLI `validate-dataset`（校验正式 `data/datasets` 时）共用 `formal_datasets_lock()`（`artifacts/cache/.formal_datasets.lock`）；`validate_dataset()` 函数本身不重入该锁，避免 `finalize` 内部提交前校验自锁死。
- **未覆盖的极端边界**：若进程在 `LeRobotDataset.save_episode()` 内部向同一 parquet/mp4 分片写入到一半时被 `SIGKILL` 强杀，LeRobot v3 分片文件本身不具备事务回滚能力；此时需人工检查或清理未闭合分片后重跑 `finalize`。
- **大数组读取策略**：`validate-dataset` 通过 `pyarrow.parquet` 仅读取轻量标量/状态/动作列，并通过 `schema_arrow` 检查 FEM 列声明（`fem_check_mode="schema_only"`），避免逐帧反序列化几百 MB 的 `finger_nodes_*` 数组拖垮验收；同时强制校对 `provenance.frames` 长度与 `observation.sim_time_s` 同步。
- **其余任务适配边界**：当前仅 `general_pickup`（seed 22）已有 90A 正式入库样本；其余任务与新 seed 仍由另一 GPU 会话推进，截至 2026-11-15 23:02 CST 尚无新的 `task_success=true` 远端/本地 `gpu_campaign_*` run 待接收。
