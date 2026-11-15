# 换装第二套机械臂：Franka Panda 上的软指

本页记录把同一套软指装到 Franka Panda 上的实现边界、已验证内容和当前阻塞。
任务 setting 不变：任务类、官方布局、物体、材质、RewardManager 都仍来自
`third_party/RoboDojo/`。

```bash
# Franka 是 configs/ 里的一个可选项；不改配置时默认仍是 x5 机械臂
python scripts/run.py collect --config configs/general_pickup_franka.json --mode export
```

## 实现分层

| 层 | 内容 |
|---|---|
| `src/soft_sim/robots/mechanism.py` | 只放数据：URDF、手/基座连杆、关节顺序、两个滑块的 origin、工具坐标系原点与 STL→滑块坐标系旋转、座面宽度、行程、手掌碰撞网格、自碰撞配置、安装 profile 名 |
| `src/soft_sim/robots/fingers.py` | 全部与机器人无关的数学：安装位姿、楔块与相机代理、FK、安装位姿校验、规划用 URDF/SRDF/TCP |
| `src/soft_sim/robots/x5/` | X5 绑定的兼容封装，保持重构前的函数签名，供旧脚本与旧计划使用 |

`mechanism.rotation` 把 STL 坐标（+X 闭合、+Z 生长）映射到滑块连杆坐标系。X5 生长沿
连杆 X，Panda 生长沿连杆 Z，闭合都沿连杆 Y；两台机器的 finger joint 都是 prismatic，
方向分别朝 -Y（side 0）和 +Y（side 1），所以 `installation_in_link` 只需要换一个旋转矩阵。

## 命令行与配置

- 新增 `task.robot`（`x5` / `franka`），缺省 `x5`，旧配置和旧数据集语义不变；
  `configs/general_pickup_franka.json` 是唯一需要改的文件。`--robot` 只作为命令行显式覆盖，
  与项目"类型化默认值 < 任务 JSON < 显式 CLI"的规则一致。
- `gripper.open_jaw_width_m` 由机制行程决定（X5 0.088、Panda 0.080），超出即拒绝。
- 数据集缺省名从 `x5_<task>_v1` 变为 `<robot>_<task>_v1`；Panda 的
  calibration `robot_type` 是 `panda_softfinger`，X5 仍是 `dual_x5_softfinger`。
- 被换装的臂占用它所替换的那台 X5 的基站，另一台 X5 留在原位当障碍物；这样官方布局的
  物体位置仍在臂展内，`dual_x5_and_franka_competition.yml` 里 Panda 的 `[0, 0.6, 0.765]`
  支撑位（距目标 0.8–1.0 m，超过 Panda 0.855 m 臂展）不使用。
- `scene.json` 新增 `tool_origin_in_hand_m`、`tool_rotation`，规划器不再假设 TCP 沿手 X 方向。
- 规划用 URDF 从上游 `panda.urdf` 生成，网格按上游 OBJ 导出约定解析
  （`<link>_visuals_<link>.obj` / `<link>_collisions_collisions.obj`），因为
  `panda.urdf` 里的 `franka_description/...` 路径在快照里不存在。
- SRDF 只写入 URDF 里真实存在的连杆，上游 `self_collision_ignore` 中的
  `attached_object` 不会变成悬空引用。

## 已验证

| 检查 | 结果 |
|---|---|
| `python scripts/run.py test --group unit` | 92 项通过（含新增 `test_finger_mechanisms.py` 11 项） |
| `python scripts/run.py test --group planning` | 11 项通过（含新增 Panda 规划 URDF/FK/TCP 一致性，容差 2e-7） |
| X5 数值回归 | 安装参数、`finger_in_link`、`tool_meshes`、`fixture_meshes`、`contact_geometry`、`link_transforms`、`mounts_from_q`、规划 URDF/SRDF 与重构前逐位一致（SRDF 仅 reason 字段此前是 `Upstream X5`） |
| Panda 的 USD 关节链 vs `panda.urdf` FK | 在 9 个关节上逐元素一致（0 误差），`Props/*.usd` 的 link 网格与 `meshes/*.obj` 的连杆坐标一致 |
| 静止间隙 | 两台机器上两个指面间隙都等于 `minimum_rest_gap_m`（0.2 mm），指体分处中面两侧 |
| `mplib` 加载 `soft_franka.urdf` | 成功，`soft_grasp_tcp` move group 可用，9 自由度 |
| X5 端到端导出（`--mode export`，run `20261114T232000Z_x5_export`） | `OFFICIAL_TASK_EXPORTED general_pickup 26 obstacles`；mount 校验 6.3e-8 m、1.5e-7；`scene.json` 新增字段按旧值写出 |

## Isaac 实测与阻塞

`--config configs/general_pickup_franka.json --mode export` 在 RTX 4070（8 GB）上跑到：

```
TASK_STAGE before_reset / after_reset / after_apply_saved_poses
TASK_STAGE fem_reset_at_measured_mounts / after_300_steps
```

也就是混合 episode（robot0 = 原双 X5 之一，robot1 = Panda）能建场景、应用官方布局、
在测量安装位初始化 FEM 并稳定跑 300 步。随后的 mount 校验失败，证据在
`artifacts/runs/20261114T230000Z_franka_export/diagnostics/mount_disagreement.json`：

- 手臂链（`panda_link0`…`panda_hand`）的 Isaac 位姿与本项目 URDF FK **逐位一致**，
  说明基座位姿、关节顺序、URDF 与 USD 都对应正确。
- 只有两个 finger link 差约 3.5 mm，且两根滑块的关节值不对称（一根 0.0、一根 0.04）。
- `SOFT_SIM_DEBUG_JOINTS=1` 记录显示 Panda 在 reset 之后就已经不在上游 init 位姿，
  300 步后 `panda_joint6` 到 6.68 rad，超过 `panda.urdf` 给的 -0.0175…3.7525 限位，
  另一根滑块被压到 0；同一 episode 里的 X5 全程精确保持在 `0…0, 0.044, 0.044`。
- PhysX 同时报 `Deformable volume contact buffer overflow`，是被失控手指压到桌面后的
  连锁反应。

结论：**上游 RoboDojo 的 Franka 是 `type: support` 的被动道具**，只出现在
`dual_x5_and_franka_competition.yml` 里，没有经过驱动与限位验证。把它当主动臂用，
需要先修它的 articulation（关节限位/驱动参数），这不是换机械臂的代码能解决的。

## 下一步选项

1. 补 articulation：给 Panda 的 joint 写入 limits 与 drive（上游 USD 缺），再用
   `--mode export` 复验 mount 校验与 FEM 初始状态。
2. 保持被动：Panda 继续当支撑臂，只把软指装在 X5 上（当前主线）。
3. 换一台在 RoboDojo 里已被验证驱动的机械臂：写一个 `FingerMechanism` 表即可，
   `fingers.py`、规划、采集、录制都不用改。

## 已知取舍

- 被换装的臂固定在 X5 基站，官方布局与可达性保持不变；这不是 RoboDojo 官方 Franka 位置。
- Panda 座面宽度按上游 finger plate 取 0.021 m（X5 是 0.024 m），安装 profile 记作
  `derived_wedge_v1`：倾角、前移、闭合间距仍由同一套 STL 几何推出，但**没有实测量测**。
- Panda 手掌网格在生长方向延伸到滑块站位 +7.6 mm，软指底座只领先 5.1 mm；实际硬件需要
  重新确定 standoff，`seat_width_m`/`forward_offset_m` 应按实物复核。
- `validation/compare_sizes.py`（原 X5 手指与软指的尺寸对比图）仍是 X5 专用；
  `check-installation --robot franka` 可用，只是不再画 X5 原手指对照。
