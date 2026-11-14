# TPU 75A、Orange 021 U 与 LeRobot v3

> **路径变更（2026-11-14 重构）**：日常入口改为 `python scripts/run.py <command>`，实现位于 `src/soft_sim/`，生成物位于 `artifacts/`。本文保留写作时的目录与命令，当前命令见 [README](../README.md)。


## 材料来源

拓竹官方产品为 **TPU 75A (2.85mm)**。数据来自产品页面链接的
[TDS V1.0](https://store.bblcdn.eu/s8/default/6c98f7913de24007b54cb75052cd39b5/Bambu_TPU_75A-TDS.pdf)，
本地原始 PDF 与提取文本统一保存在 `assets/materials/bambu_tpu_75a.tds.pdf`
和 `assets/materials/bambu_tpu_75a.tds.txt`，参数预设为同目录的 `bambu_tpu_75a.json`。

| 参数 | 官方 TDS |
| --- | --- |
| 密度，ISO 1183 | 1.17 g/cm³ = 1170 kg/m³ |
| 硬度，ASTM D2240 | 76 ± 1 Shore A，产品标称 75A |
| XY 拉伸强度 | 11.50 ± 0.50 MPa |
| XY 断裂伸长率 | >800% |
| Z 撕裂强度 | 14.00 ± 0.80 N/mm |
| 球回弹率 | 50% |
| 静态压缩强度 | 174 kPa |
| 杨氏模量 XY/Z、弯曲模量 | **未公布**，表中为 `/` |

没有将拉伸强度或静态压缩强度当成杨氏模量。PhysX 仍需要输入弹性模量，因此当前
`shared/config.py` 使用 [Gent (1958)](https://doi.org/10.5254/1.3542351)
硬度换算，以标称 Shore A 75 得到 **7.083728 MPa**：

`E[MPa] = 0.0981*(56+7.66*S)/(0.137505*(254-2.54*S))`。

这是压痕经验估算，不是拓竹公布的拉伸模量，也没有按打印件标定。历史 85A 的 5.2 MPa
来自另一份 TDS 的 Z 向拉伸测试，与此换算值来源不同；不能据此推断 75A 比 85A 更硬。
泊松比 0.45、摩擦 0.7、弹性阻尼 0.0001 继续作为假设，并随每个 episode 保存。
TDS 的强度、伸长率是来源信息，当前共旋线弹性模型不会使用它们自动模拟断裂或滞回。

两根软指统一使用 Orange 021 U 的屏幕近似 `#FF6C2F`（sRGB 255/108/47），
参考 [Texas Tech RISO guide](https://www.depts.ttu.edu/art/documents/resources/current-students/facilities-technology-risostudio-printguide_ac26.pdf)。
USD displayColor 使用线性 RGB 转换值；MuJoCo 使用 sRGB。原 FEM 网格及独立细分显示网格都应用同色。

## 数据环境与存储

IsaacLab 固定 NumPy <2；LeRobot 0.4 的 Rerun 依赖 NumPy >=2，故隔离环境：

```bash
uv add 'lerobot>=0.4,<0.5' --project envs/dataset
bash scripts/sync_envs.sh dataset
```

已锁定 LeRobot 0.4.4、Torch 2.7.0、NumPy 2.2.6。NumPy <2.3 避免上游 0.4 写入器的单元素数组转换问题。
Isaac 环境保持 NumPy 1.26；增加 h5py、同一源码快照的 isaaclab-contrib/tasks，供官方任务使用。

`--record-episode` 默认生成真正的 v3：

```text
OUTPUT/recording/
  episode.json                 # 来源、标定、阶段、结果、全局 episode_index
  finger_reference.npz         # 每个 episode 一份 FEM 参考拓扑
  dataset/                    # 或 --dataset-root 指定的公共目录
    meta/info.json            # codebase_version = v3.0
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/observation.images.{left_rgb,right_rgb,external_rgb}/chunk-000/file-000.mp4
```

`observation.state` 为实际关节位置，`action` 为目标关节位置，单位为 rad / m（Panda 后两维为指关节米制位移）。
关节速度、物体位置与 wxyz 四元数、相机外参、深度、FEM 节点、底座位姿也存为 Parquet 数值列。
深度保留米制浮点值；无效射线用零值加 `finger_depth_valid` 掩码保存，避免上游统计器遇到 NaN/Inf，诊断读取时恢复 NaN。视频为官方写入器的 H.264，有损编码；不再额外存一份每帧 RGB NPZ。
LeRobot `timestamp` 从该 episode 的零时刻开始，实际仿真时刻另存 `observation.sim_time_s`。
`observation.phase` 的整数对应每个 episode.json 的 `phases` 表。

采集时暂存一个 `capture.partial.npy` 顺序流，完成后自动调用数据环境的官方
`LeRobotDataset.add_frame/save_episode/finalize`。转换成功删除暂存文件；异常 capture 不进入训练数据。
完整采集后的转换失败会保留 `staged` 索引及顺序流，可执行：

```bash
envs/dataset/.venv/bin/python benchmarks/finalize_lerobot.py OUTPUT/recording
```

`--dataset-root data/datasets/NAME` 将多次采集追加到同一数据集。要求 FPS、特征形状相同；
显式四面体网格有固定节点数，自动体素 cooking 可能因世界姿态产生不同节点数，必须分开数据集。
一个公共数据集同一时刻只运行一个写入进程。旧数据与失败实验保留，未混入新的 75A 数据集。
公共目录下 `meta/simulation/` 同时保存每个 episode 的标定、阶段映射、结果和 FEM 参考拓扑，数据集可独立搬运。

旧格式可以显式迁移，不修改源数据：

```bash
envs/dataset/.venv/bin/python benchmarks/convert_rgb_to_lerobot.py OLD/recording \
  --dataset-root data/datasets/legacy85a --index-root data/generated/legacy85a_conversion
```

验证：

```bash
envs/dataset/.venv/bin/python -m unittest discover -s tests -p test_lerobot_recording.py
envs/dataset/.venv/bin/python benchmarks/validate_rgb_episode.py OUTPUT/recording
envs/dataset/.venv/bin/python benchmarks/check_finger_camera.py OUTPUT
envs/dataset/.venv/bin/python benchmarks/check_finger_motion.py OUTPUT/recording
envs/dataset/.venv/bin/python benchmarks/check_camera_stability.py OUTPUT/recording
```

## 官方任务接入

**2026-11-14：** 任务实现已统一为原双 X5 + `measured_wedge_v3` 安装，唯一入口是
`benchmarks/robodojo_task_soft.py`，详见 [原生 X5 软指适配](robodojo-native-x5.md) 与
[倾斜安装验证](x5-tilted-installation-2026-11-14.md)。下文 Franka 命令与
`robodojo_soft_pick.py` 均已被移除，只作为历史记录。

过去的几何改进见
[规划说明](../benchmarks/SOFT_GRASP_PLANNING.md)：75 mm 掌部安装偏移、55 mm 接触高度、
88.18 mm 完整软指、姿态搜索、截面闭合开口、桌面/夹具碰撞、12 mm 试提和物体负载规划。
新材料和新任务都须重新验证实际抓取，不沿用旧 85A 成功结论。

`benchmarks/robodojo_task_soft.py` 加载上游 task 类、保存布局、SceneManager、RobotManager 和 RewardManager。
执行器实现 `general_pickup` 与 `stack_blocks` 的完整动作流程，成功条件直接使用原任务判定
（`general_pickup` 为物体抬升 >10 cm；堆叠使用原分段评分）。

命令与数据说明见 [根 README](../README.md) 和
[LeRobot v3 采集](robodojo-rgb-collection.md)。必须先准备该布局引用的官方资产。
场景导出包含实际目标和杂物几何；规划器支持非平移的机器人基座变换，
并将其他刚体、相机支架作为凸包障碍物。房间墙体未加入该规划碰撞集合，碰撞检查范围在 scene.json 明示。
软指底座随动，机械臂未接收 FEM 反作用力矩。计划成功不等于任务执行成功。
