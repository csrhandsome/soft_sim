# RoboDojo 新软夹爪三路 RGB 采集

当前带细分形变显示且通过保持画面稳定性检查的版本为 `robodojo_episode_029_refined_surface/recording/episode.mp4`，左右保持段跳动指标为 0.208 / 0.339 px，均低于原有 0.5 px 目标。它保留 `027` 的材料、52 mm 闭合参数、接触修正上限及渲染采样，将显示表面细分后绑定到实际 FEM 节点，以呈现原 STL 长三角面遗漏的横梁弯曲。主入口已默认启用 2 mm 显示细分，`--surface-max-edge-m 0` 可保留旧显示。说明、对照与复现见 [形变排查](fem-deformation-2026-11-12.md)。验证覆盖 10 秒静止挤压与恢复，不代表运动抓取或实物动力学已标定。

`027_contact_render_quality` 保留为修正显示前的稳定物理基线；原始指内 RGB 前后对比为该 episode 下的 `camera_stability_before_after.mp4`，左右保持段跳动指标为 0.226 / 0.258 px。完整历史结果与复现命令见本文末尾。

旧 `013_local_origin` 仍保留作历史对照：右指指标为 1.100 px，未通过。本文早期各轮结果按当时配置记录，不应与最新通过版本混淆。

在项目根目录运行。需要 NVIDIA GPU；使用现有 Isaac Sim 5.1 环境和已经下载的 RoboDojo 资产。

```bash
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python \
  benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/soft_grasp_plan_55/plan.json \
  --dt .001 --solver-iterations 80 \
  --record-episode --video-fps 20 \
  --output data/generated/robodojo_episode_002_render_sync
```

每次更换 `--output` 的 episode 编号，已有 `recording` 目录会拒绝覆盖。
`--video-fps` 同时控制仿真时间采样率和视频播放率，必须整除物理步频；1 ms 步长可以使用 10、20、25、50 FPS。
当前示例为一个接近、闭合、保持、12 mm 试提、抬升和保持的 episode；试提失败时提前终止，结果中的 `success` 会反映实际抓取结果。

输出位置为 `data/generated/robodojo_episode_002_render_sync/`：

| 路径 | 内容 |
| --- | --- |
| `recording/episode.mp4` | 外部、左夹爪、右夹爪三画面，1280×480 |
| `recording/left_rgb.mp4`、`right_rgb.mp4` | 左右夹爪独立视频，各 256×256 |
| `recording/external_rgb.mp4` | 外部固定相机视频，640×480 |
| `recording/frames/frame_*.npz` | 同步原始 RGB、夹爪深度、实际/指令关节位置、实际关节速度、物体位姿、夹爪相机外参、开口和仿真时间 |
| `recording/episode.json` | 帧索引、阶段、标定、采样率、结果及完成状态 |
| `trajectory.json`、`result.json` | 原有物理轨迹和抓取结果 |

每次采集先调用 `World.render()` 将当前 PhysX/Fabric 状态同步到渲染端，再由同一个零时间步 Replicator 采集动作触发三台相机，随后读取图像和状态，期间不推进物理步。单独调用零时间步 Replicator 不保证刷新物理场景的渲染状态。
NPZ 内图像为 `uint8` RGB，键名为 `left_rgb`、`right_rgb`、`external_rgb`；它们是原始训练图像。MP4 采用有损 `mp4v` 编码，用于预览。
相机坐标采用光学约定：X 向右、Y 向下、Z 向前；`finger_camera_to_world` 的顺序固定为左、右。
`commanded_joint_positions` 是该步控制指令，`joint_positions` 和 `joint_velocities` 是采集时实际状态。
异常中断时 metadata 保持 `incomplete`，视频保留 `.partial.mp4` 后缀，训练时跳过该目录。

采集结束后先验证夹爪相机的实际投影和运动时同步，再检查数据完整性：

```bash
envs/robodojo/.venv/bin/python benchmarks/check_finger_camera.py \
  data/generated/robodojo_episode_002_render_sync

envs/robodojo/.venv/bin/python benchmarks/validate_rgb_episode.py \
  data/generated/robodojo_episode_002_render_sync/recording

# 独立检查抓取及最后一秒持续保持；仅适用于成功的 planned grasp
envs/robodojo/.venv/bin/python benchmarks/validate_grasp_run.py \
  data/generated/robodojo_episode_002_render_sync
```

这是 RoboDojo 机器人/物体资产和自定义 PhysX FEM 软夹爪的采集入口，尚未接入官方 task/observation/reward manager。
物体始终为自由刚体，软指底座由机器人轨迹驱动；没有向机械臂反馈 FEM 反作用力矩，也没有原生软体接触力标签。
更换物体时需重新导出场景并规划，流程见 [软夹爪规划](../benchmarks/SOFT_GRASP_PLANNING.md)。

## 本次生成结果（2026-11-11）

旧的 `robodojo_episode_000` 虽然视频解码、数据完整性和抓取结果通过，但实际夹爪视野错误，不能作为可用的夹爪形变图像数据。新增的 `camera_projection_validation.json` 明确记录该检查失败。

问题为 Isaac 5.1 RTX 对过小光学参数的实际投影处理：原来按米制 USD 换算得到 `focalLength=.028`、`horizontalAperture=.048`，USD 理论视场为 81.2°，实际图像约 97% 却被指端同一平面占满。
GPU 对比中，焦距/光圈等比例取 `2.8/4.8` 和 `28/48` 均能得到一致的正确图像；仅比较属性比值或整幅 RGB 标准差无法发现这个问题，后者还会把颜色通道差异算成空间纹理。

现使用 `backends/isaac_sim/cameras.py` 将针孔投影等比例归一化到安全数值，显式关闭景深。物理镜头参数和训练内参仍按原毫米配置保存；相机安装位置与朝向保持原设计：距底座 6 mm，沿软指局部 +Z 看内部横梁。
该修正用于 RoboDojo 5.1 主入口，同时替换了旧 Isaac 后端相同的投影设置；旧 6.2 运行环境本轮未实跑验证。

`check_finger_camera.py` 使用初始稳定段的实际深度与源 STL 独立射线相交结果比较，并检查图像中央区域的空间变化。后续新增的运动检查覆盖未接触物体的 `pregrasp` 阶段，只比较距镜头 40 mm 内、靠近底座的横梁，允许 2 mm 弹性位移；不把接触后的真实形变与未形变 STL 强行对齐。该检查适用于当前材料、安装及轨迹，更换这些条件后需重新评估容差。
旧视频两台相机均只有约 0.4% 的采样点符合 2 mm 深度容差，中位深度误差为 55.4 mm，能被该检查稳定拒绝。

仅修正投影的 `robodojo_episode_001_camera_fixed/recording/episode.mp4` 已完成：15 秒、20 FPS、300 帧，四路 MP4 全部逐帧解码通过。此版本后来发现运动时的渲染同步错误，应使用下述 `002` 版本。
两台相机初始各 731 个 STL 命中采样点全部满足 2 mm 深度容差，画面能看到内部横梁与闭合后的形变。
本轮抓取成功，最大提起 82.01 mm，最后 1 秒保持段最低提起 77.48 mm，最终 XY 漂移 15.89 mm。
共享层 8 项回归测试通过；相机投影验证与数据完整性验证分别保存于 `camera_projection_validation.json` 和 `validation.json`。
修正前后对比见 `data/generated/camera_before_after.png`，形变前后对比见 `data/generated/camera_fixed_deformation.png`。

## 指内画面隔帧抖动的原因

`001` 在 1 ms 物理步长下每 50 步采集，却只每 100 步调用一次 `World.render()`。相机外参每步写入 USD，但物理场景的 Fabric 渲染数据未在每次采集前刷新，造成当前相机拍摄上一渲染时刻的软指；运动时画面每隔一帧恢复正确，因此表现为剧烈来回跳动。静止保持阶段的相机位姿没有变化，不能把这一图像问题归因于机械臂振动。

`MountedSoftTool.capture(world)` 现在在每次采集前调用 `World.render()`，随后才执行零时间步 Replicator。该调用负责刷新渲染数据，不额外推进物理。运动检查能拒绝旧 `001`：121 个未接触运动帧中，左右相机最差只有 15.2% / 10.4% 的近底座射线满足 2 mm 深度容差；最大逐帧深度误差中位数分别为 3.72 / 5.45 mm。原先只检查静止首帧，因此漏掉了这个问题。

连续帧对比见 `data/generated/camera_render_sync_before_after.png`，每两行分别是左/右相机的旧版和修复版。API 中物理步、Fabric 更新与渲染接口的区别见 [Isaac Sim 5.1 SimulationContext](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/py/source/extensions/isaacsim.core.api/docs/index.html#simulation-context)。

材料确实创建并绑定到两根 PhysX FEM 软指：杨氏模量 5.2 MPa、泊松比 0.45、密度 1180 kg/m³、摩擦 0.7、弹性阻尼 0.0001。只固定底部 4 mm 的节点；本次 cooked 网格左右各有 6924 / 6721 个节点，其中 681 / 608 个固定。材料仍是 TPU 的各向同性初值，未按实物打印件标定。

旧 `001` 的 `trajectory.json` 记录的是去除底座刚性运动后的最大节点位移：闭合时约 4.07–4.95 mm，保持时约 4.94–4.97 mm，抬升阶段最大约 5.55 mm。它包含自重、惯性和接触造成的弹性位移，不等于纯接触压缩量，也不能用相邻图像的视角跳动代替形变测量。

## 渲染同步修复后的实跑结果

`robodojo_episode_002_render_sync` 使用相同的材料、规划轨迹、1 ms 物理步长和 80 次求解迭代，重新完成 15 秒、20 FPS、300 帧的抓取。首帧投影、121 帧未接触运动检查、三路原始数据完整性和四路 MP4 逐帧解码均通过；8 项共享层测试和 2 项规划安装坐标测试通过。

未接触运动阶段，左右相机的最大逐帧深度误差中位数分别降至 0.128 / 0.182 mm；每帧满足 2 mm 容差的射线比例最低为 97.4% / 91.8%。大幅隔帧跳动消失。抓取成功，最大提起 82.26 mm，最后 1 秒最低提起 77.20 mm，最终 XY 漂移 15.75 mm。

新三画面视频在 `robodojo_episode_002_render_sync/recording/episode.mp4`；相同时间戳下旧/新左右画面对比在该目录的 `render_sync_comparison.mp4`。修复后接触前与保持阶段的静态比较在 `data/generated/camera_render_sync_deformation.png`。材料和控制指令没有为增加视觉形变而调整。

## 刚性方块与全程节点检查

`003` 使用 40×40×70 mm、150 g 的解析刚性方块，重新导出几何并规划抓取，候选是 0° 倾角的上方抓取。闭合参数为 37.17 mm，底座间距 53.17 mm。物体为自由刚体；相机固定在各自软指底座上。原羽毛球外观模型在此前场景中也按刚体处理，换成方块是为了使平整接触面和软指弯曲更容易观察，未通过改变物体柔性制造差异。

```bash
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --mode export --primitive-box --box-size .04 .04 .07 --object-mass .15 \
  --output data/generated/rigid_box_scene

envs/planning/.venv/bin/python benchmarks/plan_soft_grasp.py \
  --scene data/generated/rigid_box_scene --contact-height .055 --indent .003 \
  --output data/generated/rigid_box_plan

OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/rigid_box_plan/plan.json --dt .0005 --solver-iterations 120 \
  --record-episode --video-fps 25 --closeup \
  --output data/generated/robodojo_episode_003_rigid_box

envs/robodojo/.venv/bin/python benchmarks/check_finger_camera.py \
  data/generated/robodojo_episode_003_rigid_box
envs/robodojo/.venv/bin/python benchmarks/check_finger_motion.py \
  data/generated/robodojo_episode_003_rigid_box/recording
envs/robodojo/.venv/bin/python benchmarks/validate_rgb_episode.py \
  data/generated/robodojo_episode_003_rigid_box/recording
```

本次使用 0.5 ms 步长、120 次求解迭代。软指杨氏模量、泊松比和阻尼保持原值。渲染改用 FXAA，关闭运动模糊，每次采集渲染两个同一物理时刻的子帧，避免近距离图像依赖旧 DLSS 历史。这是对图像伪影的处理，不代表软指物理振动已经消失。官方对运动伪影和子帧的说明见 [Isaac Sim 5.1 Replicator Troubleshooting](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/troubleshooting.html)。

每帧新增 `finger_nodes_left`、`finger_nodes_right`（真实 cooked FEM 节点世界坐标）和 `finger_mounts`（底座外参）。`recording/finger_reference.npz` 保存对应节点顺序的底座局部参考坐标与固定节点掩码。不能用共享体素网格替代该参考，因为 Isaac 5.1 的 cooking 会改变节点数量和顺序。

`check_finger_motion.py` 检查整段相机相对底座的安装变换，报告各阶段去除底座刚性运动后的节点位移、相对接触前的新增变形，以及保持阶段的逐帧节点变化。相机安装检查通过与真实软指保持静止是两个独立结论；报告保留物理运动，不用图像防抖掩盖软指弯曲。

### 静止机械臂的闭合、保持和松开

`003` 的原始 37.17 mm 闭合量使方块上沿把锥形软指压弯很深，原始阻尼 0.0001 下，保持阶段仍有明显节点振动。截取 8.12–8.84 s，左右软指 40 ms 内的最大局部节点变化约 4.06 / 3.59 mm。这是 FEM 的真实节点运动，不能通过只检查相机底座或初始投影将其排除。

这轮完整抓取没有通过提起保持检查：最大抬升仅 12.54 mm，后续方块滑脱。相机初始投影和未接触运动检查仍通过，说明这些成像检查不能代替接触稳定性或抓取成功验证。原始失败数据保留在 `003`，不作为成功抓取样本。

`--mode squeeze` 在已检查的张开末端接近姿态初始化，机械臂全程固定，依次执行 1 s 稳定、3 s 闭合、2 s 保持、3 s 松开和 1 s 恢复。`004` 将闭合参数增至 52 mm（底座间距 68 mm），减小对方块上沿的过度压入；弹性阻尼显式设为 0.005，杨氏模量、泊松比和密度保持不变。该阻尼是用于检查振动衰减的试验值，尚未按实物标定。

```bash
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/rigid_box_plan/plan.json --mode squeeze \
  --squeeze-width .052 --elasticity-damping .005 \
  --dt .001 --solver-iterations 80 --record-episode --video-fps 25 --closeup \
  --output data/generated/robodojo_episode_004_rigid_squeeze
```

这是接触与恢复实验，`task=stationary_squeeze_release`，`success=null`，不把它冒充提起成功。使用相同的相机、节点和数据完整性检查；没有 `pregrasp` 运动阶段时，初始投影检查的 `motion_sync_valid` 为 null，全程安装与节点运动由 `check_finger_motion.py` 单独测量。它使用稳定段最后一帧作为接触前参考。

### 固定底座网格层的修正

逐帧节点检查发现，`004` 的左指在没有接触方块时就发生了较大位移；增大阻尼的 `005` 也没有消除这个异常。进一步检查排除了孤立体网格：左右 cooked 网格各只有一个连通分量，所有节点均与固定底座连通。

实际错误在固定层的选择：PhysX 自动体网格的名义 4 mm 层位于约 4.008 mm，原来 `1e-7 m` 的容差将整层排除，只固定了更低的两层节点。不同安装姿态下，这一层是否落在阈值内也可能不同。现按体网格步距的 1%（当前为 20 μm）允许 cooking 量化偏差，固定区域实际延伸至约 4.008 mm，而非把毫米级的自由区域额外固定。静止上方抓取姿态下，左右固定节点数从 609 / 644 增至 968 / 1029。

保持原始材料和阻尼的一秒初始化检查 `rigid_pin_initialization`，末帧左右最大局部位移约 0.706 / 0.041 mm。增加了量化网格层的回归测试，防止重新漏掉整个支撑层。网格拓扑、固定节点数及实际固定高度保存于 `mesh_topology.json`；实际四面体索引保存在 `finger_reference.npz`。

采集函数还对比渲染前后的 PhysX 节点，若零时间采集使节点变化超过 0.1 μm，则拒绝采集。这直接验证渲染没有额外推进物理，不再只依赖 API 参数或文件时间戳。

`006` 使用自动体网格和修正后的固定层，机械臂保持静止，闭合到 52 mm，再保持、松开。弹性阻尼 0.005，未指定人工节点速度阻尼；杨氏模量、密度和泊松比保持原值。

```bash
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/rigid_box_plan/plan.json --mode squeeze \
  --squeeze-width .052 --elasticity-damping .005 \
  --dt .001 --solver-iterations 160 --record-episode --video-fps 25 --closeup \
  --output data/generated/robodojo_episode_006_rigid_base_fixed
```

`006` 的增大材料阻尼试验在当前 1 ms 步长和 160 次迭代下仍出现初始化异常，因此已中止，保留为 incomplete，不作为最终演示。

最终 `007` 回到原始材料参数：杨氏模量 5.2 MPa、泊松比 0.45、密度 1180 kg/m³、弹性阻尼 0.0001；使用 1 ms 步长和 80 次迭代，未覆盖 PhysX 默认节点速度阻尼。闭合参数仍为 52 mm，机械臂全程固定。

```bash
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/rigid_box_plan/plan.json --mode squeeze --squeeze-width .052 \
  --dt .001 --solver-iterations 80 --record-episode --video-fps 25 --closeup \
  --output data/generated/robodojo_episode_007_rigid_original_material
```

全程节点检查现在也检查未接触稳定段；针对当前 TPU 构形使用 2 mm 最大未加载位移阈值，更换材料或几何时需重新评估（`--maximum-unloaded-mm`）。这可以拒绝 `004`、`005` 的异常初始化，即使它们的相机安装和首帧投影检查通过。

`007` 已完成 10 秒、25 FPS、250 帧。接触前到保持阶段的最大局部形状变化约为左 7.25 mm、右 5.16 mm；松开并恢复后，最大残余变化约为左 0.123 mm、右 0.002 mm。保持阶段的逐帧节点变化 P95 为左 0.017 mm、右 0.163 mm，最极端局部节点仍可达到约 0.31 / 1.16 mm，不能宣称完全无物理振动。全程相机相对底座的安装变换保持固定，零时间采集的节点不变检查全部通过。

旧版 `007/recording/squeeze_closeup.mp4` 含相对未加载状态的真实节点位移标注；`deformation_before_loaded_released.png` 为接触前、保持、松开后的对比。原始三路 RGB、深度和节点在 `recording/frames`。本轮为静止机械臂的压缩与恢复演示，不作为稳定提起成功样本。相关 8 项共享测试、3 项节点/固定层测试及 2 项规划坐标测试通过。

### 用像素检查剩余抖动

指内镜头距离横梁很近，亚毫米节点运动仍会变成明显的画面跳动。`check_finger_motion.py` 现在额外计算每帧视锥内 FEM 节点的投影位移：保持开始后排除 0.25 s，逐帧求 P95，再取最差帧；默认上限为 0.5 px（`--maximum-hold-p95-px`）。节点包含被遮挡的部分，该指标不是实际 RGB 的检验；缺少标定或有效帧对时结果为 null。

`check_camera_stability.py` 独立读取原始 RGB，以双向一致性和亮度残差筛选稀疏光流，检查左右指内和外部画面的保持段。它不需要节点或相机位姿，默认同样要求最差帧 P95 ≤ 0.5 px；纹理不足不能判为通过。两个像素阈值均用于当前分辨率和 25 FPS 的演示验收，不是相机硬件精度或物理真实性的标准，也不能发现采样间隔内的高频振动。正常闭合和松开不纳入静止保持阈值。

```bash
envs/robodojo/.venv/bin/python benchmarks/check_camera_stability.py \
  data/generated/robodojo_episode_007_rigid_original_material/recording
```

重新检查 `007`：节点投影最差帧 P95 左右为 0.248 / 3.954 px；实际 RGB 光流为 0.470 / 4.318 px，外部画面为 0.361 px。相机安装依然通过，但右指保持画面的稳定性失败；新版全程节点检查的总 `valid` 也因此为 false。之前的毫米单位统计将全部帧与节点合并计算 P95，会淡化短时、近镜头的局部振动。

`008_fine_step` 保持原始材料，将步长减到 0.25 ms、保持 80 次求解迭代。保持阶段实际 RGB 最差帧 P95 达到左 10.879 / 右 1.669 px，因此中止并保留为 incomplete，诊断见该目录 `trial_notes.json`。该配置不是改进结果。

后续试验使用 `--vertex-velocity-damping` 显式设置节点速度阻尼，保留原有弹性材料参数。PhysX 将该系数定义为单位 1/s 的人工速度阻尼，见 [legacy deformable schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/physxschema.html)。它改变真实求解节点的动态响应，不是 RGB 后处理，也不代表已经标定 TPU 的粘弹性；需要动力学或估力数据时应另外标定，不能直接沿用演示阻尼作为材料真值。

`009_velocity_damped` 使用 50/s 节点阻尼，保持段 RGB 最差帧 P95 为左 0.922 / 右 1.426 px；`010_stable_squeeze` 使用 200/s，结果为左 0.683 / 右 2.706 px。两次都已完成完整闭合、保持、松开，但没有达到稳定性验收线。`010` 目录名称是运行前的试验标签，不代表通过；实际结论以 `camera_stability_validation.json` 和 `finger_motion_validation.json` 为准。这两次结果也说明不能假定增加速度阻尼必然单调改善接触稳定性。

后续采集的 `episode.json` 中，`calibration.deformable_dynamics` 会额外保存材料、节点阻尼覆盖值、求解次数和阻尼未标定标记；`null` 覆盖值表示沿用 PhysX 默认，不能解释为阻尼为零。既有 episode 的完整参数仍见其 `result.json`。

`011_quasistatic_squeeze` 将节点阻尼增至 1000/s，RGB 最差帧 P95 仍为左 0.671 / 右 1.192 px，未通过；最终配置不根据这个强阻尼试验选定。

### 夹持区附近的仿真原点

新增 `--local-physics-origin`（目前仅用于 squeeze）将世界坐标平移到刚性方块附近。原始代码将软指顶点直接烘焙到全局单精度坐标中，在细网格和高频求解下存在数值精度问题。平移保留机器人、物体、桌面、相机之间的相对位置；软指独立写入平移后的世界坐标，FK 的机器人底座也使用同一平移。地平面接口接收世界高度，必须显式同步，否则会在平移后的桌面上方形成额外碰撞面；这个错误试验保留在 incomplete 的 `012_local_origin`。

`calibration.world_origin_offset_m` 与 `result.json` 保存平移量；原场景坐标 = 记录的世界坐标 − 平移量。当前方块对应 `[-0.42, 0.12, -0.155] m`，不改变坐标轴方向或米制单位。重新 cooking 后两侧仿真网格为 6608 / 6665 个节点，固定节点为 960 / 1006；参考与四面体拓扑随新 episode 独立保存。

`013_local_origin` 使用原材料、原默认节点阻尼、1 ms 步长和 80 次迭代。未接触稳定段末帧，两指最大位移均约 0.029 mm，原先左指约 0.7 mm 的偏移消失；保持阶段逐帧节点变化最大值降至左 0.040 / 右 0.181 mm。此结果支持改善数值条件，但不能仅凭一次重新 cooking 就将全部变化归因于浮点精度；还需要独立 RGB 检查。

后续对照均未优于 `013`：`014_local_origin_damped` 增加 50/s 节点阻尼，左右 RGB 跳动指标为 0.273 / 3.023 px；`015_reduced_indent` 将闭合宽度调到 56 mm，反而触发更大摆动；`016_converged` 将迭代增至 255，仍有明显抖动；`017_strain_damping` 将材料弹性阻尼从 0.0001 增至 0.0005，结果也更差。`015`、`016`、`017` 均中止并保留为 incomplete，诊断见各自 `trial_notes.json`。试验目录名称不代表通过验证，失败配置不作为最终采集参数。

### 保留结果与复现

`013` 的 250 帧场景平移检查通过：扣除已记录的平移量后，相机与底座位置误差小于 1e-15 m，旋转与关节指令一致；材料、默认节点阻尼、镜头内参均未改动。初始 STL 投影、数据完整性、四路原始 MP4 全帧解码通过；两个派生预览也均解码为 250 帧、25 FPS。

| 保持段指标 | `007` 左 / 右 | `013` 左 / 右 |
| --- | --- | --- |
| 最差帧 RGB 特征位移 P95（px） | 0.470 / 4.318 | 0.470 / 1.100 |
| 最差帧节点投影位移 P95（px） | 0.248 / 3.954 | 0.173 / 0.872 |
| 最大逐帧节点变化（mm） | 0.309 / 1.162 | 0.040 / 0.181 |
| 相对接触前最大形变（mm） | 7.254 / 5.160 | 7.400 / 5.955 |

`013` 外部画面 RGB 指标为 0.133 px；右指 RGB 指标降低约 74.5%，最大逐帧节点变化降低约 84.4%。恢复段相对接触前的最大残余变化均小于 0.001 mm。所有预览均直接使用原始 RGB，没有做图像防抖。

右指保持画面仍未达到 0.5 px 目标。因此 `validation.json` 中 `data_integrity_valid=true`，但 `hold_projected_motion_valid=false`、`camera_rgb_hold_stability_valid=false`，总 `valid=false`；不能把这个改进版标为稳定性全通过。18 项相关单元测试通过，包括静止、已知像素平移、近镜头亚毫米振动以及安装通过但 RGB 抖动仍须拒绝的回归场景。

```bash
OMNI_KIT_ACCEPT_EULA=YES envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/rigid_box_plan/plan.json --mode squeeze --squeeze-width .052 \
  --local-physics-origin --dt .001 --solver-iterations 80 \
  --record-episode --video-fps 25 --closeup \
  --output data/generated/robodojo_episode_NEXT_local_origin

# 以下检查顺序会让总体验证读取像素稳定性结论。
envs/robodojo/.venv/bin/python benchmarks/check_finger_camera.py \
  data/generated/robodojo_episode_013_local_origin
envs/robodojo/.venv/bin/python benchmarks/check_finger_motion.py \
  data/generated/robodojo_episode_013_local_origin/recording
envs/robodojo/.venv/bin/python benchmarks/check_camera_stability.py \
  data/generated/robodojo_episode_013_local_origin/recording
envs/robodojo/.venv/bin/python benchmarks/validate_rgb_episode.py \
  data/generated/robodojo_episode_013_local_origin/recording
```

保持画面未达标时，后面三条检查返回非零退出码，这是验收结论，不是视频损坏。对比视频、三阶段截图、像素指标图和场景平移审计分别为 `camera_jitter_before_after.mp4`、`comparison_*.png`、`camera_jitter_metrics.png` 和 `scene_translation_validation.json`。

## 2026-11-12：冻结渲染与接触求解对照

`--frozen-render-check` 在保持结束时连续采集 12 次，不执行物理步；保存原始 RGB，以及时间、节点、相机、底座、关节和物体位姿。检查要求这些状态完全不变，再按原有 0.5 px 标准检查图像。`018_frozen_baseline` 的全部状态差为零，冻结图像指标为左 0.182 / 右 0.222 px；正常保持段为 0.854 / 1.861 px。结果支持优先处理接触求解振动，并不表示渲染噪声为零。

同一配置的重复运行会有数值差异：旧 `013` 的右指为 1.100 px，这次 `018` 为 1.861 px，不能把两次的差异误称为参数修复效果。后续对比视频使用本次完整基线 `018`，全部保留原始 RGB。

新增接触参数 `--max-depenetration-velocity`，同时作用于软指和自由刚体。它限制接触求解为消除相交而引入的修正速度，单位 m/s；软指原生默认值为无限大，定义见 [PhysX schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/physxschema.html)。这个参数改变接触瞬态响应，不能当作已经标定的材料参数，也不等于限制所有节点的实际速度。未传参仍保留原默认值。

保持阶段不再重复提交完全相同的 FEM 底座目标。比较采用实际发送给 PhysX 的 float32 坐标，没有额外运动死区；初始化会清空缓存，闭合和松开仍更新目标。材料、固定层、镜头和原始 RGB 没有为此调整。

| 对照 | 结果与采用情况 |
| --- | --- |
| `019_native_settling` | 恢复原生休眠/低速收敛阈值后，RGB 为 0.806 / 4.121 px，未通过，不作为默认参数。 |
| `020_local_fine_probe` | CPU 线程争抢导致耗时异常，早期中止；相同物理设置由 `021` 重跑。 |
| `021_local_fine_probe` | 原点平移配合 0.25 ms 步长，保持段节点投影仍为约 2.20 / 0.155 px；中止，保留 incomplete。 |
| `022_contact_limit_probe` | 0.01 m/s 上限让形变降到约 0.1 mm，虽然节点画面稳定，也因接触响应异常而排除。 |
| `023_contact_limit_1_probe` | 1 m/s 上限的节点投影约 0.020 / 0.036 px，仍有约 4 mm 形变；仅作为节点筛选，完整 RGB 见 `024`。 |
| `024_contact_limit_rgb` | 完整 10 秒、250 帧；RGB 为 0.354 / 0.519 px，右指仍略超 0.5 px。节点、安装、恢复和碰撞顶点检查通过，但 RGB 与总体验证不通过。 |
| `025_contact_limit_10_probe` | 10 m/s 上限仍产生约 4.74 / 2.64 px 节点投影跳动，保持早期中止，保留 incomplete。 |
| `026_contact_limit_half_rgb` | 0.5 m/s 上限完成完整采集，RGB 为 0.891 / 0.386 px；右指改善但左指退化，未采用。 |

`024` 保持段相对接触前的最大节点形变为左 4.204 / 右 4.052 mm；最大逐帧节点变化为 0.0092 / 0.0137 mm。恢复后与接触前的最大差异小于 0.001 mm。冻结渲染检查仍通过，左右约 0.171 / 0.189 px。

`--record-collision-nodes` 额外保存实际碰撞网格顶点，区别于 FEM 仿真节点。`check_box_contact.py` 在每帧记录的刚体位姿下独立计算解析方块的有符号距离，检查整个闭合、保持和松开过程；`024` 的最大顶点穿透为零。顶点检查只能给出网格相交的下界，不能排除仅发生在三角形或边上的相交，不作为完整无穿透证明。

`024/camera_stability_before_after.mp4` 是 `018` 与 `024` 的同步原始指内 RGB 对照，250 帧、25 FPS，逐帧解码通过；没有做时间平滑或图像防抖。`comparison_settle.png`、`comparison_hold.png`、`comparison_settle_release.png` 为同一时间的对照截图。

进一步检查 `024` 的最差 RGB 帧对：用记录的深度筛选可见碰撞顶点后，其投影位移 P95 为左约 0.016 / 右约 0.024 px，而对应 RGB 为 0.354 / 0.519 px；物体的平移变化不足 0.00001 mm。这提示微小剩余跳动还包含渲染采样波动。新增 `--rt-subframes` 与 `--lighting-samples`，分别控制同一物理状态下的渲染子帧和 RTX 直接光/反射每像素采样数，实际生效值随 `calibration.rendering` 保存。子帧仍使用零物理时间，逐次检查渲染前后的节点不变。

### 修正显示前的稳定基线：`027_contact_render_quality`

使用 1 m/s 接触修正上限、4 个零时间渲染子帧、RTX 直接光和反射各 8 samples/pixel。1 ms 物理步长、80 次求解迭代、材料、固定层、相机内参和控制轨迹保留原值，未启用额外节点速度阻尼。渲染采样提高会增加计算量，不是改变视频帧率或做时间平滑。

| 保持段指标 | 同日基线 `018` 左 / 右 | `027` 左 / 右 |
| --- | --- | --- |
| 原始 RGB 最差帧位移 P95（px） | 0.854 / 1.861 | 0.226 / 0.258 |
| 节点投影最差帧位移 P95（px） | 0.562 / 1.487 | 0.018 / 0.036 |
| 最大逐帧节点变化（mm） | 0.215 / 0.747 | 0.0076 / 0.0137 |
| 相对接触前最大形变（mm） | 7.211 / 5.703 | 4.204 / 4.045 |

`027` 完成 10 秒、250 帧、25 FPS。相机投影、固定安装、初始化、节点投影运动、原始 RGB 和数据完整性均通过，`recording/validation.json` 的 `valid=true`。四路原始 MP4 全部逐帧解码通过。额外的整段碰撞顶点检查通过，最大顶点穿透为零；该检查的边/面相交限制仍如上所述。恢复后与接触前的最大节点差异小于 0.001 mm。冻结检查中物理状态差全部为零，左右 RGB 指标约为 0.140 / 0.118 px。

针对夹爪间距和四元数的排查保存在 `pose_and_clearance_audit.json`。整段实际左右碰撞网格沿闭合方向至少间隔约 27.9995 mm，因此两指没有互相碰撞；这不排除它们与方块的夹持接触。保持段记录的关节角、底座矩阵和相机矩阵均完全不变；当前 squeeze 重放固定关节角，不逐帧做四元数逆解或姿态插值。物体本身有接触产生的微小运动，相邻保持帧最大转角约 0.0052°，四元数没有符号翻转。检查依据为记录的 25 FPS 状态，不是对所有物理子步的测量。

`027/camera_stability_before_after.mp4` 与本次基线 `018` 对齐到相同时间戳；`comparison_*.png` 显示接触前、保持和恢复状态。全部由原始 RGB 生成，未做图像防抖。24 项相关单元测试通过，覆盖静止/平移光流、冻结渲染与物理时间混淆、近镜头节点振动及带旋转的解析方块距离。

下面的参数是显式启用的演示配置，没有把接触上限或高采样强制设为所有抓取任务的默认值。接触上限改变了瞬态求解响应，保持形变仍约 4 mm，但不能据此宣称与原来的 5–7 mm 结果定量等价；运动抓取和估力数据需要另行验证。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/rigid_box_plan/plan.json --mode squeeze --squeeze-width .052 \
  --local-physics-origin --dt .001 --solver-iterations 80 \
  --max-depenetration-velocity 1 --rt-subframes 4 --lighting-samples 8 \
  --record-episode --video-fps 25 --closeup --frozen-render-check --record-collision-nodes \
  --output data/generated/robodojo_episode_NEXT_contact_render_quality

envs/robodojo/.venv/bin/python benchmarks/check_finger_camera.py \
  data/generated/robodojo_episode_027_contact_render_quality
envs/robodojo/.venv/bin/python benchmarks/check_finger_motion.py \
  data/generated/robodojo_episode_027_contact_render_quality/recording
envs/robodojo/.venv/bin/python benchmarks/check_camera_stability.py \
  data/generated/robodojo_episode_027_contact_render_quality/recording
envs/robodojo/.venv/bin/python benchmarks/check_box_contact.py \
  data/generated/robodojo_episode_027_contact_render_quality/recording
envs/robodojo/.venv/bin/python benchmarks/validate_rgb_episode.py \
  data/generated/robodojo_episode_027_contact_render_quality/recording
```
