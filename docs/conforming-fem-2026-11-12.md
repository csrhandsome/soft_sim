# 参考源码、固定区和贴合体网格（2026-11-12）

## 参考实现

公开代码已克隆到 `third_party/`，完整上游和提交号见 [清单](../third_party/sources.json)，具体阅读位置见 [参考说明](../third_party/README.md)。本轮没有编译 SOFA 或 DiffPD；借鉴其边界条件、网格和验证方法，在现有 Isaac Sim 5.1 流程中实现对照。

SOFA 的 `SoftFingerDesign/details/actuated_finger.py` 会将实际夹持区域的节点 `Rigidify` 后映射到安装件，独立处理内部接触表面。STLIB 的 `FixedBox` 则通过 ROI 和高刚度 `RestShapeSpringsForceField` 约束节点。ETH 的 `beam_env.py` 对固定端节点三个方向设置 Dirichlet 条件，并通过 `tet_mesh.py` 使用 TetGen 贴合体网格。它们的材料、形状、单位与当前 TPU 软指不同，不能直接复制数值。

## 旧结果的独立诊断

对 `robodojo_episode_029_refined_surface` 的完整节点记录：

- 固定节点最大偏移左/右约 `0.000302 / 0.000295 mm`，没有看到固定节点明显滑动。
- 保持结束时，固定区以上 4.1–6.1 mm 高度带的最大接触位移约 `0.144 mm`；6.1–10 mm 高度带约 `0.83–0.85 mm`。固定区上方的根部仍然是弹性材料，不能将其弯曲直接当作约束失效。
- 真正的横向梁（Z 恒定、位于中间高度）的最大曲线到端点连线偏差仅 `0.075 / 0.071 mm`。此前显示诊断约 3 mm 的最大值主要来自纵向侧壁长边，旧文字称其为横梁不准确。本轮诊断分开测量两者。
- 左/右自动求解四面体体积与 STL 的比值为 `1.84669 / 1.85933`。这不是显示网格的问题。左侧 19725 个求解单元中，8799 个单元中心落在 STL 实体外；中心采样不是完整几何误差积分。

新增 `benchmarks/check_clamp_deformation.py`：在随动底座坐标系中检查固定误差、最佳刚体拟合残差、四面体主伸长和变形 Jacobian，并独立测量真正横向梁的曲率。绘图保持位移原比例，没有视觉放大。

`check_finger_motion.py` 现在将固定节点误差纳入总验收，默认上限 `0.01 mm`。此前该值虽被输出，却未影响 `valid`。新增夹持滑移夹具测试确保相机稳定不能掩盖固定区滑动。

## 贴合网格与运行方式

TetGen 2 mm 参数生成 `12721` 节点、`50659` 四面体，总体积相对 STL 误差 `+0.000751%`，50659 个单元中心均位于 STL 内。2 mm 控制最大单元体积参数 `pitch³/6`，不是保证所有边长小于 2 mm；原 STL 小曲面仍会产生更小单元。

独立网格环境不会修改现有 RoboDojo 环境，安装配置与依赖锁位于 `envs/meshing/`：

```bash
uv sync --project envs/meshing --locked
OPENBLAS_NUM_THREADS=1 envs/meshing/.venv/bin/python scripts/prepare_mesh.py \
  --method tetgen --pitch .002 --output data/generated/soft_finger_tetgen_2mm.npz
```

先前隔离临时环境与锁定环境生成的节点坐标、单元索引完全一致。原有 Gmsh 路径对这份 STL 的部分曲面参数化失败，未采用该结果。

主入口新增：

- `--fem-mesh PATH`：对力学和碰撞都显式使用同一四面体网格；不再让 PhysX 自动体素化这份 STL。显示网格仍绑定到实际求解节点。
- `--pin-height-m`：明确指定固定高度，默认继续为 4 mm；改变固定区域属于边界条件实验，不自动当作真实安装方式。
- `--self-collision`：开启内部自接触，便于在较大压缩下检查内部梁壁接触。默认不变。

另外修正显式网格的初始化：`World.reset()` 会先推进未固定软体，节点整体出现约 0.0295 mm 的重力下落。如果将此状态作为参考位置，会把下落写进固定目标，并令原 STL 表面点相对参考体网格产生外插。现在先校验求解单元顺序，恢复作者指定的原始节点位置并清零速度，再安装固定目标。求解器若重排单元，会明确报错，避免误写节点。

表面绑定也补充了搜索：96 个最近单元中心不一定包含真实容纳点的细长单元，外插前增加包围半径和 AABB 检查。新增“容纳单元位于 120 个更近中心之后”的测试。旧 `031` 录制使用原初始化；`033` 使用修正后的初始化和绑定。

同参数 52 mm 闭合对照：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/rigid_box_plan/plan.json --mode squeeze --squeeze-width .052 \
  --local-physics-origin --dt .001 --solver-iterations 80 \
  --fem-mesh data/generated/soft_finger_tetgen_2mm.npz --surface-max-edge-m .002 \
  --max-depenetration-velocity 1 --rt-subframes 4 --lighting-samples 8 \
  --record-episode --video-fps 25 --closeup --record-collision-nodes \
  --output data/generated/robodojo_episode_NEXT_conforming

OPENBLAS_NUM_THREADS=1 envs/robodojo/.venv/bin/python benchmarks/check_clamp_deformation.py \
  data/generated/robodojo_episode_NEXT_conforming \
  --output data/generated/clamp_audit_NEXT --plot --profile-video
```

每阶段还保存 `*_fem.npz` 力学检查点，支持无相机的边界条件敏感性实验。

## 已完成的固定区域对照

`031` 使用贴合网格、4 mm 固定区、52 mm 闭合指令，完整录制 250 帧。`032_clamp6_sensitivity` 仅将固定区扩到 6 mm，关闭相机以加速实验；同样完成全部挤压/保持/释放阶段。

保持结束时，相对接触前的最大节点位移：4 mm 固定为左/右 `3.489 / 3.476 mm`，6 mm 固定为 `3.513 / 3.500 mm`，差约 `0.025 mm`。6 mm 固定实验中保持结束的最小变形 Jacobian 为 `0.894 / 0.889`。固定范围变化没有让响应出现明显改变，不支持将当前主要问题归因于固定端滑动。它不能反推实际安装件必须固定几毫米。

`031` 的全部 250 帧固定区误差、节点保持稳定性、RGB 保持稳定性、初始相机投影、四路视频逐帧解码和方块碰撞顶点距离均通过。所有记录帧的变形 Jacobian 为正，最小左/右 `0.906 / 0.907`。碰撞顶点无穿透不能排除仅发生在边或面上的相交。

## 更深挤压实验

`033_conforming_deeper_squeeze` 使用 44 mm 闭合指令、修正后的初始化、内部自接触和 80 次迭代。材料、时间步、4 mm 固定区保持；它不是与 52 mm 实验相同的加载条件。

保持结束的最大接触位移左/右 `8.037 / 8.049 mm`，最佳刚体拟合后的非刚体残差 RMS 约 `3.05 mm`；真正横向梁的最大曲线到端点连线偏差约 `0.162 / 0.163 mm`。显著形变主要在纵向侧壁，不能把 8 mm 节点位移称为横梁弯曲量。

全部记录帧最小变形 Jacobian 左/右 `0.820 / 0.827`，固定区最大偏移约 `0.000303 mm`。碰撞顶点最大穿透约 `0.292 / 0.300 mm`，在原 0.5 mm 上限内。冻结物理重复渲染时所有记录的物理状态差为零，三路图像稳定检查通过。

但是 `033` 未通过保持段 RGB 稳定验收：排除最初 0.25 s 后，左右最差帧 P95 为 `0.700 / 0.623 px`，超过原 `0.5 px` 上限；物理投影节点 P95 仍在上限内。完整数据与四路 MP4 解码通过，但综合 `validation.json.valid=false`。保留原始视频，不做时间滤波或阈值放宽。

`clamp_audit_033/deformation_profile.mp4` 是实际 FEM 节点在底座坐标系的侧面投影：灰色为参考、蓝/橙为当前自由节点、红色为固定区域，位移比例 x1。它是诊断可视化，不是额外相机的 RGB，也不是新的物理仿真。

随后 `034_conforming_solver200` 只将 `033` 的求解迭代从 80 提高到 200，没有改变材料或闭合量。保持结束最大接触位移为左/右 `8.340 / 8.236 mm`；所有记录帧最小 Jacobian 为 `0.867 / 0.867`，固定区最大偏移约 `0.000749 mm`。碰撞顶点最大穿透降至约 `0.000003 / 0 mm`，但稳定性更差：节点投影最差帧 P95 为 `0.722 / 0.724 px`，RGB 为 `0.981 / 1.099 px`，均超过原 0.5 px 上限。冻结物理渲染仍然通过，说明不能仅增加迭代数就宣布强接触保持稳定。

| 实验 | 闭合指令 | 固定区 | 迭代 / 自接触 | 保持结束最大接触位移，左 / 右 | 验收 |
| --- | --- | --- | --- | --- | --- |
| 029 原自动体网格 | 52 mm | 4 mm | 80 / 关 | 4.192 / 4.020 mm | 历史稳定基线 |
| 031 贴合网格 | 52 mm | 4 mm | 80 / 关 | 3.489 / 3.476 mm | 数据、接触顶点、节点和 RGB 稳定检查通过 |
| 032 固定区敏感性 | 52 mm | 6 mm | 80 / 关 | 3.513 / 3.500 mm | 无 RGB；各阶段检查点固定误差和 Jacobian 通过 |
| 033 深挤压 | 44 mm | 4 mm | 80 / 开 | 8.037 / 8.049 mm | RGB 保持稳定未通过 |
| 034 深挤压 | 44 mm | 4 mm | 200 / 开 | 8.340 / 8.236 mm | 节点投影及 RGB 保持稳定未通过 |

当前稳定通过的贴合网格对照是 `031`。`033/034` 可以证明固定区以外存在明显非刚体弯曲，但仍属于强挤压诊断，不能作为稳定训练数据或实物准确度的结论。各条件的原始视频、250 帧节点与 JSON 均保留，没有降低验收门槛。新增/相关单元测试共 33 项通过，包括刚体运动不产生应变、叠加旋转不掩盖应变、翻转单元识别、夹持滑移和完整容纳单元搜索。

可查看：

- [52 mm 稳定对照原始 RGB 视频](../data/generated/robodojo_episode_031_conforming_surface/recording/episode.mp4)
- [44 mm、80 次迭代的真实比例侧面节点视频](../data/generated/clamp_audit_033/deformation_profile.mp4)
- [44 mm、200 次迭代的侧面节点视频](../data/generated/clamp_audit_034/deformation_profile.mp4)
- [分条件原始 RGB 对照图](../data/generated/conforming_fem_comparison/comparison_hold.png)

对照图明确标注闭合指令、材料、迭代数、自接触状态和是否通过验收，不能将不同载荷的更大形变解释成材料模型已被验证更准确。

## 数值边界

显式小单元路径在 PhysX 初始化时出现 `updateNodeInverseVolumes(): tetrahedron is degenerate or inverted` 报错。官方实现对行列式使用固定绝对阈值 `1e-9`，在米制小单元上即使参考体积为正也可能触发；本例 25442/50659 个单元低于该阈值。来源：[PhysX ExtDeformableVolumeExt.cpp](https://raw.githubusercontent.com/NVIDIA-Omniverse/PhysX/main/physx/source/physxextensions/src/ExtDeformableVolumeExt.cpp)。当前安装版本报错行号与主分支不同，不能仅凭这条解释声称所有初始化告警无害。

因此仍须独立检查节点有限性、固定误差、实际变形 Jacobian、接触和保持稳定性。体积贴合本身不能证明刚度收敛、材料标定或实物一致性。新路径保留为显式可选项，未悄悄替换默认求解配置。
