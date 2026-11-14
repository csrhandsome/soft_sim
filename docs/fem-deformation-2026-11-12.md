> **入口变更（2026-11-14）**：任务入口已统一为 `benchmarks/robodojo_task_soft.py`（原双 X5）。
> 本文命令中的旧 `robodojo_soft_pick.py` 已移除；网格与材料结论仍然有效。

# STL 软指形变排查（2026-11-12）

> **路径变更（2026-11-14 重构）**：日常入口改为 `python scripts/run.py <command>`，实现位于 `src/soft_sim/`，生成物位于 `artifacts/`。本文保留写作时的目录与命令，当前命令见 [README](../README.md)。


STL 加材料的弹性参数足够建立按 STL 实体制造的初始模型。当前继续使用拓竹 TPU 85A 初值：杨氏模量 5.2 MPa、密度 1180 kg/m³、泊松比假设值 0.45；没有通过降低杨氏模量来放大形变。打印参数和实测曲线可用于之后的实物标定，不是继续修正仿真的前置条件。

## 已确认的表面离散问题

原始 STL 尺寸为 40 × 40 × 88.18 mm，812 个顶点、1700 个三角面。最长三角边为 84.417 mm；506 个三角面至少有一条边超过 10 mm。它可以准确描述未变形的平面 CAD 表面，但长三角面只能连接少量变形顶点，不能呈现侧壁等长条结构内部沿长度方向的弯曲。

读取 `robodojo_episode_027_contact_render_quality` 在 6.0 s 的真实保持节点，将原 STL 长边内部的采样点绑定到同一份求解四面体：有限元位移场的曲线与只连接边两端的直线，左右最大差异为 2.989 / 2.793 mm。这里测量的是显示离散可能遗漏的局部弯曲，不是夹持压入深度，也不是材料误差的实测值。

后续分区检查确认上述最大值主要来自纵向侧壁长边，不能称为横向梁弯曲；真正横向梁的测量和求解体网格偏大的新发现见 [贴合体网格迭代](conforming-fem-2026-11-12.md)。

可独立复现，不需要启动 GPU 仿真：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmarks/check_surface_deformation.py \
  data/generated/robodojo/conforming/robodojo_episode_027_contact_render_quality \
  --output data/generated/robodojo/conforming/fem_deformation_audit
```

输出 `surface_audit.json` 和 `beam_curve.npz`；本轮另生成 `beam_curve.png` 展示同一梁边在相同物理状态下的曲线与直线差别。

## 修正方式

`--surface-max-edge-m .002` 在原 STL 表面添加共享的边中点，生成 48265 个显示顶点、96606 个三角面，实际最长边约 1.995 mm。没有移动原表面、加厚横梁、封闭镂空或缩放模型。测试确认封闭性、体积、尺寸、采样点内外关系和三角面朝向保持。

细分仅用于独立显示网格。显示点通过重心坐标绑定到实际 cooking 后的有限元四面体，在渲染前用当前求解节点更新。碰撞网格保持原值；没有平滑物理节点、扩大显示位移或对 RGB 做时间处理。通过下面的 GPU 验证后，主入口和 `MountedSoftTool` 默认启用 2 mm 显示细分；`--surface-max-edge-m 0` 保留原始表面显示以便对照。录制元数据和 `result.json` 的 `deformable_mesh` 保存显示细分与体网格设置。

对 `027` 的全部 250 帧离线重建细分显示表面，解析方块顶点距离检查的最大显示穿透为左 0.092 / 右 0.366 mm。该结果与原碰撞网格的顶点距离不同，表明显示表面与碰撞代理仍有近似误差；顶点检查也不能排除仅发生在边或面上的相交。不能将“原碰撞顶点无穿透”直接当作新显示表面的结论。

直接用约十万个细分三角面重新生成碰撞四面体的尝试耗时过长，已停止，没有采用，也没有改变默认碰撞表面。该诊断目录为 `fem_mesh_refined_88`，不应作为完成的仿真结果。

## 体网格对照

新增 `--simulation-resolution`，控制最长轴上的 PhysX 体网格单元数；原值为 44，名义单元大小约 2.004 mm。88 对应约 1.002 mm。

原 44 网格为左右 6608 / 6665 个节点；88 网格为 33210 / 33427 个节点，116605 / 116985 个四面体，两侧均连通且连接固定底座。

本轮 `robodojo_episode_028_fem_resolution_88` 使用 88 网格、200 次迭代，其他弹性参数、52 mm 闭合指令和 1 m/s 接触修正速度上限保持。保持阶段记录到的最大节点位移约 3.495 mm；`027` 原 44 网格、80 次迭代约为 4.2 mm。这同时改变了网格和迭代数，是数值敏感性检查，不是单参数因果对照，也不能凭两个设置宣称网格已经收敛。因此没有把更细网格作为“材料变软”的修正。

当前 PhysX 本构为共旋线弹性近似，见 [PhysX 本构说明](https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#constitutive-model)；没有根据完整应力—应变曲线建模 TPU 的非线性、方向性和滞回。STL 与现有弹性参数可以支持当前初始模型，精确复现实物响应仍需之后验证。

## 带细分显示的运行命令

`robodojo_episode_029_refined_surface` 已完成完整 10 秒、250 帧、25 FPS 的三路采集和四路 MP4 逐帧解码。物理参数与稳定基线 `027` 一致，仅增加细分显示。

| 保持段指标 | `027` 原显示 左 / 右 | `029` 细分显示 左 / 右 |
| --- | --- | --- |
| 原始 RGB 最差帧位移 P95（px） | 0.226 / 0.258 | 0.208 / 0.339 |
| 节点投影最差帧位移 P95（px） | 0.018 / 0.036 | 0.021 / 0.036 |
| 相对接触前最大节点形变（mm） | 4.204 / 4.045 | 4.208 / 4.054 |

投影、固定安装、初始化、节点保持运动、RGB 稳定性、原碰撞顶点距离和数据完整性检查全部通过。冻结渲染的物理状态差全部为零，左右图像指标约 0.126 / 0.095 px。以上是当前静止挤压测试的验收，不能替代运动抓取或实物材料标定。

新版视频为 `data/generated/robodojo/conforming/robodojo_episode_029_refined_surface/recording/episode.mp4`；同一时间的原始左右 RGB 对照为该目录的 `comparison_hold.png`。细分显示修复了已确认的曲线显示遗漏；体网格与材料本构的定量精度仍需独立验证。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  envs/robodojo/.venv/bin/python benchmarks/robodojo_soft_pick.py \
  --grasp-plan data/generated/archive-early/rigid_box_plan/plan.json --mode squeeze --squeeze-width .052 \
  --local-physics-origin --dt .001 --solver-iterations 80 --simulation-resolution 44 \
  --surface-max-edge-m .002 --max-depenetration-velocity 1 \
  --rt-subframes 4 --lighting-samples 8 --record-episode --video-fps 25 \
  --closeup --frozen-render-check --record-collision-nodes \
  --output data/generated/robodojo_episode_NEXT_refined_surface
```

单元验证：在 RoboDojo 环境中，表面细分、有限元显示绑定、相机运动、冻结渲染、RGB 稳定性、方块距离与共享逻辑共 28 项测试通过，无跳过。

```bash
OPENBLAS_NUM_THREADS=1 envs/robodojo/.venv/bin/python -m unittest \
  tests.test_mesh_refinement tests.test_finger_motion tests.test_frozen_render \
  tests.test_camera_stability tests.test_box_contact tests.test_shared -v
```
