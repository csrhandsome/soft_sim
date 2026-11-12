# 软指仿真参考源码

这些目录是用户要求的本地 Git clone，保留各自 Git 历史和许可证，不将第三方源码提交到本项目。固定的实际提交见 [sources.json](sources.json)。未递归拉取各项目的构建子模块，也未安装其完整运行环境。

| 目录 | 上游 | 本轮阅读重点 |
| --- | --- | --- |
| `SoftRobots/` | https://github.com/SofaDefrost/SoftRobots | `examples/tutorials/SoftFingerDesign/details/actuated_finger.py`：对安装区域做 `Rigidify`、对内部表面分组碰撞；`CableGripper/details/finger.py`：固定区和独立碰撞表面 |
| `STLIB/` | https://github.com/SofaDefrost/STLIB | `python3/src/stlib3/physics/constraints/fixedbox.py`：ROI 加高刚度约束；`physics/deformable/elasticmaterialobject.py`：体网格、本构和重心映射 |
| `diff_pd_public/` | https://github.com/mit-gfx/diff_pd_public | `python/py_diff_pd/common/tet_mesh.py`：用 TetGen 从封闭表面生成贴合四面体网格 |
| `diffPD_sim2real/` | https://github.com/srl-ethz/diffPD_sim2real | `python/realbeam_experiments/Environments/beam_env.py`：固定三个方向的节点自由度；`clamped_beam_Case_A-1.py`：相同材料下比较四面体/六面体；材料、阻尼分别标定 |
| `somo/` | https://github.com/GrauleM/somo | `somo/sm_continuum_manipulator.py`：固定基座；其弹簧关节刚性连杆近似不适合验证本 STL 横梁局部应变 |

## 论文与代码可用性

- [Fin-QD](https://arxiv.org/html/2311.12477v1)：公开论文描述 Gmsh 四面体、SOFA、横梁自接触和材料实验；本轮未找到可确认属于该论文的公开代码仓库，不能声称已 clone。
- [Vision-based Force Estimation Using Deformable Grippers](https://rave.ohiolink.edu/etdc/view?acc_num=case1718120033936051)：公开论文讨论 TPU 夹爪、视觉形变和 SOFA 逆有限元；本轮未找到可确认配套的公开代码仓库。

## 在本项目中借鉴的方法

1. 固定区误差必须单独验收，不能只检查相机稳定或网格连接。
2. 除总体积，还需检查细梁截面、孔洞、接触和数值收敛。本轮发现旧 PhysX 自动体网格体积约为 STL 的 1.85 倍，增加 TetGen 显式网格对照路径。
3. 横梁曲率和侧壁长边曲率分开测量；旧显示诊断的约 3 mm 最大值来自纵向侧壁长边，不能称为横向梁弯曲。
4. 材料继续使用 TPU 85A 的 5.2 MPa 初值。不同项目材料和硬度对应的参数不能直接替换。

复现实验、结果与已知限制见 [本轮检查记录](../docs/conforming-fem-2026-11-12.md)。
