# 实测位姿到 FEM 目标的 GPU 更新

`sim/gpu_follow.py` 在本地 Isaac Sim 5.1 运行时使用 Warp，将实际 articulation link
位姿转换成安装坐标系和 FEM 目标。固定变换、局部节点和夹持掩码只上传一次。
每步在当前 Torch CUDA stream 上计算安装位姿、四指目标和夹具误差，通过一个批量
PhysX 软体视图提交目标。软体视图的节点顺序在初始化时用各单指视图的完整状态核对，
不假定 USD 遍历顺序等于机器人／左右手指顺序。

四元数归一化、矩阵和节点变换保留原 SciPy/NumPy 路径的 float64 计算，最终目标仍为
float32。不使用命令关节角替代实际 link 位姿，不改求解器迭代数、步长、材料、网格、
夹持掩码、碰撞过滤、相机或验收阈值。夹具的独立 PhysX 刚体视图仍在每步读取并计算
该步误差；仅将误差和安装位姿的 CPU 下载推迟到采样／诊断实际使用时。

GPU 路径要求软体 sleep threshold 为零。它会重复提交静止目标，所以不能直接用于
依赖睡眠省算力的独立场景。原 CPU 路径仍保留逐值相等时不重复提交的行为。

## 对照测试

```bash
python scripts/run.py test --group unit
python scripts/run.py test --group gpu-follow --verbose
```

`gpu-follow` 必须能访问真实 CUDA。数值测试用不同位姿、非单位四元数、180 度旋转和
每指 8491 个节点，对照独立 SciPy/NumPy 计算；还验证非顺序 PhysX 行映射、固定掩码、
静止重复更新，以及纯物理跟随路径不会调用 `Tensor.cpu()` 或 `Tensor.item()`。

完整抓取 A/B 需要一个成功的 general_pickup run，配置必须是 80 次迭代和 1 ms 步长：

```bash
OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SOFT_SIM_FOLLOW_SOURCE_RUN=20261114T091003Z_general-pickup_s22_72e465 \
python scripts/run.py test --group gpu-follow --verbose
```

测试冻结同一份源码、配置和已执行计划，在独立进程依次运行 GPU、CPU、CPU 重复跟随。
三次都执行完整物理、三路渲染、帧校验和真实数据集提交。基准数据写入新的独立数据集。
`artifacts/cache/benchmarks/follow_replay_<timestamp>/comparison.json` 保存源文件哈希、
原始日志、配置、耗时和任务结果；三份 `*_states.npz` 保留逐采样时刻的数值状态。

测试比较相同步数、采样时刻、命令和计划，并要求三次官方任务判定成功、三路画面提交。
每帧检查所有手指的 Jacobian 为正、夹持误差小于原有 10 μm。安装矩阵元素差异始终
小于 1e-6，关节位置差异小于 1e-5，控制命令绝对差异小于 1e-12 rad。
停靠臂的保持命令来自预稳定后的实测关节值；相同 CPU 重跑也会在数值零附近出现约
1e-18 rad 的差异，因此不要求这些保持值逐位相同。

求解后的 FEM 节点不假定逐值确定：用相同 CPU 路径的第二次回放直接测量重复运行差异。
求解状态的比较阈值为 `min(硬上限, max(绝对下限, 2 × CPU 重跑最大差异))`。
节点和物体位置的绝对下限为 0.1 mm；节点硬上限为实际缩放网格 pitch 的一半
（本参考任务为 1.5 mm），物体位置上限为 1 mm，方向最短夹角上限为 2 度，
Jacobian 差异上限为 0.03。
CPU 自身重跑差异超过硬上限也会失败。最终抬升差异有独立的 0.5 mm 硬上限。
节点和物体位置使用最大三维欧氏距离，物体方向使用归一化四元数的最短角距离，
矩阵等其他数组使用元素绝对差最大值。
原始差异、CPU 重跑差异和最终阈值全部写入报告。这是在测量同一离散求解器的重复性，
不是实物精度证明，也没有放宽采集自身的物理验收阈值。

`control_and_physics_s` 包含每步控制目标设置、外层 scene.update、工具跟随以及完整
env.sim_step 的调用墙钟时间，排除逐帧采样和记录；它不是纯 PhysX kernel 时间。
计时未在每个物理步增加 CUDA 同步，因此完整采集墙钟时间也必须一起比较，以防异步工作
只是转移到了采样阶段。测试要求 GPU 两项耗时都低于两次 CPU，原始各 phase 耗时均保留。

只重新核验保留证据、不启动仿真：

```bash
SOFT_SIM_FOLLOW_REPORT=artifacts/cache/benchmarks/follow_replay_<timestamp>/comparison.json \
python scripts/run.py test --group gpu-follow --verbose
```

已有完整变体不会重跑；若需要继续缺失的变体，设置 `SOFT_SIM_FOLLOW_RESUME` 指向该
报告，并保留上面的 EULA 和线程环境变量。续跑前会检查冻结源码的全部哈希。

本优化不处理项目已有的小单元初始化告警；相关解释和物理准确度边界见
[贴合网格数值边界](conforming-fem-2026-11-12.md#数值边界)。性能对照不能替代参考状态、
网格收敛和材料标定验证。
