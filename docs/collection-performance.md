# 采集代码性能验证

后续的单机优化重点、RoboDojo 资源资料和“短测筛选后再完整回放”的验证顺序，见
[RoboDojo 资源需求与软指优化重点](robodojo-resource-notes.md)。

本轮优化针对自有代码，不修改 RoboDojo、PhysX FEM、1000 Hz 物理步长、相机数量、
分辨率、RTX 子帧数或光照采样数。

- 每个 phase 一次生成全部位置、速度和夹爪宽度，再一次传入设备。原先每个物理步都校验
  整段轨迹、插值、分配数组并分别传输位置和速度。测试逐值比较全部采样时刻，包括路径
  节点、最终时刻、单点保持和张开／闭合；float32 控制值也必须一致。
- 安装在原生滑块上的固定变换只计算一次。纯物理步继续更新 FEM 目标和夹具检查，
  相机 USD 位姿写入延至渲染前；每次渲染仍采用最新测量安装位姿。
- 显示网格插值、FEM 校验和数据记录共用同一份节点快照。双工具每帧的节点读取从
  7 次减至 3 次（一次读取包含两根手指），显示网格写入从 6 次减至 4 次。
- 静止参考四面体体积在初始化时缓存，逐帧仍计算当前体积、雅可比和夹持误差。

capture 仍执行 Fabric 刷新、零时间 Replicator 渲染和渲染后 FEM 回读，若节点位移
超过原有的 1e-7 m 阈值就拒绝采样。三路相机仍在同一次渲染中采样。

## 复现

```bash
python scripts/run.py test --group unit
python scripts/run.py test --group performance
```

性能测试先检查输出等价性和读取／写入次数，再预热两种实现、交替计时 5 轮，比较中位数。
每轮内重复运行降低计时噪声。独立性能组会断言轨迹准备至少加速 2 倍、CPU 帧暂存至少
加速 1.05 倍、CPU 相机跟随至少加速 1.1 倍；普通单元测试只检查确定性的正确性和操作次数。

结果及每轮原始计时保存至
`artifacts/cache/benchmarks/collection_performance.json`，每次测试覆盖。

2026-11-14 在 core 环境（Python 3.12、NumPy 2.3.4）的实测中位数：

| 测量范围 | 优化前 | 优化后 | 加速倍数 |
|---|---:|---:|---:|
| 1500 步命令准备，257 个轨迹点 | 25.51 ms | 0.154 ms | 166.2× |
| 单帧 CPU 采集暂存 | 14.11 ms | 9.86 ms | 1.43× |
| 500 步相机跟随，包含安装位姿校验 | 76.64 ms | 51.01 ms | 1.50× |

CPU 帧暂存测试使用每根手指 8491 个 FEM 节点、20000 个显示顶点、256×256 RGB/depth；
模拟 Kit 接口，实际执行 NumPy 复制、显示网格插值、图像转换和物理静止检查。对照路径
保留优化前的重复节点读取与显示更新顺序，两条路径共用图像检查实现。
相机测试使用 500 个运动物理步，按 20 FPS 写入位姿，模拟 USD 接口。

这些测试衡量 CPU 代码片段，**不包含 PhysX、RTX 或真实 CUDA 传输，也不代表整个
episode 的加速倍数**。需在具备 NVIDIA 驱动的环境执行完整 episode，才能测量总收益。

## GPU 完整回放对比

```bash
OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
SOFT_SIM_PERFORMANCE_SOURCE_RUN=20261114T091003Z_general-pickup_s22_72e465 \
python scripts/run.py test --group gpu-performance --verbose
```

该组在 RoboDojo 环境运行，需要能访问真实 GPU；若执行器隔离了 GPU，应在沙箱外执行。
使用指定成功 run 的配置和已执行计划，依次完整回放优化前、优化后两条 episode。
两次都运行原任务物理、四根 FEM 软指、三路相机、逐帧校验及真实暂存／数据集提交。
使用新的独立数据集，不向源 run 的数据集追加。

基线默认从优化前提交 `1a51a2b3a67e47d8e9467e2e71a50480edb873ac` 读取
execution、soft_tool、collect 三个模块，复制到基准产物目录，仅给 collect 插入相同范围的
计时。其余模块在测试开始时一并快照，供两次回放共用，保留相机配置等独立修复，并防止
共享工作区在两次进程启动之间发生变化。可用
`SOFT_SIM_PERFORMANCE_BASELINE_REV` 指定其他优化前提交；工作区源码不切换、不覆盖。

测试检查配置、计划、物理步数、采样帧数、各 phase 步数和渲染参数一致，要求两次任务成功、
三路画面全部提交，并断言完整采集墙钟时间缩短。启动、场景构建和最终视频编码不计入
采集耗时。结果、来源提交、源码哈希、GPU 信息、各 phase 计时及原始日志写入
`artifacts/cache/benchmarks/gpu_<timestamp>/`。单次 A/B 是该硬件上的一次观测；
评估稳定收益时，应多次复跑并观察温度、频率和其他 GPU 负载。

### 本次 GPU 结果

2026-11-14 在沙箱外使用 RTX 4070 Laptop（驱动 580.178.04）完整回放同一计划，
两次均为 13,440 个物理步、268 组同步画面，官方任务判定成功，数据集均已提交。

| 计时范围 | 优化前 | 优化后 | 耗时减少 |
|---|---:|---:|---:|
| 完整采集 | 669.11 s | 655.69 s | 13.42 s，2.01% |
| 采样、校验和记录 | 189.67 s | 177.39 s | 12.27 s，6.47% |
| 其余控制与物理步等 | 479.44 s | 478.29 s | 1.15 s，0.24% |

CPU 微基准的倍数不能直接应用于整条 episode；本次主要收益来自采样、校验和记录，
其余控制与物理步等仍占大部分时间。这里只报告单次 A/B 的观测，未证明 2% 是稳定收益。

证据保存在 `artifacts/cache/benchmarks/gpu_20261114T115509Z/comparison.json`，
完整日志及各 phase 计时在同目录。首次运行时共享工作区新增了默认的 `task.robot: x5`
字段；旧版固定支持 X5，两个导出场景的 robot_name 也都是 x5。配置比较将旧版缺省值
归一化为 x5，原始配置快照不改写。两次代码启动时未冻结公共源码，后续测试已补上快照
隔离，因此本组结果也应按一次观测解读。

仅重新校验这两条已完成回放的原始配置、计时、任务和提交记录，可执行：

```bash
SOFT_SIM_PERFORMANCE_REPORT=artifacts/cache/benchmarks/gpu_20261114T115509Z/comparison.json \
python scripts/run.py test --group gpu-performance --verbose
```

该方式不重新运行物理仿真，会明确打印 `GPU_PERFORMANCE_RECHECK`。

## 求解器迭代次数扫描

完整 episode 只能给出"每物理步 34 ms"，无法说明其中多少是变形体求解。
`tests/integration/gpu_solver` 在隔离的双指 FEM 上（无机器人、桌面、物体、接触、渲染）
按多个 `solver_iterations` 各跑一段物理，把每步拆成两项：

- `fixed_ms_per_step`：与迭代次数无关的固定开销（场景更新、变形体簿记、同步、kernel 发射）
- `marginal_ms_per_iteration`：每增加一次迭代的边际成本

同时记录相对最高迭代次数的变形漂移（末帧节点最大位移差 / 参考最大下沉量），
因为减少迭代只有在手指仍然按同样方式变形时才是免费的。

```bash
python scripts/run.py test --group gpu-solver --verbose
```

可用环境变量：`SOFT_SIM_SOLVER_ITERATIONS=80,40,20,10`、
`SOFT_SIM_SOLVER_DURATION_S=0.3`、`SOFT_SIM_SOLVER_MIN_FREE_MIB=3000`、
`SOFT_SIM_SOLVER_DRIFT_LIMIT=0.05`。显存不足（另一个 Isaac Sim 实例在跑）时该组会
明确 skip 并给出空闲显存数值，不会静默通过。结果写入
`artifacts/cache/benchmarks/solver_<timestamp>/sweep.json`，每档的日志、
`report.json` 与末帧节点快照同目录保存。

该扫描只回答"80 次迭代花掉多少时间、少迭代会带来多大变形误差"。是否降低迭代次数还需
完整 episode 复跑一次，确认抓取仍成功且 `object_lift_m` 不变。

2026-11-14 在 RTX 4070 Laptop（8 GB）上的实测（隔离双指探针，1 s 物理，3 轮交替取中位数，
轮间离散 1–10%，逐条证据见 `artifacts/cache/benchmarks/solver_20261114T131545Z/sweep.json`）：

| 迭代次数 | 稳态 ms/步 | 变形漂移 | 相对漂移 | GPU 功率 | GPU 利用率 |
|---:|---:|---:|---:|---:|---:|
| 80 | 10.74 | 0.000 mm | 0.0% | 51.8 W | 78% |
| 40 | 5.59 | 0.319 mm | 8.2% | 49.1 W | 62% |
| 20 | 3.25 | 3.101 mm | 79.4% | 45.4 W | 20% |

拟合：`step_ms = 0.68 + 0.1254 x iterations`，**迭代次数解释了 93.4% 的每步耗时**。
两项结论：

- 80 次迭代不是“白给的保险”：它占了几乎全部求解时间，但降到 40 次就会让自由悬垂
  变形偏 8.2%，20 次完全不可用（79%）。是否接受 40 次必须完整 episode 验证。
- 该探针在 80 次迭代时 GPU 利用率 78%、功率 51.8 W（功耗墙 55 W），说明变形求解确实
  把 GPU 压满了；“等同步”不是主要成本，**功耗墙和串行迭代才是**。

显存注意：卡空闲时 Kit 会按纹理预算自己涨到 7.3 GB（卡被占时只到 4.0 GB），所以
8 GB 卡上同时只能跑一个 Isaac 进程；扫描中途有别的实例启动会直接 OOM（测试因此会在每个
候选前重查空闲显存并明确报错）。

## 实际采集计时

后续 collect/replay 会生成 `runs/<run_id>/diagnostics/collection_timing.json`，记录：

- 总采集墙钟时间、物理步数、仿真时长、实际记录帧数，以及 dt、FPS 和渲染设置。
- 各 phase 的墙钟时间、命令准备时间、采样与校验／记录时间。

总计时从首次执行前开始，包含后续 operation 的规划，不包含初始场景搭建、首次规划及
最终数据集转换。GPU 操作可能异步，分项计时反映调用链的墙钟耗时，并未插入额外 CUDA
同步；对比完整 episode 时，应使用相同配置、计划、采样数量及相同范围的总墙钟时间。
