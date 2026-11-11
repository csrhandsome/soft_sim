> 当前安装与验证命令见 [根 README](../README.md)。以下旧 Isaac 6.2 实验说明保留作参考；RoboDojo 主环境为 5.1。

# 双软手指与指内相机

当前场景把一份 STL 实例化为左右两根 TPU 软手指。每根手指底部各有一台相机，镜头沿该手指的局部 +Z 看向内部横梁。左右相机使用完全相同的局部安装位置和安装旋转，随各自手指底座运动；焦距可以独立随机化，也可以在扫描实验中设为同一值。软指是体变形模型，不是只会变换位姿的刚性 STL。

## 代码分层

```text
shared/
  config.py        SI 单位配置、材料初值、参数验证
  mesh.py          STL、体网格、拓扑与源文件校验
  geometry.py      手指/相机坐标变换、相机内参
  experiment.py    物体采样、夹爪构形、安装倾角、闭合轨迹
  environment.py   BaseGripperEnv、Observation、ForceReading
  dataset.py       统一采集、NPZ、JSON、PNG 预览
  force.py         无原生力报告时的显式准静态估计
  learning.py      双相机图像估力的 NumPy ridge 基线
  cli.py           两个后端共用的实验命令
backends/mujoco_sim/soft_gripper_env.py     MuJoCoGripperEnv(BaseGripperEnv)
backends/isaac_sim/soft_gripper_env.py   IsaacSimGripperEnv(BaseGripperEnv)
backends/isaac_sim/assets.py            官方/本地 USD 路径解析
backends/isaac_sim/inspect_assets.py    现成资产的关节与连接点检查
```

新增后端只需继承 `BaseGripperEnv`，实现建场景、设置边界状态、物理步进、节点/物体状态读取、接触力和双相机渲染。共享层执行相同轨迹和采集流程，后端分别映射自己的材料、软体和传感 API。

## 构形、倾角与相机焦距

| 配置 | 含义 |
| --- | --- |
| `gripper.profile=robotiq_2f85` | 参考 DROID 使用的 Robotiq 2F-85 的平行夹持分支；近似连杆引起的指端升高 |
| `gripper.profile=panda` | 平行直线开合，最大标称开口 80 mm |
| `gripper.profile=angular` | 两侧底座绕相反的 Y 轴向内转动 |
| `mount_tilt_range_deg` | 软手指相对安装座的向内倾角；平行构形中每个实验内恒定 |
| `final_jaw_width_range_m` | 平行构形的闭合终点；与软指安装倾角独立 |
| `close_angle_range_deg` | 仅角度构形使用的最终闭合增量；叠加安装倾角 |
| `adapter_offset_m` | 每侧从标称刚性指面到软指底面中心的外移适配距离，默认 20 mm |
| `camera.position_in_finger_m` | 每根软指内固定的镜头位置，默认距底部 6 mm |
| `camera.mount_rpy_deg` | 相机相对软指的固定安装旋转，两台相同 |
| `camera.focal_length_range_mm` | 焦距采样范围；像素焦距 `fx=width*focal_length_mm/sensor_width_mm` |

`jaw_width_m` 是替换软手指之前的标称刚性指面开口，不是变形后两个软表面的实际最小距离。安装座中心间距为 `jaw_width_m + 2*adapter_offset_m`。软指本身约 40 mm 宽，需要适配件；这里没有推断真实螺孔或机械安装尺寸。

这些构形是可复现的运动学边界条件，并非完整 Robotiq 机构的动力学复刻。Robotiq 预设参考公开 MJCF 的 55/37.5 mm 连杆向量和 0–0.8 rad 范围，近似平行夹持时的指端轨迹；没有求解完整四连杆、包络抓取或电机力矩。完整官方刚性模型当前可检查/导出，但尚未装配到采集后端中。

焦距只改变成像倍率，镜头在软指内部的位置和朝向保持固定。相机跟随的是固定底座坐标系，不随软材料的局部弯曲改变安装姿态，符合镜头刚性安装在底座的假设。若实际镜头贴在可弯曲区域，需要改为绑定该区域的局部形变坐标系。

## 模型与材料

STL 没有单位。读取到 812 个唯一顶点、1700 个三角面，封闭、单连通部件，边界尺寸 `40×40×88.18`；默认按毫米导入，成为 `0.04×0.04×0.08818 m`。如果源文件实际单位不同，修改 `mesh.scale_to_m` 后重新生成网格。

默认附带 2 mm 体素体网格，4261 个节点、13152 个四面体。总体积相对 STL 约大 2.4%；该数值仅是体积检查，不能代表局部几何或刚度误差。体素是保留可分辨孔洞的近似，会改变细梁截面和角部。精细力研究应使用贴合表面的体网格，并检查网格与时间步收敛。

```bash
# 默认仅依赖 NumPy，两个环境使用同一份生成文件
uv run --no-sync python scripts/prepare_mesh.py

# 可选 Gmsh 贴面网格；首次需要联网解析 gmsh，Linux 可能需要系统 libGLU
# Gmsh 路径当前尚未实际运行验证
uv run --with gmsh python scripts/prepare_mesh.py --method gmsh --pitch 0.003
```

网格保存源 STL 的 SHA256、单位、分辨率、生成方法和体积。STL 或单位改变后，旧网格会被拒绝，防止两个环境意外模拟不同几何。

TPU 85A 初值来自拓竹 TDS：密度 `1180 kg/m³`，Z 向杨氏模量 `5.2 MPa`，XY 向 `6.8 MPa`。当前采用 Z 向模量的各向同性近似；泊松比 `0.45`、摩擦 `0.7` 和阻尼都是待标定假设。没有用 Shore 85A 硬度直接推算力值。真实打印的填充、壁厚、方向、应变率、非线性与滞回尚未建模，应使用实测力—位移曲线标定。两个引擎的阻尼语义不同，具体映射写入实验元数据，不承诺相同参数产生相同力曲线。

MuJoCo 默认使用三线性插值，网格控制单元为 `2×2×5`，固定底部插值平面，用于当前可运行的初版；它比完整节点自由度更硬，会影响力估计。完整节点模式和 Isaac 固定 `mesh.pin_height_m` 指定的底部厚度带；两种固定方式的差别写入 MuJoCo 后端元数据。完整节点模式可设 `simulation.mujoco_dof=full`，但细网格和 MPa 级材料需要大幅增加迭代数/调整时间步。当前完整节点模式在默认求解设置下会出现惯量/迭代收敛警告，因此程序会拒绝该实验，不会写入训练数据。

## 采集

以下命令均从仓库根目录运行。

```bash
# 先采三种基本形状，默认每次随机物体、安装倾角和两台相机的焦距
MUJOCO_GL=egl uv run --no-sync python backends/mujoco_sim/run.py --episodes 10

# 无相机的物理验证，无法用于图像训练
uv run --no-sync python backends/mujoco_sim/run.py --no-render --output data/runs/physics

# 图形窗口；有窗口的环境可按需不设置 MUJOCO_GL
uv run --no-sync python backends/mujoco_sim/run.py --gui --output data/runs/gui

# 同一物体，系统比较 3 个安装倾角 × 2 个焦距 × 2 个闭合终点
MUJOCO_GL=egl uv run --no-sync python backends/mujoco_sim/run.py \
  --episodes 3 --sweep-tilts 0 5 10 --sweep-focals 2 3.2 \
  --sweep-widths 0.01 0.025 --output data/runs/sweep

# Panda / 角度构形
uv run --no-sync python backends/mujoco_sim/run.py --config configs/panda.json --no-render --output data/runs/panda
uv run --no-sync python backends/mujoco_sim/run.py --config configs/angular.json \
  --sweep-angles 15 22 28 --no-render --output data/runs/angular

# Isaac Sim，在宿主机有 GPU 的终端运行
python backends/isaac_sim/run.py --episodes 3
python backends/isaac_sim/run.py \
  --config configs/isaac_ycb.json --episodes 3 --output data/runs/isaac_ycb
```

采集到的左右 RGB、光轴深度、相机内外参、当前开口和倾角、物体位姿、软指节点及去除底座刚性运动后的形变按帧保存。每个实验有 `reference.npz`、`reference_0/1.png`、`final_0/1.png`、`frame_*.npz`、`episode.json`；左右顺序固定。相机外参使用光学坐标 `+X 右、+Y 下、+Z 前`。背景/无效深度为 NaN。图像和力标签来自同一物理状态。

完整实验先写入隐藏的 `.incomplete` 目录，成功后才改为正式目录；训练跳过未完成的实验。相同输出与种子会拒绝覆盖。默认物体固定，适合受控挤压实验；设 `object.fixed=false` 可改为自由物体，但当前没有托举、释放时序和提起成功测试，自由物体可能在闭合前掉落。`bilateral_contact` 只表示双侧接触，`grasp_success` 保留为空。

## SSH 远程可视化：WebRTC

SSH 负责启动远端进程，官方 WebRTC livestream 负责传输界面。无需设置 `DISPLAY` 或使用 `ssh -X`。`--gui` 仍用于服务器本地窗口；Linux 无 `DISPLAY` 时会明确提示改用 `--livestream`。

在服务器上、仓库根目录执行，把 `SERVER_IP` 换成客户端能直达的服务器 IPv4 地址（局域网、VPN 或公网地址）：

```bash
python backends/isaac_sim/run.py \
  --livestream --livestream-ip SERVER_IP --episodes 3

# 查看官方夹爪资产也支持同一串流模式
python backends/isaac_sim/inspect_assets.py robotiq_2f85 \
  --livestream --livestream-ip SERVER_IP
```

程序使用安装包内的 `isaacsim.exp.full.streaming.kit`，以 `headless=True`、`hide_ui=False` 启动，启用完整界面和 WebRTC 服务。采集场景会增加独立的总览相机；两个训练相机的安装和采集流程保持原有定义。采集中按采集步距刷新界面；`--no-render` 只关闭训练相机数据，远程视口仍刷新。采集结束后持续渲染最后场景，不推进采集物理步，也不追加训练数据，直到在 SSH 终端按 `Ctrl+C` 退出。需要在采集过程中连接时，可增加 `--episodes`；默认也可以连接查看保留的最后场景。重复采集请更换 `--output` 或 `--seed`，避免覆盖已有数据。

客户端连接方式：

1. 桌面端：在本机安装官方 **Isaac Sim WebRTC Streaming Client**，服务器地址填写与 `--livestream-ip` 一致的地址，然后连接。
2. 浏览器：另行部署官方 Web viewer，并把它的 WebRTC 连接设置指向本项目进程的服务器地址和端口。当前脚本仅启动 Isaac Sim 串流服务，不自动启动 HTTP 页面。官方 Docker Compose viewer 可作为部署参考，但其默认配置会同时启动另一份 Isaac Sim，需调整为连接本项目进程。

服务器防火墙、云安全组以及 NAT 转发需允许 TCP `49100`（信令）和 UDP `47998`（画面），建议限定客户端来源。普通 SSH 的 TCP 端口转发无法代替 UDP 画面链路。服务器 GPU 需要支持 NVENC，且 NVIDIA 驱动可用；先用 `nvidia-smi` 确认。

可选端口参数为 `--livestream-signal-port 49200 --livestream-stream-port 48100`，客户端也必须使用相同端口。`--livestream-ip` 对应官方 `primaryStream/publicIp`，端口参数对应 `primaryStream/signalPort` 和 `primaryStream/streamPort`。不填写 IP 时使用官方自动检测，但远程/NAT 场景建议显式指定。JSON 配置也支持 `simulation.livestream`、`livestream_ip`、`livestream_signal_port`、`livestream_stream_port`。

日志中的 `app ready` 或本项目的初始化提示只代表启动阶段完成；请检查 `Started primary stream server ...` 的实际监听端口，最终以客户端成功显示界面为准。端口占用时官方服务可能选择其他端口，需查看日志或使用空闲端口重启。黑屏时检查 GPU 编码支持、双方地址一致、TCP/UDP 都可达，以及服务仍在运行。

官方说明：[Livestream Clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)、[Remote Workstation Deployment](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_advanced_remote_setup.html)。已核对 Isaac Sim 6.2 的 streaming 配置与 `SimulationApp` 参数，并在 ASUS RTX 4070 Laptop GPU → Mac 客户端上验证实际画面，具体配置见下。

### ASUS → Mac 的已验证连接

ASUS 的 Tailscale IPv4 是 `100.95.209.124`，Mac 与其 VPN 已互通。Mac 使用官方 Apple Silicon **Isaac Sim WebRTC Streaming Client 2.0.0**，安装于 `~/Applications/Isaac Sim WebRTC Streaming Client.app`。Server 填 `100.95.209.124`，Signal 填 `49100`，Stream 填 `47998`，点击 Connect。已实际确认 Mac 收到夹爪和测试物体画面。三轮均完成，每轮 53 帧；核对了节点坐标和力估计均有限、双 RGB 为 320×320、采样间隔 25 ms。种子 0、1、2 分别抽到球、球、方块，结束后保留最后的方块场景。此验证只覆盖运行与数据完整性，不代表接触物理已标定。

从 ASUS 的项目根目录运行以下命令；每次重跑更换输出目录或种子：

```bash
python backends/isaac_sim/run.py \
  --config configs/isaac_webrtc_preview.json \
  --livestream --livestream-ip 100.95.209.124 \
  --episodes 3 --output data/runs/isaac_webrtc_preview_new
```

预览配置单独将最终标称开口设为 40–45 mm、安装倾角设为 0–2°、时间步设为 0.5 ms、求解迭代设为 200，图像采集间隔仍为 25 ms。默认 5–25 mm 开口的大幅挤压在当前 PhysX 软体网格上出现非有限状态和 `SVD did not converge`；仅将时间步减小到 0.25 ms 仍不能解决。此预览配置用于查看场景，不代表原始大幅挤压已修复。

本次实机启动还修复了三个阻塞：启动 Kit 前完整导入环境中的 `charset_normalizer`，避免与 Kit 自带编译模块混用；用 `add_ground_plane()` 本地创建地面，避免依赖不可达的 NVIDIA 资产服务器；关闭 capture-on-play 时用 Replicator 显式零时间步采集，避免相机空帧。关闭时先打印活动异常，防止 `SimulationApp.close()` 直接终止进程而隐藏错误。

后台预览日志为 `data/logs/isaac_webrtc_preview.log`，输出为 `data/runs/isaac_webrtc_preview`；原始失败采集保留在 `data/runs/isaac_webrtc` 及 `data/logs/*incomplete`。原始源码备份保留在 `data/logs/*.before_visualization_fix`。串流中断后，可用客户端 Connection → Reset，重新填写服务器地址并连接。

## 力标签和学习

MuJoCo 只累计当前测试物体与各个软手指之间的原生接触力，排除地面、另一根手指等接触。每侧 `normal_force_n` 是该侧接触法向力幅值之和；`contact_force_world_n` 是物体施加在手指上的世界系合力，含摩擦。`squeeze_force_n` 定义为两侧法向幅值的平均值。不能用左右合力相加后接近零来表示“没有夹持力”。

Isaac Sim 的普通刚体接触传感器不能保证返回软体—刚体接触对的力。因此此后端的 `force_valid=false`、原生法向/合力标签为 NaN；单独的 `estimated_force_world_n` 是根据四面体形变进行的准静态、共旋线弹性平衡估计，补偿自由节点重力，忽略惯性和阻尼，也不能分离摩擦。它并非 PhysX 求解器的原生接触报告或已标定真值。

```bash
# 至少三个独立物体种子，默认仅学习有效原生标签
uv run --no-sync python scripts/train_force.py data/runs/mujoco

# 从已保存的双相机图像推理，不启动仿真器
uv run --no-sync python scripts/predict_force.py \
  --model data/models/force_ridge.npz \
  --frame data/runs/backends/mujoco_sim/episode_00000000_00000/frame_000052.npz

# 明确选择估计标签时，仅使用 Isaac 最后半段 hold 帧，投影到世界夹持 X 轴
# 不与原生法向幅值混为同一个训练目标
python scripts/train_force.py data/runs/isaacsim \
  --target-source estimated --output data/models/isaac_estimated_ridge.npz
```

NumPy ridge 是方便检查数据流程的基线，输入仅为双相机图像、张开参考图、相机内参、安装角和开口，没有读取仿真顶点或物体身份作为特征。训练/验证按物体采样种子划分，保证同一物体的相邻帧、焦距扫描和倾角扫描不会跨集合泄漏；标准化只使用训练集。当前只跑通了少量实验，不能据此宣称实物精度或跨物体泛化。后续可替换 `shared/learning.py` 为图像编码器，并增加材料、接触与形变的真实标定数据。

## Isaac 官方资产

官方库提供 Robotiq 2F-85/2F-140/Hand-E、Franka Panda 等机器人与夹爪，也有 YCB 日常物体。资产可能位于远程服务器或另外下载的资产包，安装 Python wheel 不等于全部模型已经下载。

```bash
python backends/isaac_sim/inspect_assets.py --list
python backends/isaac_sim/inspect_assets.py robotiq_2f85 --gui
python backends/isaac_sim/inspect_assets.py panda --export data/generated/panda_reference.usda
```

检查脚本加载真实 USD，列出刚体路径和关节连接，写入 `data/generated/isaac_asset_report.json`。它不会假定不同版本 USD 的指端路径相同。当前采集场景使用共享简化运动学；若要用完整夹爪关节驱动软指，下一步应利用报告中的真实指端连接点建立 attachment/底座位姿映射，并替换当前运动学边界条件，不能仅加载外观模型就声称已完成动力学集成。

`configs/isaac_ycb.json` 已接入香蕉、马克笔、罐头、食品盒等 USD 物体。也可以在 `object.isaac_usd_assets` 中填写 `/Isaac/...` 片段、完整 URI 或本地 USD 路径。物体按采样直径缩放并居中，碰撞采用每个 Mesh 的凸包近似，凹槽不会完整保留。MuJoCo 不直接读取 USD，应使用共有基本形状做两个后端的对照，或另行提供同几何的转换网格。

## 验证范围

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless SOFT_SIM_TEST_RENDER=1 \
  uv run --no-sync python -m unittest discover -s tests -v
```

测试覆盖真实三种物体接触/形变、左右接触力方向、改变镜头焦距不改变物理状态、双相机 RGB/depth 尺寸与内参、固定局部相机姿态、体网格单位、FEM 估计的刚性不变性/已知拉伸响应，以及基线模型保存/加载。MuJoCo 全流程已实跑。Isaac 仅做源码接口核对、配置/命令行和语法验证；当前沙箱不能访问 GPU，不能声明 GPU 软体仿真或远程 USD 资产已验证。

## 参考

- [DROID 硬件清单：Robotiq 2F-85](https://droid-dataset.github.io/droid/hardware-setup/shopping-list.html)
- [公开 Robotiq MJCF](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85)
- [公开 Panda 手爪 MJCF](https://github.com/google-deepmind/mujoco_menagerie/blob/main/franka_emika_panda/hand.xml)
- [拓竹 TPU 85A TDS](https://cdn.shopify.com/s/files/1/0645/5876/0155/files/Bambu_TPU_85A_Technical_Data_Sheet_a7233668-497b-487f-af8a-3adb49fafaa1.pdf)
- [MuJoCo flex 参考](https://mujoco.readthedocs.io/en/latest/XMLreference.html#deformable-flex)
- [Isaac 夹爪资产目录](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots_manipulator.html)
- [Isaac YCB 与可变形资产示例](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_isaac_snippets.html)
- [Isaac 软体/刚体接触力报告讨论](https://github.com/isaac-sim/IsaacSim/discussions/637)
