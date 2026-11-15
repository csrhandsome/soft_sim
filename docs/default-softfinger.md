# 默认软指：80 次迭代与固定区消元

2026-11-15，用户查看 80 次与 40 次完整抓取 RGB 后，选择加速后的 80 次方案作为默认。
原数值评估及筛选结果保留，不因默认选择而改写。

普通 `collect`、`replay`、`export-scene` 和 `assembly` 入口默认使用：

- 80 次位置求解迭代，1 ms 时间步。
- 节点速度阻尼 20，与本轮已查看的 80 次录制一致。
- 固定区单元消元；保留全部原节点、全部自由单元和完整碰撞网格。
- 固定区显示点随一个保留的全固定单元运动，保持原 STL 外形。

以现有 4 mm TetGen 网格为例，每指求解单元从 33,417 减为 15,948，碰撞网格仍为
33,417 个单元；8,491 个原节点和 3,367 个自由节点均保留。原完整网格作为输入，运行时
确定性生成消元后的求解拓扑，不依赖实验目录、预导出的消元网格或 `variant_worker`。
原源网格文件不被覆盖。实际单元数、消元状态和支持单元写入运行的 `mesh_settings`。

正常使用不需要新增参数：

```bash
python scripts/run.py collect --config configs/general_pickup.json --seed 22 \
  --material bambu_tpu_85a --dataset-id x5_pickup_v1
```

仅需重新准备原完整网格时，仍使用既有命令：

```bash
python scripts/run.py prepare-mesh --method tetgen --pitch .004
```

对照或排查时可关闭消元，迭代数仍为 80：

```bash
python scripts/run.py collect --config configs/general_pickup.json --seed 22 \
  --material bambu_tpu_85a --no-fixed-cell-elimination
```

JSON 中对应 `mesh.eliminate_fixed_cells: false`。显式 CLI 参数优先于配置文件；
`--solver-iterations` 仍可用于单独的迭代实验。GPU 位姿更新仍通过既有
`SOFT_SIM_FOLLOW_BACKEND=gpu` 选择，本次默认切换针对固定区消元与 80 次物理配置。

不要将历史实验导出的 reduced NPZ 当作普通 `--fem-mesh` 输入：该文件已经丢弃一部分
全固定单元，不能再充当完整碰撞网格。正式入口使用原完整 NPZ 自动派生求解网格。
实验 `variant_worker` 自行管理的预消元资产继续禁用这层自动消元，避免重复处理。

默认实现生成的节点、四面体顺序和支持单元与已查看的 80 次实验资产逐项一致。
105 项单元测试通过，覆盖弹性能量、自由单元保留、完整碰撞拓扑、固定表面刚体随动、
关闭开关、重复消元拒绝和默认入口配置。普通代码路径的 GPU 短测确认四指迭代数均为
80，碰撞网格未缩减，单元 Jacobian 为正。普通默认入口的完整抓取也已成功，提交了
268 帧、13.4 s 三视角录制；记录帧四指最小 Jacobian 为 0.80285，固定区最大误差约
0.002925 mm。规划 11 项、数据集 4 项回归测试同样通过；本次验证见
[运行证据](../artifacts/cache/benchmarks/default80_20261115/verification.json)。

历史完整对照测得：控制＋物理耗时减少约 28.7%～29.7%，含三相机采集耗时减少约
21.3%～22.2%。这些是此前使用 GPU 更新路径的测量，不能视作所有布局、材料和配置的
固定收益。本次接入不会消除既有小单元初始化告警；详见
[80 次评估](softfinger-80iter-evaluation.md)与 [40 次评估](softfinger-40iter-evaluation.md)。
