# 拓竹 TPU 90A 西瓜红（51901 / Blaze）

2026-11-14 核对官方 TPU 90A Technical Data Sheet V1.0 与商品页。

| 仿真参数 | 当前值 | 依据 |
| --- | --- | --- |
| 名义硬度 | Shore A 90 | 商品型号；V1.0 TDS 未列实测硬度 |
| 密度 | 1240 kg/m³ | TDS：1.24 g/cm³ |
| 杨氏模量 | 5.3 MPa | TDS XY：5.3 ± 0.7 MPa，作为各向同性近似 |
| XY 拉伸强度 | 12.5 MPa | TDS：12.5 ± 0.8 MPa，仅记录，非失效模型 |
| XY 断裂伸长率下界 | 650% | TDS：>650%，仅记录 |
| 配色 | sRGB #D21B3C | 官方 Blaze 深红端；浅粉端为 #F1AAA8 |

官方西瓜红为整卷一次渐变。小尺寸软指采用深红端均匀配色；不人为在每根软指上复现整卷渐变。
显示色先从 sRGB 转成线性 RGB，再传给 USD displayColor。

[参数 JSON](bambu_tpu_90a.json)、[原始 PDF](bambu_tpu_90a.tds.pdf)、
[提取文本](bambu_tpu_90a.tds.txt)。

当前 PhysX 材料为各向同性近似。TDS Z 向杨氏模量为 4.4 ± 0.6 MPa，
没有据此实现打印方向、非线性、滞回或断裂；泊松比 0.45、摩擦 0.7、
弹性阻尼 0.0001 为原仿真假设，未冒充官方实测值。
V1.0 TDS 的试样为 100% 填充、225°C 喷嘴、35°C 热床、34 mm/s，
试样在测试前于 70°C 干燥 12 小时。

来源：

- [拓竹官方 TPU 90A TDS V1.0](https://store.bblcdn.com/s6/default/62f87cb83a1e47a58c73c73240f8410a/Bambu_TPU_90A_Technical_Data_Sheet_582bf8f6-1f0a-474c-aeda-9e72af3689dc.pdf)
- [拓竹官方商品页及 HEX 色表](https://eu.store.bambulab.com/products/tpu-85a-tpu-90a)

复现本次三视角视频（运行结束后使用命令打印的 run_id）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
python scripts/run.py collect --config configs/general_pickup.json --seed 22 \
  --dataset-id x5_pickup90a_watermelon_v1 --solver-iterations 80 \
  --vertex-velocity-damping 20 --texture-max-resolution 512
python scripts/run.py preview --run-id <run_id> --video tpu90a_watermelon_red.mp4
```
