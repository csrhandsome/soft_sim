# 拓竹 TPU 85A（自定义西瓜红外观）

2026-11-14 核对官方 TPU 85A Technical Data Sheet V1.0。

| 参数 | 值 | 依据 |
| --- | --- | --- |
| 名义硬度 | Shore A 85 | 商品型号；该 TDS 未列实测硬度 |
| 密度 | 1180 kg/m³ | TDS：1.18 g/cm³ |
| 仿真杨氏模量 | 6.8 MPa | TDS XY：6.8 ± 0.7 MPa，作为各向同性近似 |
| Z 方向杨氏模量 | 5.2 ± 0.6 MPa | TDS，仅记录打印方向差异 |
| XY 拉伸强度 | 12.0 MPa | TDS：12.0 ± 0.8 MPa，仅记录 |
| XY 断裂伸长率下界 | 700% | TDS：>700%，仅记录 |
| 配色 | sRGB #D21B3C | 用户指定沿用 90A 西瓜红深红端，非官方 85A 色号 |

当前 PhysX 材料为各向同性近似，不包含打印方向、非线性、滞回或断裂。
泊松比 0.45、摩擦 0.7、弹性阻尼 0.0001 为原项目假设，未冒充官方实测值。
TDS 的试样为 100% 填充、225°C 喷嘴、35°C 热床、34 mm/s，
试样在测试前于 70°C 干燥 12 小时。

[参数 JSON](bambu_tpu_85a.json)、[原始 PDF](bambu_tpu_85a.tds.pdf)、
[提取文本](bambu_tpu_85a.tds.txt)。
来源：[拓竹官方 TDS V1.0](https://cdn.shopify.com/s/files/1/0645/5876/0155/files/Bambu_TPU_85A_Technical_Data_Sheet_a7233668-497b-487f-af8a-3adb49fafaa1.pdf)、
[拓竹官方色表](https://eu.store.bambulab.com/products/tpu-85a-tpu-90a)。
