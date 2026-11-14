# 拓竹 TPU 75A

| 参数 | 值 | 依据 |
| --- | --- | --- |
| 名义硬度 | Shore A 75 | 商品型号 |
| 实测硬度 | Shore A 76 ± 1 | 原始 TDS V1.0 |
| 密度 | 1170 kg/m³ | TDS：1.17 g/cm³ |
| 仿真杨氏模量 | 约 7.084 MPa | Gent 1958，由名义硬度 75A 估算；非拓竹实测拉伸模量 |
| XY 拉伸强度 | 11.5 MPa | TDS：11.50 ± 0.50 MPa，仅记录 |
| XY 断裂伸长率下界 | 800% | TDS：>800%，仅记录 |
| 配色 | #FF6C2F | 历史自定义橙色，不作为拓竹官方色号 |

仿真使用各向同性材料近似，不包含断裂、打印方向或滞回。
泊松比 0.45、摩擦 0.7、弹性阻尼 0.0001 为原项目假设，需要实物标定。

[参数 JSON](bambu_tpu_75a.json)、[原始 PDF](bambu_tpu_75a.tds.pdf)、
[提取文本](bambu_tpu_75a.tds.txt)。
来源：[拓竹 TDS V1.0](https://store.bblcdn.eu/s8/default/6c98f7913de24007b54cb75052cd39b5/Bambu_TPU_75A-TDS.pdf)、
[Gent 模量估算论文](https://doi.org/10.5254/1.3542351)。
