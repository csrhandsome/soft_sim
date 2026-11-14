# 输入资产

所有软指源网格和拓竹材料资料都放在 `assets/`。运行时缩放网格、场景、日志和录制仍放在
`artifacts/`，不覆盖输入资产。

## 材料

每种材料统一使用小写 `bambu_tpu_<硬度>` 名称，四个文件一一对应：

| 后缀 | 内容 |
| --- | --- |
| `.json` | 仿真实际加载的 SI 参数，包含物理来源、模量依据与配色来源 |
| `.tds.pdf` | 本地下载的拓竹原始技术数据表，Git 忽略 |
| `.tds.txt` | 从同名 PDF 提取的文本，便于搜索；本地派生文件，Git 忽略 |
| `.md` | 中文参数摘要、仿真假设和来源说明 |

| 预设 | 密度 kg/m³ | 仿真杨氏模量 MPa | 当前外观 |
| --- | ---: | ---: | --- |
| [bambu_tpu_75a](materials/bambu_tpu_75a.md) | 1170 | 7.084（估算） | 历史自定义橙色 |
| [bambu_tpu_85a](materials/bambu_tpu_85a.md) | 1180 | 6.8（TDS XY） | 自定义西瓜红 #D21B3C |
| [bambu_tpu_90a](materials/bambu_tpu_90a.md) | 1240 | 5.3（TDS XY） | 官方西瓜红/Blaze 深红端 #D21B3C |

名义 Shore 硬度与拉伸模量不是同一个指标。保留 TDS 原值，不按硬度排序重写模量；
75A 未公布拉伸模量，历史 Gent 估值不能作为三种材料刚度的实测对照。
三者泊松比、摩擦和阻尼均为仿真假设，详见各材料说明。

JSON 和 Markdown 是版本控制中的必要材料资料；PDF 和提取文本只在本地保留。
重新下载和生成文本：

```bash
python - <<'PY'
import json
from pathlib import Path
import subprocess
from urllib.request import urlretrieve
for path in Path('assets/materials').glob('bambu_tpu_*.json'):
    pdf = path.with_suffix('.tds.pdf')
    urlretrieve(json.loads(path.read_text())['source'], pdf)
    subprocess.run(['pdftotext', '-layout', str(pdf),
                    str(path.with_suffix('.tds.txt'))], check=True)
PY
```

```bash
# 默认 90A；CLI 明确指定的材料优先于任务 JSON
python scripts/run.py collect --config configs/general_pickup.json \
  --material bambu_tpu_85a --dataset-id x5_pickup85a_watermelon_v1 \
  --solver-iterations 80 --vertex-velocity-damping 20 --texture-max-resolution 512
```

任务 JSON 也可以写 `"material": {"preset": "bambu_tpu_85a"}`，并覆盖其中的个别字段。
每次运行会把完整生效参数写入 `config.resolved.json`，历史录制保留自己的材料快照。

软指默认使用不透明、非金属的哑光材质：`roughness=1.0`、
`specular_reflectivity=0.0`，抑制镜面高光、关闭清漆层，保留漫反射明暗。
这两个外观参数是视觉设定，不是 TDS 实测值；可在任务 JSON 的 `material` 中覆盖，
取值均为 `[0, 1]`，并随运行保存。外观设定不改变弹性、摩擦或阻尼。
修改材质后需重新 `collect`；`preview` 只拼接已录制的 RGB，不会重新渲染旧录制。

## 软指网格

`meshes/soft_finger.stl` 是共同源几何（STL 数值单位为 mm，读取时乘 0.001）；
`meshes/soft_finger_voxel2mm.npz` 是历史体素四面体网格。
75A、85A、90A 共用同一几何，不为不同材料复制 STL。
当前任务使用 `artifacts/cache/meshes/soft_finger_tetgen_4mm.npz` 的贴合四面体网格，
运行内再生成 0.75 倍缩放副本。
