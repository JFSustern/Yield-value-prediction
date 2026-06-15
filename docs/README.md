# 数据与实验产物说明

项目数据只按研究体系分为 `data/lian2025/` 和 `data/zhou1999/`。每个体系内部再区分高保真与低保真数据。

## Lian 2025

```text
data/lian2025/
├── high_fidelity/
│   ├── all_400.csv
│   ├── table6.csv
│   └── splits/
│       ├── seed_0/
│       ├── seed_1/
│       ├── seed_2/
│       ├── seed_3/
│       └── seed_42/
└── low_fidelity/
    ├── dataset.csv
    ├── train.csv
    └── test.csv
```

| 文件 | 样本数 | 用途 |
|---|---:|---|
| `high_fidelity/table6.csv` | 16 | 整理后的论文 Table 6 数据 |
| `high_fidelity/all_400.csv` | 400 | 当前 Lian 主实验的高保真工作数据 |
| `high_fidelity/splits/seed_*/train.csv` | 30 | 小样本训练 |
| `high_fidelity/splits/seed_*/eval.csv` | 10 | 早停评估 |
| `high_fidelity/splits/seed_*/test.csv` | 360 | 最终测试 |
| `low_fidelity/train.csv` | 1600 | 低保真预训练 |
| `low_fidelity/test.csv` | 400 | 低保真评估 |

主要字段：

- `Phi`：固体体积分数。
- `SP_percent`：增塑剂掺量的百分比数值，例如 `0.50` 表示 0.50%。
- `Tau0_Pa`：屈服应力，单位 Pa。
- `phi_max`：低保真生成数据中的校验字段，不作为模型输入。

风险说明：`all_400.csv` 尚缺少独立的数据扩展脚本和完整来源记录。正式对外使用前，应补充从 16 条 Table 6 数据到 400 条工作数据的生成或采集过程。

## Zhou 1999

```text
data/zhou1999/
├── high_fidelity/
│   ├── dataset.csv
│   ├── train.csv
│   ├── eval.csv
│   └── test.csv
└── low_fidelity/
    ├── dataset.csv
    ├── train.csv
    └── test.csv
```

| 文件 | 样本数 | 用途 |
|---|---:|---|
| `high_fidelity/dataset.csv` | 50 | 高保真工作数据全集 |
| `high_fidelity/train.csv` | 30 | 高保真微调 |
| `high_fidelity/eval.csv` | 10 | 早停评估 |
| `high_fidelity/test.csv` | 10 | 最终测试 |
| `low_fidelity/train.csv` | 1600 | YODEL 低保真预训练 |
| `low_fidelity/test.csv` | 400 | 低保真评估 |

主要字段：

- `phi`：固体体积分数。
- `d_s_um`：表面平均粒径，单位 um。
- `tau_Pa`：屈服应力，单位 Pa。
- `powder`：AKP 粉末型号。
- `m1_true` / `m1_lf`：生成与校验辅助字段，不作为模型输入。

风险说明：当前高保真工作数据使用 Zhou 1999 图形视觉估算、YODEL 方程标定和模拟测量噪声构造。正式发表前应重新精确数字化，并保留原始点位、标定截图和复核记录。

## 模型权重

```text
multi_fidelity/models/lian2025/
├── low_fidelity.pth
├── hifi_only.pth
└── multifidelity.pth
```

- `low_fidelity.pth`：Lian 低保真预训练权重。
- `hifi_only.pth`：仅使用高保真数据训练的对照权重。
- `multifidelity.pth`：低保真预训练后使用高保真数据微调的权重。

训练脚本会直接覆盖这些文件。权重通过 `model_state_dict` 加载，模型类为 `LianPINN`。

## 结果目录

```text
multi_fidelity/results/
├── lian2025/
│   ├── multiseed_results.json
│   ├── logs/
│   └── plots/
└── zhou1999/
    ├── summary.json
    ├── baselines_summary.json
    └── *.png
```

结果文件是实验快照，不是自动化测试结果。代码、数据或训练配置变化后，应重新运行对应实验再引用。
