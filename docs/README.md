# 数据与实验产物说明

本文档用于交接当前仓库的数据、训练产物和结果快照。项目数据只按研究体系分为 `data/lian2025/` 与 `data/zhou1999/`；每个体系内部再区分 `high_fidelity/` 和 `low_fidelity/`。

## Lian 2025 数据

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

| 文件 | 当前样本数 | 用途 |
|---|---:|---|
| `high_fidelity/table6.csv` | 16 | 论文 Table 6 整理数据；当前主实验固定测试集 |
| `high_fidelity/all_400.csv` | 400 | 当前 Lian 高保真工作数据，用于训练/评估划分 |
| `high_fidelity/splits/seed_42/train.csv` | 360 | 当前默认 seed 的训练缓存 |
| `high_fidelity/splits/seed_42/eval.csv` | 40 | 当前默认 seed 的评估缓存 |
| `high_fidelity/splits/seed_*/test.csv` | 360 | 历史划分遗留/参考文件；当前主训练不以它作为测试集 |
| `low_fidelity/dataset.csv` | 2000 | Lian 低保真生成数据全集 |
| `low_fidelity/train.csv` | 1600 | 低保真预训练 |
| `low_fidelity/test.csv` | 400 | 低保真评估 |

当前训练代码中的 Lian 主口径：

- `N_TRAIN = 360`
- `N_EVAL = 40`
- 高保真训练/评估来自 `high_fidelity/all_400.csv`
- 测试集由 `load_paper_test()` 单独读取 `high_fidelity/table6.csv`

`splits/seed_0`、`seed_1`、`seed_2`、`seed_3` 中仍保留过往 30/10/360 的稀疏划分文件；运行当前多种子脚本时，会按 360/40 重新写入对应 seed 的 `train.csv` 和 `eval.csv`。交接时请以代码入口和本节“当前主口径”为准。

主要字段：

| 字段 | 含义 |
|---|---|
| `Phi` | 固体体积分数 |
| `SP_percent` | 增塑剂掺量百分比数值，例如 `0.50` 表示 0.50% |
| `Tau0_Pa` | 屈服应力，单位 Pa |
| `phi_max` | 低保真数据中的物理参数/校验字段，不作为模型输入 |

风险说明：`all_400.csv` 是当前工作数据，但尚缺少独立的数据扩展脚本和完整来源记录。正式对外使用前，应补充从 16 条 Table 6 数据到 400 条工作数据的生成或采集过程。

## Zhou 1999 数据

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

| 文件 | 当前样本数 | 用途 |
|---|---:|---|
| `high_fidelity/dataset.csv` | 50 | Zhou 高保真工作数据全集 |
| `high_fidelity/train.csv` | 30 | 高保真微调 |
| `high_fidelity/eval.csv` | 10 | 早停/模型选择评估 |
| `high_fidelity/test.csv` | 10 | 最终测试 |
| `low_fidelity/dataset.csv` | 2000 | YODEL 低保真生成数据全集 |
| `low_fidelity/train.csv` | 1600 | 低保真预训练 |
| `low_fidelity/test.csv` | 400 | 低保真评估 |

主要字段：

| 字段 | 含义 |
|---|---|
| `phi` | 固体体积分数 |
| `d_s_um` | 表面平均粒径，单位 um |
| `tau_Pa` | 屈服应力，单位 Pa |
| `powder` | AKP 粉末型号 |
| `m1_true` / `m1_lf` | 生成与校验辅助字段，不作为模型输入 |

风险说明：当前高保真工作数据使用 Zhou 1999 图形视觉估算、YODEL 方程标定和模拟测量噪声构造。正式发表前应重新精确数字化，并保留原始点位、标定截图和复核记录。

## 训练入口

| 命令 | 产物 |
|---|---|
| `python -m multi_fidelity.src.training.train_lian2025` | Lian 主实验日志、模型权重、散点图/训练曲线 |
| `python -m multi_fidelity.src.training.train_lian2025 arch` | Lian 架构搜索结果 |
| `python -m multi_fidelity.src.training.train_lian2025 hparam` | Lian 超参数搜索结果 |
| `python -m multi_fidelity.src.training.train_lian2025 ablation` | Lian 冻结层数消融 |
| `python -m multi_fidelity.src.training.train_lian2025 sufficient` | Lian 训练样本量敏感性实验 |
| `python -m multi_fidelity.src.training.run_lian2025_baselines` | Lian 多方法基线 |
| `python run_lian2025_multiseed.py` | Lian 多随机种子鲁棒性实验 |
| `python -m multi_fidelity.src.training.run_zhou1999_experiment` | Zhou 四策略实验 |
| `python -m multi_fidelity.src.training.run_zhou1999_baselines` | Zhou 多方法基线 |

## 模型权重

```text
multi_fidelity/models/lian2025/
├── low_fidelity.pth
├── hifi_only.pth
└── multifidelity.pth
```

| 文件 | 含义 |
|---|---|
| `low_fidelity.pth` | Lian 低保真预训练权重 |
| `hifi_only.pth` | 仅使用高保真数据训练的对照权重 |
| `multifidelity.pth` | 低保真预训练后用高保真数据微调的权重 |

训练脚本会直接覆盖这些文件。权重通过 `model_state_dict` 加载，模型类为 `LianPINN`。

## 结果目录

```text
multi_fidelity/results/
├── lian2025/
│   ├── logs/
│   │   ├── train_lian2025_results.json
│   │   └── baseline_results_lian2025_new.json
│   ├── multiseed_results.json
│   └── plots/
└── zhou1999/
    ├── summary.json
    ├── baselines_summary.json
    └── *.png
```

| 文件 | 内容 |
|---|---|
| `lian2025/logs/train_lian2025_results.json` | Lian 低保真、PI-MFNN、HF-only 主实验结果 |
| `lian2025/logs/baseline_results_lian2025_new.json` | Lian MLP、软约束 PINN、PGNN、多保真基线等结果 |
| `lian2025/multiseed_results.json` | Lian 5 个随机种子的 HF-only 与 PI-MFNN 对比 |
| `lian2025/plots/` | Lian 主实验、消融、多种子图 |
| `zhou1999/summary.json` | Zhou 物理公式、LF-only、HF-only、PI-MFNN 四策略结果 |
| `zhou1999/baselines_summary.json` | Zhou 多方法基线结果 |
| `zhou1999/*.png` | Zhou 排名图和四策略散点图 |

当前快照中的关键结果：

| 体系 | 方法 | 测试集 | R2 | MAE | MAPE |
|---|---|---:|---:|---:|---:|
| Lian | PI-MFNN | 16 | 0.8785 | 0.1161 | 18.80 |
| Lian | HF-only | 16 | 0.7835 | 0.1462 | 22.02 |
| Zhou | PI-MFNN | 10 | 0.9628 | 67.02 | 10.92 |
| Zhou | HF-only | 10 | 0.8504 | 131.76 | 17.57 |

以上结果文件是实验快照，不是自动化测试结果。代码、数据或训练配置变化后，应重新运行对应实验再引用。

## 生成器覆盖范围

| 命令 | 会覆盖 |
|---|---|
| `python -m multi_fidelity.src.data.generator_lian2025` | `data/lian2025/low_fidelity/` |
| `python -m multi_fidelity.src.data.generator_zhou1999` | `data/zhou1999/low_fidelity/` 和 `data/zhou1999/high_fidelity/` |

运行生成器前，请确认是否需要备份当前 CSV。
