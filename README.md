# PI-MFNN 屈服应力预测

本仓库实现物理硬约束多保真神经网络（PI-MFNN），用于 Lian 2025 水泥浆体与 Zhou 1999 Al2O3 悬浮液的屈服应力预测。

| 体系 | 输入 | 网络反演量 | 物理层 |
|---|---|---|---|
| Lian 2025 | `Phi`, `SP_percent` | 最大堆积分数 `phi_max` | Lian 屈服应力方程 |
| Zhou 1999 | `phi`, `d_s_um` | 颗粒间力参数 `m1_eff` | 完整 YODEL 方程 |

## 环境

建议使用 Python 3.9-3.12。

```bash
cd /path/to/Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本机已有 Conda 环境时也可直接使用：

```bash
conda activate pytorch
```

## 项目结构

```text
Project/
├── data/
│   ├── lian2025/
│   │   ├── high_fidelity/
│   │   │   ├── all_400.csv
│   │   │   ├── table6.csv
│   │   │   └── splits/seed_*/
│   │   └── low_fidelity/
│   │       ├── dataset.csv
│   │       ├── train.csv
│   │       └── test.csv
│   └── zhou1999/
│       ├── high_fidelity/
│       │   ├── dataset.csv
│       │   ├── train.csv
│       │   ├── eval.csv
│       │   └── test.csv
│       └── low_fidelity/
│           ├── dataset.csv
│           ├── train.csv
│           └── test.csv
├── multi_fidelity/
│   ├── src/
│   │   ├── data/                   # 两个体系的数据生成器
│   │   ├── model/                  # 物理硬约束模型
│   │   ├── physics/                # 独立物理公式
│   │   └── training/               # 训练与对比实验
│   ├── models/lian2025/            # Lian 参考权重
│   └── results/
│       ├── lian2025/
│       └── zhou1999/
├── docs/README.md                  # 数据与产物口径
├── paper/                          # 参考论文
├── run_lian2025_multiseed.py       # Lian 多随机种子实验
└── requirements.txt
```

## 运行实验

Lian 2025 主实验：

```bash
python -m multi_fidelity.src.training.train_lian2025
```

Lian 训练脚本还提供以下模式：

```bash
python -m multi_fidelity.src.training.train_lian2025 main
python -m multi_fidelity.src.training.train_lian2025 arch
python -m multi_fidelity.src.training.train_lian2025 hparam
python -m multi_fidelity.src.training.train_lian2025 ablation
python -m multi_fidelity.src.training.train_lian2025 sufficient
```

Zhou 1999 四策略实验与多方法基线：

```bash
python -m multi_fidelity.src.training.run_zhou1999_experiment
python -m multi_fidelity.src.training.run_zhou1999_baselines
```

Lian 多随机种子实验：

```bash
python run_lian2025_multiseed.py
```

## 重新生成数据

```bash
python -m multi_fidelity.src.data.generator_lian2025
python -m multi_fidelity.src.data.generator_zhou1999
```

生成器会覆盖对应体系的 `low_fidelity/` 数据；Zhou 生成器还会覆盖其 `high_fidelity/` 数据。运行前应确认是否需要保留当前 CSV。

## 核心代码

- `multi_fidelity/src/model/pinn_lian2025.py`：Lian 基础模型。
- `multi_fidelity/src/model/pinn_lian2025_configurable.py`：架构搜索使用的可配置 Lian 模型。
- `multi_fidelity/src/model/pinn_zhou1999.py`：Zhou/YODEL 模型。
- `multi_fidelity/src/training/train_lian2025.py`：Lian 主实验、搜索和消融。
- `multi_fidelity/src/training/run_zhou1999_experiment.py`：Zhou 四策略实验。
- `multi_fidelity/src/training/run_zhou1999_baselines.py`：Zhou 多方法基线。

## 交接注意事项

1. `data/lian2025/high_fidelity/all_400.csv` 是当前主实验使用的工作数据，其从 Table 6 数据扩展到 400 条的过程仍需补充来源记录。
2. `data/zhou1999/high_fidelity/` 是基于论文图形视觉估算、YODEL 标定和噪声模拟得到的工作数据，不应表述为精确数字化的原始实验数据。
3. 训练会覆盖 `multi_fidelity/models/lian2025/` 中的参考权重。
4. 训练日志和图片写入 `multi_fidelity/results/<体系>/`；正式引用结果前应重新核验代码、数据和随机种子。
5. 从仓库根目录执行命令，避免相对路径产生额外目录。

## 参考文献

- Lian et al. (2025), *Materials* 18, 2983. DOI: `10.3390/ma18132983`
- Flatt and Bowen (2006), *Journal of the American Ceramic Society* 89(4), 1244-1256. DOI: `10.1111/j.1551-2916.2005.00888.x`
- Zhou et al. (1999), *Journal of Rheology* 43(3), 651-671. DOI: `10.1122/1.551029`
