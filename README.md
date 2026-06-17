# PI-MFNN 屈服应力预测

本项目实现物理约束多保真神经网络（PI-MFNN），用于两个浆体/悬浮液体系的屈服应力预测与对比实验。

| 体系 | 任务 | 模型反演量 | 物理层 |
|---|---|---|---|
| Lian 2025 水泥浆体 | 由 `Phi`、`SP_percent` 预测 `Tau0_Pa` | 最大堆积分数 `phi_max` | Lian 屈服应力方程 |
| Zhou 1999 Al2O3 悬浮液 | 由 `phi`、`d_s_um` 预测 `tau_Pa` | 颗粒间力参数 `m1_eff` | YODEL 方程 |

代码当前重点是复现实验流程、保留数据划分口径、生成结果快照。正式对外使用前，请优先复核 `docs/README.md` 中的数据来源说明。

## 环境准备

建议使用 Python 3.9-3.12。

```bash
cd /path/to/Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

依赖集中在 `requirements.txt`：PyTorch、NumPy、Pandas、Matplotlib。

## 项目结构

```text
Project/
├── data/
│   ├── lian2025/
│   │   ├── high_fidelity/          # Lian 高保真工作数据与划分
│   │   └── low_fidelity/           # Lian 低保真生成数据
│   └── zhou1999/
│       ├── high_fidelity/          # Zhou 高保真工作数据与划分
│       └── low_fidelity/           # Zhou 低保真 YODEL 生成数据
├── docs/
│   └── README.md                   # 数据、结果、复核风险说明
├── multi_fidelity/
│   ├── src/
│   │   ├── data/                   # 数据生成脚本
│   │   ├── model/                  # PI-MFNN / PINN 模型
│   │   ├── physics/                # 独立物理公式
│   │   └── training/               # 主实验、基线、消融入口
│   ├── models/lian2025/            # Lian 训练权重，运行训练会覆盖
│   └── results/
│       ├── lian2025/               # Lian 日志与图
│       └── zhou1999/               # Zhou 日志与图
├── paper/                          # 参考论文
├── run_lian2025_multiseed.py       # Lian 多随机种子实验
└── requirements.txt
```

## 数据口径

数据已经按体系收拢到 `data/lian2025/` 和 `data/zhou1999/`。

- Lian 主实验使用 `data/lian2025/high_fidelity/all_400.csv` 随机划分 360 条训练、40 条评估；固定测试集为 `data/lian2025/high_fidelity/table6.csv` 的 16 条论文表格数据。
- Lian 的 `splits/seed_*/` 是缓存划分。当前训练入口会重新生成/覆盖对应 seed 的 `train.csv` 与 `eval.csv`；测试集仍以 `table6.csv` 为准。
- Zhou 使用 `data/zhou1999/high_fidelity/train.csv`、`eval.csv`、`test.csv` 的 30/10/10 划分。
- 低保真数据分别位于两个体系的 `low_fidelity/`，每个体系当前为 1600 条训练、400 条测试。

更详细的字段、样本数、结果文件与风险说明见 `docs/README.md`。

## 运行实验

从仓库根目录执行命令，避免相对路径写错。

Lian 2025 主实验：

```bash
python -m multi_fidelity.src.training.train_lian2025
```

Lian 2025 其他模式：

```bash
python -m multi_fidelity.src.training.train_lian2025 main
python -m multi_fidelity.src.training.train_lian2025 arch
python -m multi_fidelity.src.training.train_lian2025 hparam
python -m multi_fidelity.src.training.train_lian2025 ablation
python -m multi_fidelity.src.training.train_lian2025 sufficient
```

Lian 2025 多方法基线：

```bash
python -m multi_fidelity.src.training.run_lian2025_baselines
```

Lian 2025 多随机种子：

```bash
python run_lian2025_multiseed.py
```

Zhou 1999 四策略实验与多方法基线：

```bash
python -m multi_fidelity.src.training.run_zhou1999_experiment
python -m multi_fidelity.src.training.run_zhou1999_baselines
```

重新生成低保真数据：

```bash
python -m multi_fidelity.src.data.generator_lian2025
python -m multi_fidelity.src.data.generator_zhou1999
```

注意：生成器会覆盖对应体系的 `low_fidelity/` 数据；Zhou 生成器还会覆盖 `data/zhou1999/high_fidelity/`。运行前请确认是否需要保留当前 CSV。

## 核心入口

| 文件 | 作用 |
|---|---|
| `multi_fidelity/src/model/pinn_lian2025.py` | Lian 物理硬约束模型 |
| `multi_fidelity/src/model/pinn_lian2025_configurable.py` | Lian 架构搜索模型 |
| `multi_fidelity/src/model/pinn_zhou1999.py` | Zhou/YODEL 物理硬约束模型 |
| `multi_fidelity/src/training/train_lian2025.py` | Lian 命令行入口 |
| `multi_fidelity/src/training/lian2025_experiments.py` | Lian 主实验、搜索、消融实现 |
| `multi_fidelity/src/training/run_lian2025_baselines.py` | Lian 多方法基线 |
| `run_lian2025_multiseed.py` | Lian 多随机种子鲁棒性实验 |
| `multi_fidelity/src/training/run_zhou1999_experiment.py` | Zhou 四策略实验 |
| `multi_fidelity/src/training/run_zhou1999_baselines.py` | Zhou 多方法基线 |

## 当前结果快照

已有结果保存在 `multi_fidelity/results/`，它们是当前代码与当前数据下的实验快照，不等同于自动化测试结果。

| 体系 | 文件 | 关键口径 |
|---|---|---|
| Lian 主实验 | `multi_fidelity/results/lian2025/logs/train_lian2025_results.json` | PI-MFNN 在 16 条 Table 6 测试集上 `R2=0.8785`，HF-only 为 `R2=0.7835` |
| Lian 基线 | `multi_fidelity/results/lian2025/logs/baseline_results_lian2025_new.json` | 记录 MLP、软约束 PINN、残差 MFNN、Meng MFNN、混合训练等对比 |
| Lian 多种子 | `multi_fidelity/results/lian2025/multiseed_results.json` | 5 个 seed 的 HF-only 与 PI-MFNN 对比 |
| Zhou 四策略 | `multi_fidelity/results/zhou1999/summary.json` | PI-MFNN 测试 `R2=0.9628`，HF-only 为 `R2=0.8504` |
| Zhou 基线 | `multi_fidelity/results/zhou1999/baselines_summary.json` | 记录 9 类无约束、软约束、硬约束和多保真方法 |

## 交接注意事项

1. `data/lian2025/high_fidelity/all_400.csv` 是当前高保真工作数据，但从 16 条 Table 6 到 400 条数据的扩展过程仍需补充来源记录。
2. `data/zhou1999/high_fidelity/` 是基于 Zhou 1999 图形视觉估算、YODEL 标定和噪声模拟得到的工作数据，不应表述为精确数字化原始实验点。
3. Lian 训练会覆盖 `multi_fidelity/models/lian2025/` 中的权重文件。
4. 结果图和日志会写入 `multi_fidelity/results/<体系>/`；引用结果前建议重新运行对应实验并记录随机种子。
5. 仓库中如出现 `__pycache__/`，它只是 Python 缓存，不参与实验逻辑。

## 参考文献

- Lian et al. (2025), *Materials* 18, 2983. DOI: `10.3390/ma18132983`
- Flatt and Bowen (2006), *Journal of the American Ceramic Society* 89(4), 1244-1256. DOI: `10.1111/j.1551-2916.2005.00888.x`
- Zhou et al. (1999), *Journal of Rheology* 43(3), 651-671. DOI: `10.1122/1.551029`
