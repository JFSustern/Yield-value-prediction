# PI-MFNN 屈服应力预测

本仓库实现了一个物理硬约束的多保真神经网络（PI-MFNN），用于浓悬浮液/水泥浆体屈服应力预测。当前研究包含两个验证体系：

| 体系 | 输入 | 网络反演量 | 物理约束层 |
|---|---|---|---|
| Lian 2025 水泥浆体 | `Phi`, `SP_percent` | 最大堆积分数 `phi_max` | `tau = m1 * phi^3 / [phi_max * (phi_max - phi)]` |
| Zhou 1999 Al2O3 悬浮液 | `phi`, `d_s_um` | 颗粒间力参数 `m1_eff` | 完整 YODEL 方程 |

> 当前仓库是研究实验代码，不是 Web 服务。未来 Agent 的需求草案见
> `docs/agent-requirements.md`，该功能尚未实现。

## 快速开始

建议使用 Python 3.9-3.12。训练依赖 PyTorch、NumPy、Pandas 和 Matplotlib。

```bash
cd /path/to/Project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

运行 Lian 2025 主实验（30 条训练、10 条评估、360 条测试）：

```bash
python -m multi_fidelity.src.training.train_v3
```

运行 Zhou 1999 四策略实验与多方法基线：

```bash
python -m multi_fidelity.src.training.run_zhou1999_experiment
python -m multi_fidelity.src.training.run_zhou1999_baselines
```

运行 Lian 2025 多随机种子实验：

```bash
python run_multiseed.py
```

训练命令会更新 `multi_fidelity/models/` 下的权重，并在
`multi_fidelity/results/` 下生成日志和图片。需要保留历史结果时，请在运行前另存输出目录或提交当前权重。

## 项目结构

```text
Project/
├── data/                           # 训练、评估和测试 CSV
│   ├── high_fidelity/              # Lian 体系原始/扩展数据及划分
│   ├── synthetic_table6_v2/        # Lian 体系低保真合成数据
│   ├── zhou1999_hf/                # Zhou 体系 HF 工作数据及划分
│   └── zhou1999_lf/                # Zhou 体系低保真合成数据
├── multi_fidelity/
│   ├── src/
│   │   ├── data/                   # 数据生成器
│   │   ├── model/                  # Lian/Zhou 物理硬约束模型
│   │   ├── physics/                # 独立物理公式
│   │   └── training/               # 训练、消融和基线实验
│   ├── models/                     # 当前参考模型权重
│   └── results/                    # 当前参考结果和运行生成物
├── docs/                           # 数据说明和后续需求草案
├── paper/                          # 论文参考材料，不参与代码运行
├── run_multiseed.py                # Lian 多种子鲁棒性实验
└── requirements.txt
```

## 常用实验

`train_v3` 使用一个位置参数选择实验：

```bash
python -m multi_fidelity.src.training.train_v3 main        # 默认主实验
python -m multi_fidelity.src.training.train_v3 arch        # 网络架构搜索
python -m multi_fidelity.src.training.train_v3 hparam      # 微调超参搜索
python -m multi_fidelity.src.training.train_v3 ablation    # 四策略消融
python -m multi_fidelity.src.training.train_v3 sufficient  # 320 条 HF 训练对比
```

重新生成低保真数据：

```bash
python -m multi_fidelity.src.data.generator_lian2025
python -m multi_fidelity.src.data.generator_zhou1999
```

以上生成器会覆盖对应目录中的 CSV，请先确认是否需要保留现有数据。

## 核心代码

- `multi_fidelity/src/model/pinn_lian2025_v2.py`：Lian 主模型，网络预测 `phi_max`，物理层计算 `tau0`。
- `multi_fidelity/src/model/pinn_lian2025_v3.py`：可配置宽度、深度和激活函数的架构搜索模型。
- `multi_fidelity/src/model/pinn_zhou1999_v1.py`：Zhou/YODEL 硬约束模型，网络预测 `m1_eff`。
- `multi_fidelity/src/training/train_v3.py`：Lian 训练、架构搜索、超参搜索、消融和充足数据实验。
- `multi_fidelity/src/training/run_zhou1999_experiment.py`：Zhou 四策略对比。
- `multi_fidelity/src/training/run_zhou1999_baselines.py`：Zhou 多方法基线对比。

## 数据与结果口径

详细字段、样本数和来源风险见 `docs/README.md`。交接时尤其注意：

1. `data/high_fidelity/hifi_table6.csv` 有 16 条整理后的 Table 6 记录。
2. `data/high_fidelity/hf_all400.csv` 是当前 Lian 实验使用的 400 条工作数据；交付前应补充其扩展/生成过程和审核记录。
3. `data/zhou1999_hf/` 当前数据由论文图形视觉估算、YODEL 标定和噪声模拟构成，不能直接表述为精确数字化的原始实验数据。
4. `multi_fidelity/results/multiseed_results.json` 是已有实验产物；随机种子逻辑已修正，正式引用前应重新运行 `run_multiseed.py`。

## 交接检查

- 从仓库根目录执行命令，避免相对路径写入错误位置。
- 确认 Python 和 PyTorch 版本后再重训；模型权重由 `state_dict` 加载。
- 不要把测试集用于早停或超参选择；当前脚本使用独立 `eval` 集早停。
- 正式发表前重新核验数据来源、随机种子结果和结果 JSON。
- Git 工作区可能包含新训练权重和未提交实验数据，提交前逐项确认，不要批量丢弃。

## 参考文献

- Lian et al. (2025), *Materials* 18, 2983. DOI: `10.3390/ma18132983`
- Flatt and Bowen (2006), *Journal of the American Ceramic Society* 89(4), 1244-1256. DOI: `10.1111/j.1551-2916.2005.00888.x`
- Zhou et al. (1999), *Journal of Rheology* 43(3), 651-671. DOI: `10.1122/1.551029`
