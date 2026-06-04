# PI-MFNN — Physics-Informed Multi-Fidelity Neural Network

> 论文：《基于物理信息多保真神经网络的浓悬浮液屈服应力预测：硬约束架构与跨体系通用性》

---

## 项目结构

```
Project/
├── data/
│   ├── high_fidelity/             # Lian 2025 体系：真实 HF 测量数据
│   ├── synthetic_table6_v2/       # Lian 2025 体系：LF 合成数据 (2000条)
│   ├── v3_split_seed42/           # Lian 2025 体系：HF train/eval/test 划分
│   ├── zhou1999_lf/               # Zhou 1999 体系：LF 合成数据 (2000条)
│   └── zhou1999_hf/               # Zhou 1999 体系：HF 数字化实验数据 (50条)
│
├── multi_fidelity/
│   ├── src/
│   │   ├── model/
│   │   │   ├── pinn_lian2025_v2.py   # Lian 2025 主模型（NPE→φ_max）
│   │   │   ├── pinn_lian2025_v3.py   # 可配置版（架构搜索用）
│   │   │   └── pinn_zhou1999_v1.py   # Zhou 1999 主模型（NPE→m₁_eff）
│   │   ├── physics/
│   │   │   └── lian2025.py           # Lian 2025 物理方程
│   │   ├── data/
│   │   │   ├── generator_lian2025.py # Lian 2025 LF 数据生成器
│   │   │   └── generator_zhou1999.py # Zhou 1999 LF+HF 数据生成器
│   │   └── training/
│   │       ├── train_v3.py                  # Lian 2025 体系主实验
│   │       └── run_zhou1999_experiment.py   # Zhou 1999 体系跨体系验证
│   │
│   ├── models/
│   │   ├── low_fidelity/lian_v3_low.pth           # LF 预训练
│   │   └── high_fidelity/
│   │       ├── lian_v3_hifi_only.pth              # 纯 HF 基线
│   │       └── lian_v3_multifidelity.pth          # PI-MFNN 最终模型 ★
│   │
│   └── results/zhou1999_exp/      # Zhou 1999 实验结果
│
└── docs/README.md
```

---

## 两个验证体系

| 体系 | 物理方程 | NPE 反演隐变量 | HF 数据 |
|------|---------|--------------|---------|
| **Lian 2025（水泥浆）** | `τ = m₁φ³/[φ_max(φ_max-φ)]` | φ_max（最大堆积分数） | 作者自有真实测量 |
| **Zhou 1999（Al₂O₃陶瓷）** | `τ = m₁_eff·φ(φ-φ₀)²/[φ_max(φ_max-φ)]` | m₁_eff（颗粒间力参数） | Zhou 1999 实验数字化 |

---

## 快速运行

```bash
# 环境：Python 3.9+, PyTorch 2.x
conda activate pytorch

# Lian 2025 主实验
cd /path/to/Project
python -m multi_fidelity.src.training.train_v3

# Zhou 1999 跨体系验证
python -m multi_fidelity.src.training.run_zhou1999_experiment \
    --lf-data data/zhou1999_lf \
    --hf-data data/zhou1999_hf \
    --out-dir multi_fidelity/results/zhou1999_exp

# 重新生成论文数据（需先运行 train_v3）
python /path/to/paper/latex/scripts/make_nature_figures.py
```

---

## 关键结果（seed=42，30条HF训练）

| 体系 | 纯HF MAPE | PI-MFNN MAPE | 多保真增益 |
|------|---------|-------------|---------|
| Lian 2025 | 15.9% | **10.2%** (R²=0.928) | −5.7 pp |
| Zhou 1999 | 17.6% | **10.9%** (R²=0.963) | −6.7 pp |

---

## 参考文献

- Lian et al. (2025). *Materials* 18, 2983. DOI: 10.3390/ma18132983
- Flatt & Bowen (2006). *J. Am. Ceram. Soc.* 89(4): 1244–1256. DOI: 10.1111/j.1551-2916.2005.00888.x
- Zhou et al. (1999). *J. Rheol.* 43(3): 651–71. DOI: 10.1122/1.551029
