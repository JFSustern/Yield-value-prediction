# 多保真度学习 (Multi-Fidelity Learning)

**创建时间**: 2026-03-15
**目标**: 实现低保真度 → 高保真度的两阶段训练策略

## 项目结构

```
multi_fidelity/
├── src/
│   ├── model/          # 模型定义
│   │   └── pinn.py     # PINN 模型 (从原始复制并修改)
│   ├── data/           # 数据处理
│   │   └── loader.py   # 数据加载器
│   └── training/       # 训练脚本
│       ├── train_low_fidelity.py   # 低保真度训练
│       └── train_high_fidelity.py  # 高保真度微调
├── models/             # 训练好的模型
│   ├── low_fidelity/   # 低保真度模型
│   └── high_fidelity/  # 高保真度模型
└── results/            # 训练结果和评估
    ├── plots/          # 可视化图表
    └── logs/           # 训练日志
```

## 数据说明

### 低保真度数据 (合成数据)
- **路径**: `../data/synthetic_aligned/`
- **样本数**: 1707 (训练: 2731, 测试: 683)
- **参数范围**:
  - d50: 7.00 - 18.68 μm
  - Phi: 0.454 - 0.520
  - Tau0: 8.01 - 63.72 Pa

### 高保真度数据 (论文 Table 5)
- **路径**: `../data/high_fidelity/cement_paste_data.csv`
- **样本数**: 36
- **参数范围**:
  - d50: 7.12 - 15.40 μm
  - Phi: 0.350 - 0.450
  - Tau0: 10.12 - 50.35 Pa

## 训练流程

### Phase 1: 低保真度训练
1. 使用对齐后的合成数据
2. 训练完整的 PINN 模型
3. 添加物理层 Loss 修正
4. 保存模型: `models/low_fidelity/pinn_low.pth`

### Phase 2: 高保真度微调
1. 加载低保真度模型
2. 冻结神经网络层
3. 仅微调物理层参数 (Φ_m, G_max)
4. 使用论文 Table 5 数据
5. 小学习率训练 (1e-5 ~ 1e-4)
6. 保存模型: `models/high_fidelity/pinn_high.pth`

## 与原始方法对比

| 特征 | 原始方法 | 多保真度方法 |
|------|---------|-------------|
| **数据范围** | Phi 0.63-0.67, Tau0 70-120 Pa | Phi 0.35-0.52, Tau0 8-64 Pa |
| **训练策略** | 单阶段训练 | 两阶段训练 + 微调 |
| **真实数据** | 无 | 使用论文数据微调 |
| **物理约束** | 基础约束 | 强化物理层 Loss |
| **模型路径** | `../models/` | `multi_fidelity/models/` |

## 快速开始

### 1. 训练低保真度模型
```bash
cd multi_fidelity
python src/training/train_low_fidelity.py
```

### 2. 高保真度微调
```bash
python src/training/train_high_fidelity.py
```

### 3. 评估对比
```bash
python src/training/evaluate_comparison.py
```

## 预期效果

- ✅ 低保真度模型: 在合成数据上 Loss < 20 Pa, R² > 0.8
- ✅ 高保真度模型: 在论文数据上泛化能力提升
- ✅ 验证 YODEL 物理机理的普适性

---

**注意**: 原始代码和模型已备份至 `../backup_original/`
