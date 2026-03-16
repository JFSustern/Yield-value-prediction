# 原始代码和模型备份

**备份时间**: 2026-03-15
**备份原因**: 开始多保真度学习实验,隔离原始代码以便对比

## 备份内容

### 1. 源代码备份
- **路径**: `src_backup/`
- **内容**: 
  - `physics/yodel.py` - YODEL 物理引擎
  - `model/pinn.py` - 原始 PINN 模型
  - `model/baseline.py` - 基线模型
  - `model/mfnn.py` - MFNN 模型
  - `data/generator.py` - 原始数据生成器
  - `data/loader.py` - 数据加载器
  - `train.py` - 原始训练脚本
  - `test.py` - 原始测试脚本
  - 其他分析和训练脚本

### 2. 模型备份
- **路径**: `models_backup/`
- **内容**:
  - `yodel_pinn.pth` - 原始 PINN 模型 (136KB, 2026-03-03)
  - `yodel_pinn_dual.pth` - 双阶段 PINN (203KB, 2026-01-07)
  - `pure_nn.pth` - 纯 NN 基线 (201KB, 2025-12-18)
  - `yodel_mfnn.pth` - MFNN 模型 (157KB, 2026-03-05)
  - `mfnn_stage1.pth` - MFNN 阶段1 (157KB, 2026-03-05)

### 3. 原始数据备份
- **路径**: `data_synthetic_backup/`
- **参数范围**:
  - d50: 20-32 μm
  - Phi: 0.63-0.67
  - Tau0: 26-73 Pa
- **样本数**: 2305

## 新工作目录

多保真度学习的代码和模型将存放在 `multi_fidelity/` 目录:

```
multi_fidelity/
├── src/
│   ├── model/      # 新的模型定义
│   ├── data/       # 数据处理
│   └── training/   # 训练脚本
├── models/         # 新训练的模型
└── results/        # 训练结果和评估
```

## 对比说明

- **原始方法**: 高固含量 (Phi 0.63-0.67),目标 Tau0 70-120 Pa
- **新方法**: 多保真度学习,对齐论文数据 (Phi 0.35-0.52, Tau0 8-64 Pa)

## 恢复方法

如需恢复原始代码:
```bash
cp -r backup_original/src_backup/* src/
cp -r backup_original/models_backup/* models/
```

---

**注意**: 此备份仅用于对比和参考,请勿修改备份内容。
