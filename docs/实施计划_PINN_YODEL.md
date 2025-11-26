# 实施计划：PINN + YODEL 技术路线

**版本**: 1.0
**日期**: 2025-11-24
**状态**: 待执行

本计划旨在重构项目，从错误的 KD 模型转向基于 YODEL 的 PINN 架构。由于缺乏实测物性数据，我们将采用“合成数据预训练 + 物理嵌入”的策略。

## 阶段 1: 基础架构与机理层 (Foundation & Physics)

**目标**: 建立项目骨架，实现可微分的 YODEL 物理引擎。

1.  **目录重构**:
    -   创建 `src/physics`, `src/data`, `src/model` 模块化结构。
    -   清理旧的 `main.py` 逻辑。
2.  **物理引擎 (`src/physics/yodel.py`)**:
    -   实现 YODEL 主方程: $\tau_0 = m_1 \frac{\Phi_m(\Phi_m-\Phi)}{\Phi(\Phi-\Phi_c)^2}$
    -   实现辅助方程: $\Phi_c(PSD)$, $m_1(G_{max}, PSD)$。
    -   实现动力学修正: $\Phi_m(E_{mix})$。
    -   **关键点**: 确保所有计算使用 PyTorch 张量操作，支持自动微分。

## 阶段 2: 数据工程 (Data Engineering)

**目标**: 解决数据缺失问题，构建“虚拟实验室”。

1.  **真实数据解析 (`src/data/loader.py`)**:
    -   读取 `data/` 下的 Excel 文件。
    -   计算每锅的混合功 $E_{mix} = \int (Tq \cdot \omega) dt$。
    -   提取温度 $T$ 的统计特征。
2.  **合成数据生成器 (`src/data/generator.py`)**:
    -   **PSD 模拟**: 基于 Log-Normal 分布生成 $d_{50}, \sigma$。
    -   **配方模拟**: 在 [0.60, 0.75] 范围内采样 $\Phi$。
    -   **标签生成**: 调用物理引擎生成理论 $\tau_0$ (Ground Truth)。
    - **混合数据集**: 将真实 $E_{mix}$ 分布与模拟物性参数结合。
    - **数据持久化**: 将生成的合成数据集保存为 CSV/Parquet 文件到 `data/synthetic/` 目录，便于审计和复用。

## 阶段 3: PINN 模型构建 (Model Architecture)

**目标**: 搭建嵌入式物理神经网络。

1.  **网络架构 (`src/model/pinn.py`)**:
    -   **输入层**: $[\Phi, E_{mix}, T, d_{50}, \sigma]$
    -   **隐藏层**: MLP (多层感知机) 用于特征提取。
    -   **物理参数头**: 输出中间物理量 $[\hat{\Phi}_m, \hat{G}_{max}]$。
    -   **物理层 (Differentiable Physics Layer)**: 硬编码 YODEL 方程，无训练参数，仅做前向计算。
2.  **损失函数**:
    -   $\mathcal{L}_{total} = \mathcal{L}_{MSE}(\tau_{pred}, \tau_{target}) + \lambda \mathcal{L}_{phy\_constraint}$
    -   约束: 确保 $\hat{\Phi}_m > \Phi$ 等物理限制。

## 阶段 4: 训练与验证 (Training & Validation)

**目标**: 验证模型能否学习物理规律。

1.  **训练脚本 (`src/train.py`)**:
    -   实现训练循环。
    -   集成 TensorBoard 监控 Loss 和物理参数演变。
2.  **评估**:
    -   验证模型在未见过的 $\Phi$ 和 $E_{mix}$ 组合下的预测能力。
    -   检查中间变量 $\hat{\Phi}_m$ 是否随 $E_{mix}$ 增加而增加（物理一致性检查）。

## 阶段 5: 集成与交付 (Integration)

**目标**: 提供统一入口。

1.  **主程序 (`main.py`)**:
    -   提供命令行接口 (CLI) 运行训练、生成数据或预测。
2.  **文档**:
    -   更新 README 和使用说明。

---

## 实施清单 (Checklist)

### Phase 1
- [ ] 创建目录结构 `src/{data,physics,model}`
- [ ] 实现 `src/physics/yodel.py` (YODEL 核心公式)

### Phase 2
- [ ] 实现 `src/data/loader.py` (Excel -> Emix)
- [ ] 实现 `src/data/generator.py` (PSD/Phi -> Synthetic Data)

### Phase 3
- [ ] 实现 `src/model/pinn.py` (NN + Physics Layer)

### Phase 4
- [ ] 实现 `src/train.py` (Training Loop)
- [ ] 编写 `main.py` 入口

### Phase 5
- [ ] 运行测试并生成最终报告

