# 新一代技术方案：基于 PINN 与 YODEL 的屈服应力预测

**版本**: 1.0
**日期**: 2025-11-24
**作者**: CatPaw (AI Assistant)

## 1. 背景与目标

当前项目缺乏关键的物性数据（PSD, $\Phi$, $\tau_0$），仅有过程数据。为了实现对火箭燃料屈服应力的精准预测，本方案旨在：
1.  利用物理常识和统计规律**补全缺失数据**。
2.  构建 **PINN (Physics-Informed Neural Network)** 架构，融合 **YODEL 机理**。
3.  实现从“过程参数”到“最终性能”的端到端预测。

## 2. 核心机理模型 (The Physics)

我们将采用 **YODEL (Yield Stress Model)** 作为核心物理骨架，并辅以动力学修正。

### 2.1 YODEL 主方程
$$ \tau_0 = m_1 \frac{\Phi_m(\Phi_m-\Phi)}{\Phi(\Phi-\Phi_c)^2} $$

其中：
- $\Phi$: 固含量 (Volume Fraction)
- $\Phi_m$: 最大堆积密度 (Maximum Packing Fraction)
- $\Phi_c$: 渗透阈值 (Percolation Threshold)
- $m_1$: 几何预因子，与颗粒间作用力 $G_{max}$ 和粒径分布 $R_{v,50}$ 有关。

### 2.2 过程-结构关联 (Process-Structure Link)
混合功 $E_{mix}$ 通过打散团聚体提高最大堆积密度 $\Phi_m$：
$$ \Phi_m(E_{mix}) = \Phi_{m,min} + (\Phi_{m,max} - \Phi_{m,min}) \cdot (1 - e^{-k_E E_{mix}}) $$

### 2.3 热动力学修正 (Thermodynamics)
温度 $T$ 影响颗粒间作用力或早期固化：
$$ G_{max}(T) = G_0 \cdot \exp\left(\frac{E_a}{R}(\frac{1}{T_0} - \frac{1}{T})\right) $$

## 3. 数据生成与补全策略 (Data Augmentation)

由于缺乏实测数据，构建“虚拟实验室”生成训练集。

| 参数 | 来源/生成方法 | 分布假设 |
| :--- | :--- | :--- |
| **混合功 $E_{mix}$** | **真实数据提取** | 从 `data/*.xlsx` 计算 $\int (Tq \cdot \omega) dt$ |
| **温度 $T$** | **真实数据提取** | 从 `data/*.xlsx` 提取均值/终值 |
| **固含量 $\Phi$** | **随机生成** | Uniform(0.60, 0.75) (典型高固含范围) |
| **粒径 $d_{50}$** | **随机生成** | Log-Normal($\mu_{geo}, \sigma_{geo}$) |
| **粒径宽度 $\sigma$** | **随机生成** | Uniform(1.2, 2.0) |
| **标签 $\tau_0$** | **机理计算** | 代入上述参数至 YODEL + 修正模型 |

## 4. PINN 网络架构 (Physics-Integrated)

采用 **嵌入式物理 (Hard-constrained PINN)** 架构，确保输出始终符合物理定律。

```mermaid
graph LR
    subgraph Inputs
    I1[Phi]
    I2[PSD (d50, sigma)]
    I3[Emix]
    I4[T]
    end

    subgraph "Neural Network (Parameter Estimator)"
    H1[Hidden Layers]
    O1[Pred: Phi_m_eff]
    O2[Pred: G_max_eff]
    end

    subgraph "Differentiable Physics Layer"
    P1[YODEL Equation]
    end

    subgraph Output
    Y[Tau_0]
    end

    I1 --> H1
    I2 --> H1
    I3 --> H1
    I4 --> H1

    H1 --> O1
    H1 --> O2

    I1 --> P1
    I2 --> P1
    O1 --> P1
    O2 --> P1

    P1 --> Y
```

### 4.1 优势
- **可解释性**: 网络预测的是物理参数 ($\Phi_m, G_{max}$)，而非黑盒输出。
- **泛化性**: 物理层保证了在训练数据之外的区域（如极端 $\Phi$）也能保持趋势正确。
- **数据效率**: 只需要少量数据即可校准 NN 中的参数映射关系。

## 5. 实施路线

1.  **数据工程**:
    -   编写 `src/data_loader.py`: 读取 Excel，计算 $E_{mix}$。
    -   编写 `src/generator.py`: 实现 PSD 生成和 YODEL 标签生成。
2.  **模型构建**:
    -   编写 `src/model.py`: 实现 PINN 架构 (PyTorch)。
    -   实现自定义 `YODEL_Layer` (继承 `nn.Module`)。
3.  **训练与验证**:
    -   预训练：使用纯机理数据训练 NN 学习 $E_{mix} \to \Phi_m$ 的映射。
    -   评估：检查模型在不同 $\Phi$ 和 $E_{mix}$ 下的预测趋势是否符合物理直觉。

