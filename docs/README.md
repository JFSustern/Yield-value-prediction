# 火箭推进剂屈服应力预测系统 (PINN-YODEL)

## 项目概述

本项目开发了一个基于**物理信息神经网络(PINN)**的火箭固体推进剂浆料**屈服应力(τ₀)**预测系统。核心创新是将**YODEL物理机理模型**嵌入深度学习框架,实现"数据驱动+物理约束"的混合建模。

### 核心特点

- **物理-AI混合建模**: 将YODEL颗粒网络理论与神经网络深度融合
- **工序模拟**: 基于真实混合工序的9阶段能量追踪
- **多时间点预测**: 同时预测48min(稀释前)、83min(高速混合后)、111min(固化后)三个关键时间点
- **高精度**: 训练Loss收敛至~16 Pa, R²>0.8
- **可解释性强**: 网络预测物理参数(Φₘ, G_max),而非黑盒输出

### 技术指标

| 指标 | 数值 |
|------|------|
| 代码总量 | 2023行Python |
| 训练Loss | ~16 Pa (Log-MSE) |
| 测试R² | >0.8 |
| 预测误差 | <2 Pa (验证集) |
| 数据规模 | ~2788样本(合成) + 15-20批次(真实) |

## 项目结构

```
Project/
├── src/                        # 源代码
│   ├── physics/                # 物理引擎
│   │   └── yodel.py           # YODEL机理实现 (125行)
│   ├── model/                  # 模型定义
│   │   ├── pinn.py            # PINN模型 (88行)
│   │   └── baseline.py        # 基线模型 (34行)
│   ├── data/                   # 数据模块
│   │   ├── generator.py       # 数据生成器 (240行)
│   │   └── loader.py          # 数据加载器 (68行)
│   ├── analysis/               # 分析工具 (831行)
│   │   ├── feature_analysis.py
│   │   ├── param_check.py
│   │   └── check_physics.py
│   ├── train.py               # PINN训练 (159行)
│   ├── test.py                # 测试评估 (201行)
│   ├── train_baseline.py      # 基线训练 (92行)
│   └── compare_models.py      # 模型对比 (105行)
├── data/                       # 数据目录
│   ├── 20251121处理后/        # 真实混合数据 (15-20个Excel)
│   ├── synthetic/             # 合成数据与结果
│   │   ├── dataset.csv        # 全量数据
│   │   ├── train_data.csv     # 训练集
│   │   ├── test_data.csv      # 测试集
│   │   └── plots/             # 可视化图表
│   ├── 工序.docx              # 工序流程文档
│   └── 数据生成策略.xlsx      # 生成策略参考
├── models/                     # 训练好的模型
│   ├── yodel_pinn.pth         # PINN模型权重
│   └── pure_nn.pth            # 基线模型权重
├── docs/                       # 文档
│   ├── 完整技术方案.md        # 统一技术方案
│   ├── 代码说明.md            # 代码结构说明
│   ├── 数据说明.md            # 数据详细说明
│   ├── 参数文档.md            # 参数详解(原始)
│   ├── 架构文档_PINN_YODEL_v2.md  # 架构设计(原始)
│   └── 新一代技术方案_PINN_YODEL.md  # 技术方案(原始)
├── main.py                     # CLI入口
└── 1202.3804v2.pdf            # YODEL论文

总代码量: 2023行Python代码
```

## 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n pinn python=3.9
conda activate pinn

# 安装依赖
pip install torch numpy pandas openpyxl matplotlib scikit-learn
```

### 2. 生成合成数据

```bash
python main.py generate --samples 15
```

生成15000个样本，过滤后约2700个有效样本，保存至`data/synthetic/`

### 3. 训练PINN模型

```bash
python main.py train
```

训练结果:
- 模型保存: `models/yodel_pinn.pth`
- 训练曲线: `data/synthetic/plots/training_history.png`

### 4. 测试评估

```bash
python main.py test
```

输出:
- 预测结果: `data/synthetic/test_results.csv`
- 评估图表: `data/synthetic/plots/test_evaluation.png`
- 三时间点对比: `data/synthetic/plots/dual_stage_eval.png`

### 5. 对比基线模型

```bash
# 训练纯NN基线
python src/train_baseline.py

# 对比PINN vs Pure NN
python src/compare_models.py
```

## 核心模块说明

### 1. 物理引擎 (`src/physics/yodel.py`)

实现YODEL屈服应力预测机理:

```python
# 核心公式
τ₀ = m₁ * [Φ(Φ - Φ_c)²] / [Φ_m(Φ_m - Φ)]
```

**关键函数**:
- `calc_phi_c(d50, sigma)`: 计算渗透阈值Φ_c (基于Log-Normal分布)
- `calc_m1(d50, G_max)`: 计算几何预因子m₁
- `calc_phi_m_dynamic(phi_m0, Emix)`: 混合功对最大堆积的动力学修正
- `yodel_mechanism(phi, phi_m, phi_c, m1)`: YODEL主方程

### 2. PINN模型 (`src/model/pinn.py`)

混合架构: **神经网络(黑盒) + 物理层(白盒)**

```
输入[5维]: Phi, d50, sigma, Emix, Temp
  ↓ (归一化)
隐藏层: 3层×128维 + Tanh激活
  ↓
预测: [Phi_m_raw, G_max_raw]
  ↓ (物理约束转换)
物理解码: Phi_m ∈ [Phi, 0.76], G_max > 0
  ↓
物理计算: m1 = calc_m1(d50, G_max)  # 强制计算!
          Phi_c = calc_phi_c(d50, sigma)
  ↓
YODEL机理: Tau0 = yodel_mechanism(...)
```

**关键设计**:
- m₁被**强制计算**而非预测,确保 m₁ ∝ d50⁻² 的物理规律
- Sigmoid约束Φ_m在合理范围
- G_max基准值 = 80000

### 3. 数据生成器 (`src/data/generator.py`)

基于工序模拟的合成数据生成:

**9个工序阶段**:
1. Initial Mix (10 min)
2. Add Oxidizer 1-3 (30 min)
3. Mix End (8 min) → **T1检查点 (48 min, Phi=Phi_1)**
4. High Speed Mix (35 min) → **T_new检查点 (83 min, Phi=Phi_1, 固化剂未加)**
5. Add Curing Agent (3 min)
6. Final Mix (25 min) → **T2检查点 (111 min, Phi=Phi_2)**
7. Resting (15 min)

**参数范围**:
- d50 ∈ [20, 32] μm
- sigma ∈ [1.4, 1.9]
- Phi_2 ∈ [0.63, 0.67]
- ratio_curing ∈ [0.04, 0.06] (4-6%)

**物理过滤**:
- Tau0_1 ∈ [70, 120] Pa
- Tau0_2 < 200 Pa
- Phi < Phi_m (无堵塞)

## 技术创新点

### 1. PINN混合架构

不同于标准PINN,采用:
- NN预测**隐变量**(Φ_m, G_max)
- 物理层**确定性计算**(m₁, Φ_c, τ₀)
- 保证物理规律被严格遵守

### 2. YODEL机理

基于颗粒网络理论:
- **Φ_c**: 渗透阈值,颗粒网络形成的临界固含量
- **Φ_m**: 最大堆积,含混合功动力学修正
- **m₁**: 几何预因子,与颗粒间作用力相关

### 3. 工序模拟

真实混合过程的数字孪生:
- 9阶段能量追踪
- 3个关键时间点预测
- 固化剂效应建模

### 4. Log-MSE Loss

适配长尾分布(0~3500 Pa):
```python
loss = mean((log(1+pred) - log(1+true))²)
```

## 项目演进历程

### 第1阶段: 基础架构 (2025-11)
- KD模型 → YODEL模型
- 项目重构,模块化设计

### 第2阶段: 优化突破 (2025-12)
- **关键优化**: MSE → Log-MSE
- 解决Loss居高不下问题
- 双阶段屈服预测

### 第3阶段: 参数调整 (2026-01)
- 数据自定义生成策略(摆脱基准Excel)
- 固化剂比例: 1.5-2.5% → 4-6%
- Tau0约束: (50-150) → (70-120) Pa
- 时间点扩展: 2个 → 3个

## 实验结果

### 训练效果
- Loss收敛至 ~16 Pa
- 预测的Φ_m ≈ 0.72-0.74 (合理)
- 预测的G_max ≈ 2.2×10⁵ (~220 nN, 合理)

### 测试表现
- R² > 0.8
- 验证集误差 < 2 Pa
- 物理参数预测符合理论范围

### PINN vs 纯NN
- PINN在小样本下泛化性更好
- PINN预测趋势符合物理直觉
- 纯NN容易过拟合

## 数据统计

| 特征 | 最小值 | 最大值 | 均值 | 标准差 |
|------|-------|-------|------|--------|
| Phi(固含量) | 0.600 | 0.740 | 0.671 | 0.040 |
| d50(μm) | 5.02 | 49.96 | 27.82 | 13.12 |
| sigma | 1.20 | 1.99 | 1.61 | 0.23 |
| Emix(J) | 3.4e8 | 4.5e8 | 4.1e8 | 3.2e7 |
| Tau0(Pa) | 0.01 | 3508.07 | 104.22 | 276.81 |

**关键相关性**:
- sigma ↔ Phi_c: 0.9998 (几何标准差完全决定渗透阈值)
- m1 ↔ Tau0: 0.5988 (强度因子是屈服值的关键驱动)
- d50 ↔ m1: -0.6897 (粒径越大,m1越小)

## 文档体系

1. **README.md** (本文件): 项目总览和快速开始
2. **完整技术方案.md**: 整合的技术方案文档
3. **代码说明.md**: 详细的代码结构和使用说明
4. **数据说明.md**: 数据格式和生成策略
5. **参数文档.md**: 原始参数详解
6. **架构文档_PINN_YODEL_v2.md**: 原始架构设计
7. **新一代技术方案_PINN_YODEL.md**: 原始技术方案

## 后续优化方向

1. **参数精细化**: 持续调整Tau0目标范围和固化剂比例
2. **多时间点扩展**: 从3个扩展到4-5个关键点
3. **工序更新**: 适配新的混合工序表
4. **异常检测**: 更完善的物理约束过滤
5. **泛化能力**: 扩大PSD参数范围

## 参考文献

- Flatt & Bowen (2006): "Yodel: A Yield Stress Model for Suspensions"
- Journal of the American Ceramic Society, Vol. 89, No. 4, pp. 1244-1256
- DOI: 10.1111/j.1551-2916.2005.00888.x

---

**版本**: 1.0
**最后更新**: 2026-02-08
