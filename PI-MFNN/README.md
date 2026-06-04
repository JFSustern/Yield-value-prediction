# PI-MFNN v3 模型发布包

**项目**: 物理信息多保真神经网络（PI-MFNN）预测高固含量悬浮液屈服应力  
**版本**: v3（最终版）  
**发布日期**: 2026-06-02  
**最优结果**: 测试集 R²=**0.9321**，MAE=**0.0496 Pa**，MAPE=**10.1%**

---

## 任务描述

预测水泥浆等高固含量悬浮液的屈服应力 τ₀（单位：Pa）。

**输入**: [Φ（体积固含量），SP%（超塑化剂掺量百分比）]  
**输出**: τ₀（屈服应力，Pa）

物理公式（Lian 2025，硬约束嵌入模型）：
```
τ₀ = m₁ · φ³ / [φ_max(SP) · (φ_max(SP) − φ)]
m₁ = 0.72 Pa（固定标定值）
```
神经网络学习 SP% → φ_max 的隐式映射，φ_max 逐样本满足物理约束（φ_max > φ + 0.05）。

---

## 包内容

```
PI-MFNN_v3_release/
├── README.md                       ← 本文件
├── models/
│   └── lian_v3_multifidelity.pth   ← PI-MFNN 主模型
└── data/
    └── hifi_test.csv               ← 测试集（360 条）
```

---

## 模型架构

**模型类**: `LianPINN_v3`（配置：h64_l3_tanh）

| 参数 | 值 |
|------|-----|
| 输入维度 | 2（Φ, SP%）|
| 隐藏层宽度 | 64 |
| 隐藏层数量 | 3 |
| 激活函数 | Tanh |
| 总参数量 | 8,577 |
| 约束方式 | 硬约束（物理公式嵌入 forward） |

---

## 数据说明

### data/hifi_test.csv

- **来源**: 400 条生成数据中按 seed=42 划分的测试子集
- **样本数**: 360 条
- **字段**:

| 字段 | 说明 | 单位 |
|------|------|------|
| `Phi` | 体积固含量 | 无量纲 |
| `SP_percent` | 超塑化剂掺量 | % |
| `tau0_Pa` | 屈服应力（真值） | Pa |

---

## 快速推理

### 环境要求
```
Python >= 3.8
torch >= 1.10
```

### 模型定义（直接复制可用）

```python
import torch
import torch.nn as nn

class LianPINN_v3(nn.Module):
    M1 = 0.72  # Pa

    def __init__(self, hidden_dim=64, n_hidden_layers=3, activation='tanh'):
        super().__init__()
        act = {'tanh': nn.Tanh, 'gelu': nn.GELU, 'silu': nn.SiLU}[activation]
        layers = [nn.Linear(2, hidden_dim), act()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        phi = x[:, 0]
        x_norm = x * torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        raw = self.net(x_norm).squeeze(-1)
        phi_max = phi + 0.05 + torch.sigmoid(raw) * (0.95 - phi - 0.05)
        tau0 = self.M1 * phi**3 / (phi_max * (torch.relu(phi_max - phi) + 1e-6))
        return tau0, phi_max
```

### 加载与推理

```python
model = LianPINN_v3()
model.load_state_dict(torch.load('models/lian_v3_multifidelity.pth', map_location='cpu'))
model.eval()

x = torch.tensor([[0.48, 0.30]], dtype=torch.float32)  # [Phi, SP%]
with torch.no_grad():
    tau0, phi_max = model(x)
print(f"τ₀ = {tau0.item():.4f} Pa,  φ_max = {phi_max.item():.4f}")
```

