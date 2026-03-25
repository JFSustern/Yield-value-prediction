"""
PINN 模型 - 基于 Lian et al. (Materials 2025) 公式

输入特征: [Phi, d50_um, sigma, SP_percent]  (4维)
物理公式: τ = m1 * φ³ / [φ_max * (φ_max - φ)]

神经网络任务: 从可测量输入预测两个难以直接测量的物理参数
  - φ_max: 最大堆积分数 (主要受增塑剂 SP 影响)
  - m1:    屈服应力系数 (受粉体特性影响, Table 6 标定值 0.72 Pa)
"""

import torch
import torch.nn as nn

from multi_fidelity.src.physics.lian2025 import paper_yield_stress


class LianPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        """
        Args:
            input_dim: 输入维度 [Phi, d50_um, sigma, SP_percent]
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 标准 MLP: 从可测量输入预测物理参数 (φ_max, m1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)   # 输出: [raw_phi_max, raw_m1]
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 4] -> (Phi, d50_um, sigma, SP_percent)
        Returns:
            tau0_pred: 预测屈服应力 [Pa]
            (phi_max_pred, m1_pred): 中间物理参数
        """
        # 输入归一化 (基于 Table 6 典型值)
        # Phi~0.48, d50~16, sigma~1.7, SP~0.6
        scale = torch.tensor([2.0, 0.06, 0.6, 1.5], device=x.device, dtype=x.dtype)
        x_norm = x * scale

        out = self.net(x_norm)

        phi        = x[:, 0]
        raw_phi_max = out[:, 0]
        raw_m1      = out[:, 1]

        # φ_max 解码: 必须大于 φ, 范围限制在 [φ+0.05, 0.95]
        phi_max_pred = phi + 0.05 + torch.sigmoid(raw_phi_max) * (0.95 - phi - 0.05)

        # m1 解码: 必须 > 0, 论文值约 0.72 Pa, 允许范围 [0.1, 5.0]
        m1_pred = 0.1 + torch.sigmoid(raw_m1) * 4.9

        # 论文公式计算屈服应力
        tau0_pred = paper_yield_stress(phi, phi_max_pred, m1_pred)

        return tau0_pred, (phi_max_pred, m1_pred)


if __name__ == "__main__":
    model = LianPINN(input_dim=4, hidden_dim=128)
    total = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total:,}")

    # 模拟 Table 6 输入: Phi=0.48, d50=16.6, sigma=1.7, SP=0.6%
    x = torch.tensor([[0.48, 16.6, 1.7, 0.6]])
    tau0, (phi_max, m1) = model(x)
    print(f"Input:        Phi={x[0,0]:.3f}, d50={x[0,1]:.1f}μm, sigma={x[0,2]:.2f}, SP={x[0,3]:.2f}%")
    print(f"Predicted:    tau0={tau0.item():.4f} Pa")
    print(f"Intermediate: phi_max={phi_max.item():.4f}, m1={m1.item():.4f} Pa")
    print(f"Table 6 ref:  tau0≈0.19-1.95 Pa")
