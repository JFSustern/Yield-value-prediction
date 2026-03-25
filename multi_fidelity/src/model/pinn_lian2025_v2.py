"""
LianPINN v2 - 严格对齐论文公式

论文: Lian et al., Materials 2025, 18, 2983
公式: τ = m1 * φ³ / [φ_max(SP) * (φ_max(SP) - φ)]

输入 (2维): [Phi, SP_percent]
神经网络预测: φ_max  (不可直接测量，受 SP 影响)
固定参数:     m1 = 0.72 Pa

网络结构: MLP 4层
  Linear(2→64) → Tanh → Linear(64→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1)
  输出: raw_phi_max → sigmoid 解码为合法的 φ_max
"""

import torch
import torch.nn as nn


class LianPINN_v2(nn.Module):

    # 论文 Table 6 标定的固定参数
    M1 = 0.72  # Pa

    def __init__(self, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),   # 输出: raw_phi_max
        )

    def forward(self, x):
        """
        Args:
            x: [batch, 2]  -> (Phi, SP_percent)
        Returns:
            tau0_pred:    [batch]  预测屈服应力 (Pa)
            phi_max_pred: [batch]  预测最大堆积分数
        """
        phi = x[:, 0]
        sp  = x[:, 1]

        # 输入归一化 (基于 Table 6 典型值: Phi~0.48, SP~0.6)
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        x_norm = x * scale

        raw = self.net(x_norm).squeeze(-1)   # [batch]

        # φ_max 解码: 必须 > φ，物理上限 0.95
        # 范围: [φ + 0.05, 0.95]
        phi_max_pred = phi + 0.05 + torch.sigmoid(raw) * (0.95 - phi - 0.05)

        # 论文公式 (m1 固定)
        epsilon = 1e-6
        diff_m = torch.relu(phi_max_pred - phi) + epsilon
        tau0_pred = self.M1 * phi**3 / (phi_max_pred * diff_m)

        return tau0_pred, phi_max_pred


if __name__ == "__main__":
    model = LianPINN_v2(hidden_dim=64)
    total = sum(p.numel() for p in model.parameters())
    print(f"模型: LianPINN_v2  |  参数量: {total:,}")
    print(f"输入: [Phi, SP_percent]  |  固定参数: m1={LianPINN_v2.M1} Pa")

    # 模拟 Table 6 样本
    test_cases = torch.tensor([
        [0.502, 0.40],   # Mix 4: τ₀=1.95 Pa
        [0.503, 0.50],   # Mix 5: τ₀=0.97 Pa
        [0.459, 1.00],   # Mix 3: τ₀=0.19 Pa
    ])
    tau0, phi_max = model(test_cases)
    print(f"\n{'Phi':<8} {'SP%':<8} {'τ₀预测':<12} {'φ_max预测':<12}")
    print("-" * 45)
    for i in range(len(test_cases)):
        print(f"{test_cases[i,0].item():.3f}    {test_cases[i,1].item():.2f}    "
              f"{tau0[i].item():.4f} Pa    {phi_max[i].item():.4f}")
    print(f"\n论文 Table 6 真实值: 1.95 / 0.97 / 0.19 Pa")
