"""
YodelPINN v1 — 基于 YODEL 方程的物理硬约束模型

参考: Flatt & Bowen (2006) J. Am. Ceram. Soc. 89(4): 1244–1256
     YODEL: Yield stress MODel for suspensions

物理方程 (PCL, 不含可训练参数):
    τ₀ = m_y * φ² / [φ_max(c_d) * (φ_max(c_d) − φ)²]

    与 Lian 2025 的结构区别:
      Lian:  τ₀ = m₁ * φ³ / [φ_max * (φ_max − φ)]     (线性分母，立方固含量)
      YODEL: τ₀ = m_y * φ² / [φ_max * (φ_max − φ)²]   (平方分母，平方固含量)

应用对象: 氧化铝(Al₂O₃)等无机陶瓷悬浮液
材料常数: m_y = 0.12 Pa  (陶瓷悬浮液典型值, 对应体系标定)

输入 (2维): [phi, c_d]
  phi  : 固体体积分数  (可测量)
  c_d  : 分散剂掺量 %  (可测量，类比水泥浆中的 SP%)

神经网络预测: φ_max  (不可直接测量，受分散剂影响)

网络结构: MLP 4层
  Linear(2→64) → Tanh → Linear(64→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1)
  与 LianPINN_v2 架构完全对称，PCL 替换为 YODEL 方程
"""

import torch
import torch.nn as nn


class YodelPINN_v1(nn.Module):
    """
    YODEL 硬约束 PI-MFNN 模型。

    模型结构与 LianPINN_v2 完全对称：
      - NPE (Neural Parameter Estimator): 4 层 MLP, 64 神经元 Tanh, 参数量 8,577
      - 约束解码器: Sigmoid 映射确保 φ_max > φ
      - PCL (Physics Computation Layer): YODEL 方程，无可训练参数
    """

    # 陶瓷悬浮液 YODEL 标定常数 (Al₂O₃ 体系参考值)
    M_Y = 0.12   # Pa

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # NPE: 与 LianPINN_v2 结构完全一致，仅 PCL 方程不同
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),   # 输出: raw_phi_max (未解码)
        )

    def forward(self, x: torch.Tensor):
        """
        前向推断。

        Args:
            x: Tensor [batch, 2] — (phi, c_d)
               phi : 固体体积分数
               c_d : 分散剂掺量 (%)

        Returns:
            tau0_pred    : Tensor [batch]  — 预测屈服应力 (Pa)
            phi_max_pred : Tensor [batch]  — 反演最大堆积分数
        """
        phi = x[:, 0]

        # 输入归一化 (陶瓷体系典型值: phi~0.40, c_d~0.80)
        scale = torch.tensor([2.5, 1.2], device=x.device, dtype=x.dtype)
        x_norm = x * scale

        raw = self.net(x_norm).squeeze(-1)   # [batch]

        # φ_max 解码: 物理约束 φ_max > φ
        # 范围: [φ + 0.05, 0.90]  (陶瓷悬浮液 φ_max 通常 < 0.80)
        phi_max_pred = phi + 0.05 + torch.sigmoid(raw) * (0.90 - phi - 0.05)

        # YODEL PCL: τ₀ = m_y * φ² / [φ_max * (φ_max − φ)²]
        epsilon = 1e-6
        diff = torch.relu(phi_max_pred - phi) + epsilon
        tau0_pred = self.M_Y * phi**2 / (phi_max_pred * diff**2)

        return tau0_pred, phi_max_pred


if __name__ == "__main__":
    model = YodelPINN_v1(hidden_dim=64)
    total = sum(p.numel() for p in model.parameters())
    print(f"模型: YodelPINN_v1  |  参数量: {total:,}")
    print(f"输入: [phi, c_d]   |  固定参数: m_y={YodelPINN_v1.M_Y} Pa")
    print(f"物理方程: τ₀ = m_y * φ² / [φ_max * (φ_max − φ)²]")
    print()

    # 典型陶瓷悬浮液样本测试
    test_cases = torch.tensor([
        [0.40, 0.50],   # 中等固含 + 低分散剂 → 高 τ₀
        [0.35, 1.00],   # 低固含 + 中等分散剂 → 中等 τ₀
        [0.45, 1.50],   # 高固含 + 高分散剂 → 较高 τ₀
    ])
    tau0, phi_max = model(test_cases)
    print(f"{'phi':<6} {'c_d%':<6} {'τ₀预测(Pa)':<14} {'φ_max预测'}")
    print("-" * 45)
    for i in range(len(test_cases)):
        print(f"{test_cases[i,0].item():.2f}   {test_cases[i,1].item():.2f}   "
              f"{tau0[i].item():.4f} Pa      {phi_max[i].item():.4f}")
    print()
    print("对比 LianPINN_v2: 相同 4层-64神经元-Tanh 架构，仅 PCL 方程不同")
    print(f"  LianPINN: τ = 0.72 * φ³ / [φ_max * (φ_max − φ)]")
    print(f"  YodelPINN: τ = {YodelPINN_v1.M_Y} * φ² / [φ_max * (φ_max − φ)²]")
