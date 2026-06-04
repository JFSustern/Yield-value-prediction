"""
ZhouPINN v1 — 基于完整 YODEL 方程的物理硬约束模型

参考数据:  Zhou, Solomon, Scales, Boger (1999)
           "The yield stress of concentrated flocculated suspensions
            of size distributed particles." J. Rheol. 43(3): 651–71.
           DOI: 10.1122/1.551029

物理方程 (YODEL Eq. 42, Flatt & Bowen 2006):
    τ = m₁_eff × φ(φ − φ₀)² / [φ_max_ref × (φ_max_ref − φ)]

固定参数 (来自 Zhou 1999 + YODEL paper 文献值):
    φ_max_ref = 0.570   四种 AKP 粉末的最大堆积分数（压滤独立测定）
    φ₀        = 0.026   渗流阈值（YODEL paper 拟合值）

NPE 反演参数 (隐变量):
    m₁_eff : 颗粒间力参数 (Pa)，取决于粒径 d_s，不可在线直接测量
             理论关系: m₁_eff ≈ K / d_s²  (K 为 Hamaker 常数相关量)

与 Lian 2025 体系的核心区别:
    Lian:  NPE 反演 φ_max (堆积参数, 受 SP% 控制)   PCL = Lian 方程
    Zhou:  NPE 反演 m₁_eff (颗粒间力, 受粒径控制)    PCL = YODEL 方程
           → 不同隐变量, 不同方程形式, 不同材料体系, 真实实验 HF 数据

输入 (2维): [phi, d_s_um]
    phi    : 固体体积分数  (可测量)
    d_s_um : 表面平均粒径 μm (可测量，BET/Coulter)

网络结构: 与 LianPINN_v2 对称
    Linear(2→64) → Tanh → Linear(64→64) → Tanh → Linear(64→64) → Tanh → Linear(64→1)
"""

import torch
import torch.nn as nn


class ZhouPINN_v1(nn.Module):
    """
    Zhou 1999 / YODEL 硬约束 PI-MFNN 模型。

    架构与 LianPINN_v2 对称（同 4层-64神经元-Tanh），
    区别仅在 PCL 方程形式及隐变量物理意义。
    """

    # 文献固定参数 (Zhou 1999 + YODEL paper)
    PHI_MAX_REF = 0.570   # 四种 AKP 粉末拟合的最大堆积分数
    PHI_0       = 0.026   # 渗流阈值
    M1_LO       = 50.0    # m₁_eff 解码下界 (Pa)
    M1_HI       = 3000.0  # m₁_eff 解码上界 (Pa)

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # NPE: 与 LianPINN_v2 完全对称
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),   # 输出: raw_m1 (未解码)
        )

    def forward(self, x: torch.Tensor):
        """
        前向推断。

        Args:
            x: Tensor [batch, 2] — (phi, d_s_um)
               phi    : 固体体积分数
               d_s_um : 表面平均粒径 (μm)

        Returns:
            tau_pred : Tensor [batch]  — 预测屈服应力 (Pa)
            m1_pred  : Tensor [batch]  — 反演颗粒间力参数 (Pa)
        """
        phi    = x[:, 0]

        # 输入归一化 (AKP 系列典型值: phi~0.35, d_s~0.35 μm)
        scale  = torch.tensor([2.5, 3.0], device=x.device, dtype=x.dtype)
        x_norm = x * scale

        raw   = self.net(x_norm).squeeze(-1)   # [batch]

        # m₁_eff 解码: Sigmoid → [M1_LO, M1_HI]
        m1_pred = self.M1_LO + torch.sigmoid(raw) * (self.M1_HI - self.M1_LO)

        # YODEL PCL (Eq. 42, Flatt & Bowen 2006):
        #   τ = m₁_eff × φ(φ − φ₀)² / [φ_max_ref × (φ_max_ref − φ)]
        phi0    = self.PHI_0
        phi_max = self.PHI_MAX_REF
        epsilon = 1e-6

        numerator   = m1_pred * phi * torch.clamp(phi - phi0, min=epsilon) ** 2
        denominator = phi_max * (torch.clamp(phi_max - phi, min=epsilon))
        tau_pred    = numerator / denominator

        return tau_pred, m1_pred


if __name__ == "__main__":
    model = ZhouPINN_v1(hidden_dim=64)
    total = sum(p.numel() for p in model.parameters())
    print(f"模型: ZhouPINN_v1  |  参数量: {total:,}")
    print(f"输入: [phi, d_s_um]")
    print(f"固定: φ_max={ZhouPINN_v1.PHI_MAX_REF}, φ₀={ZhouPINN_v1.PHI_0}")
    print(f"反演: m₁_eff ∈ [{ZhouPINN_v1.M1_LO}, {ZhouPINN_v1.M1_HI}] Pa")
    print(f"PCL:  τ = m₁_eff × φ(φ−φ₀)² / [φ_max(φ_max−φ)]")
    print()

    # AKP 系列典型样本 (已知 m₁ 分别约 310/470/830/1380 Pa)
    # 粒径 d_s (μm): AKP-15=0.580, AKP-20=0.401, AKP-30=0.247, AKP-50=0.185
    test_cases = torch.tensor([
        [0.40, 0.580],  # AKP-15, φ=0.40, 预期 τ ≈ 480 Pa
        [0.40, 0.401],  # AKP-20, φ=0.40, 预期 τ ≈ 730 Pa
        [0.40, 0.247],  # AKP-30, φ=0.40, 预期 τ ≈ 1300 Pa
        [0.40, 0.185],  # AKP-50, φ=0.40, 预期 τ ≈ 2100 Pa
    ])
    tau, m1 = model(test_cases)
    print(f"{'phi':<6} {'d_s/μm':<8} {'τ预测/Pa':<14} {'m₁反演/Pa'}")
    print("-" * 45)
    for i in range(len(test_cases)):
        print(f"{test_cases[i,0].item():.2f}   {test_cases[i,1].item():.3f}   "
              f"{tau[i].item():>10.1f}    {m1[i].item():>8.1f}")
    print()
    print("与 LianPINN_v2 对比:")
    print(f"  LianPINN: 反演 φ_max (堆积参数),    PCL = m₁×φ³/[φ_max×(φ_max-φ)]")
    print(f"  ZhouPINN: 反演 m₁_eff (颗粒间力), PCL = m₁×φ(φ-φ₀)²/[φ_max×(φ_max-φ)]")
