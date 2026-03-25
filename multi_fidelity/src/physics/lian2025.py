"""
论文公式实现
来源: Lian et al., Materials 2025, 18, 2983
"Constitutive Modeling of Rheological Behavior of Cement Paste
 Based on Material Composition"

屈服应力公式 (Eq.2 / Eq.4):
  τ = m1 * φ³ / [φ_max * (φ_max - φ)]

含增塑剂时 φ_max = φ_max(SP), m1 和 af 保持不变。
"""

import torch


def paper_yield_stress(phi, phi_max, m1):
    """
    论文 Eq.2 / Eq.4 屈服应力公式

    Args:
        phi:     固含量 (Tensor)
        phi_max: 最大堆积分数 (Tensor), 含增塑剂时随 SP 掺量变化
        m1:      屈服应力系数 (Tensor) [Pa], Table 6 标定值 = 0.72 Pa
    Returns:
        tau0 (Tensor) [Pa]
    """
    epsilon = 1e-6
    diff_m = torch.relu(phi_max - phi) + epsilon
    denominator = phi_max * diff_m
    return m1 * phi**3 / denominator
