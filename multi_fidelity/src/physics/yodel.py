# src/physics/yodel.py

import numpy as np
import torch


def calc_phi_c(d50, sigma):
    """
    计算渗透阈值 Phi_c (Percolation Threshold)

    文献来源：
    - Flatt & Bowen (2006): "Yield stress of suspensions of non-aggregated particles"
    - Journal of the American Ceramic Society, Vol. 89, No. 4, pp. 1244-1256
    - DOI: 10.1111/j.1551-2916.2005.00888.x
    - 关键页码：p. 1251

    原始公式：
    Phi_c = 0.28 * (1 + sigma_PSD / d50)

    其中：
    - 0.28：基础渗透阈值（球形颗粒随机堆积）
    - sigma_PSD：线性空间的粒径分布标准差
    - d50：中位径
    - sigma_PSD / d50：变异系数 (Coefficient of Variation, CV)

    对于 Log-Normal 分布，线性空间标准差与几何标准差的关系：
    sigma_PSD = d50 * sqrt(exp((ln(sigma))^2) - 1)

    因此：
    CV = sigma_PSD / d50 = sqrt(exp((ln(sigma))^2) - 1)

    Args:
        d50: 中位径 (Tensor) [um]
        sigma: 几何标准差 (Tensor) [无量纲]
    Returns:
        Phi_c (Tensor) [无量纲]
    """
    # 从几何标准差转换到线性空间的变异系数
    # 对于 Log-Normal 分布：CV = sqrt(exp((ln(sigma))^2) - 1)
    ln_sigma = torch.log(sigma)
    cv = torch.sqrt(torch.exp(ln_sigma**2) - 1)

    # 代入 YODEL 原始公式
    phi_c = 0.28 * (1 + cv)

    return phi_c

def calc_m1(d50, G_max):
    """
    计算几何预因子 m1

    文献公式 (Eq. 41): m1 = (1.8/pi^4) * (G_max / R_v50) * F_sigma

    Args:
        d50: 粒径 (Tensor) [um]
        G_max: 颗粒间最大作用力 (Tensor) [Force]
    Returns:
        m1 (Tensor) [Pa]
    """
    # 几何常数
    const = 1.8 / (np.pi ** 4)

    # 半径
    R = d50 / 2.0

    # 计算 m1
    # + 1e-8 是为了防止除零
    return const * G_max / (R ** 2 + 1e-8)

def calc_phi_m_dynamic(phi_m0, Emix, k_E=1e-8, phi_m_ultimate=0.74):
    """
    动力学修正：混合功改善最大堆积密度
    Phi_m = Phi_m0 + Delta * (1 - exp(-k * E))

    Args:
        phi_m0: 初始最大堆积 (Tensor)
        Emix: 混合功 (Tensor) [J]
        k_E: 速率常数 [1/J]。
             对于高能混合过程 (E ~ 10^8 J)，k_E 应在 1e-8 量级，
             以体现渐进的结构演化。
        phi_m_ultimate: 极限堆积 (FCC ~ 0.74)
    Returns:
        Phi_m_eff (Tensor)
    """
    delta = phi_m_ultimate - phi_m0
    # 确保 delta 非负
    delta = torch.clamp(delta, min=0.0)
    return phi_m0 + delta * (1 - torch.exp(-k_E * Emix))

def yodel_mechanism(phi, phi_m, phi_c, m1):
    """
    YODEL 主方程
    tau0 = m1 * (Phi_m * (Phi_m - Phi)) / (Phi * (Phi - Phi_c)^2)

    Args:
        phi: 固含量
        phi_m: 最大堆积
        phi_c: 渗透阈值
        m1: 强度因子
    Returns:
        tau0 (Tensor)
    """
    # 数值稳定性处理
    epsilon = 1e-6

    # 物理约束 1: Phi < Phi_m (否则堵塞/无穷大)
    # 物理约束 2: Phi > Phi_c (否则无屈服值/流体)

    # 分子: Phi * (Phi - Phi_c)^2
    diff_c = phi - phi_c
    diff_c = torch.relu(diff_c) # 确保非负，如果 phi < phi_c，则为0
    numerator = phi * (diff_c ** 2)

    # 分母: Phi_m * (Phi_m - Phi)
    # 软约束：如果 Phi -> Phi_m，分母 -> 0，tau0 -> inf
    diff_m = phi_m - phi
    diff_m = torch.relu(diff_m) + epsilon # 确保分母非零且正
    denominator = phi_m * diff_m

    tau0 = m1 * numerator / denominator

    # 如果 Phi < Phi_c，理论上 tau0 = 0
    mask = (phi > phi_c).float()
    return tau0 * mask

