# src/physics/yodel.py

import torch


def calc_phi_c(d50, sigma):
    """
    计算渗透阈值 Phi_c (Percolation Threshold)
    基于 YODEL 文献近似: Phi_c ~ 0.28 * (1 + sigma_PSD / d50)
    这里简化假设 sigma_PSD (标准差) 与几何标准差 sigma (无量纲) 的关系。
    对于 Log-Normal 分布，变异系数 CV ~ sqrt(exp(sigma^2) - 1)。
    这里采用简化工程近似。

    Args:
        d50: 中位径 (Tensor)
        sigma: 几何标准差 (Tensor)
    Returns:
        Phi_c (Tensor)
    """
    # 简化的物理近似：粒径分布越宽(sigma越大)，堆积越密，但渗透阈值通常与配位数有关
    # 引用 Flatt 2006: Phi_c = 0.28 (对于球体) * 修正系数
    # 这里使用一个与 sigma 正相关的经验修正，假设宽分布更容易形成网络
    # 注意：YODEL 原文 Phi_c 是纯几何参数
    return 0.28 * (1 + (sigma - 1.0) * 0.5)

def calc_m1(d50, G_max):
    """
    计算几何预因子 m1
    m1 ~ G_max / R^2

    Args:
        d50: 粒径 (Tensor) [um]
        G_max: 颗粒间最大作用力 (Tensor) [Force]
    Returns:
        m1 (Tensor) [Pa]
    """
    # R ~ d50 / 2
    # m1 = const * G_max / (d50^2)
    # 这里的 const 包含几何因子，设为 1.0 (吸收到 G_max 中)
    return G_max / ((d50 / 2.0) ** 2 + 1e-8)

def calc_phi_m_dynamic(phi_m0, Emix, k_E=1e-3, phi_m_ultimate=0.74):
    """
    动力学修正：混合功改善最大堆积密度
    Phi_m = Phi_m0 + Delta * (1 - exp(-k * E))

    Args:
        phi_m0: 初始最大堆积 (Tensor)
        Emix: 混合功 (Tensor) [J]
        k_E: 速率常数
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

    # 分子: Phi_m * (Phi_m - Phi)
    numerator = phi_m * (phi_m - phi)
    numerator = torch.relu(numerator) # 确保非负

    # 分母: Phi * (Phi - Phi_c)^2
    # 软约束：如果 Phi < Phi_c，分母不应为0，且 tau0 应为 0
    diff_c = phi - phi_c
    diff_c = torch.relu(diff_c) + epsilon # 确保分母非零且正

    denominator = phi * (diff_c ** 2)

    tau0 = m1 * numerator / denominator

    # 如果 Phi < Phi_c，理论上 tau0 = 0
    mask = (phi > phi_c).float()
    return tau0 * mask

