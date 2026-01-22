import numpy as np
import torch

from src.physics.yodel import calc_phi_c, calc_m1, calc_phi_m_dynamic, yodel_mechanism


def check_params():
    # 模拟参数
    n = 1000

    # 调整目标：
    # 1. Tau0_1 (48min) in [70, 120]
    # 2. Tau0_new (83min) approx 0.8 * Tau0_1

    # 微调参数以平衡 Tau0
    d50 = np.random.uniform(22.0, 28.0, n)
    sigma = np.random.uniform(1.5, 1.6, n)
    phi_2 = np.random.uniform(0.64, 0.66, n)

    # 显著降低固化剂比例，减弱稀释效应
    ratio = np.random.uniform(0.04, 0.06, n)

    # 计算 Phi_1
    phi_1 = phi_2 / (phi_2 + (1 - ratio) * (1 - phi_2))

    # 模拟 Emix
    # Point 1 (48 min)
    emix_1 = np.random.uniform(3e6, 4e6, n)
    temp_1 = np.random.uniform(47, 50, n)

    # Point New (83 min) - High Speed Mix 后，能量大幅增加
    # 35 min * 30 rpm * 2000 Nm * 2pi ~ 2.6e7 J
    emix_new = emix_1 + np.random.uniform(2.5e7, 2.7e7, n)
    temp_new = np.random.uniform(52, 53, n)

    # 转 Tensor
    t_d50 = torch.tensor(d50, dtype=torch.float32)
    t_sigma = torch.tensor(sigma, dtype=torch.float32)
    t_phi_1 = torch.tensor(phi_1, dtype=torch.float32)
    t_phi_2 = torch.tensor(phi_2, dtype=torch.float32) # 83min 时固含量已降至 Phi_2

    t_emix_1 = torch.tensor(emix_1, dtype=torch.float32)
    t_temp_1 = torch.tensor(temp_1, dtype=torch.float32)

    t_emix_new = torch.tensor(emix_new, dtype=torch.float32)
    t_temp_new = torch.tensor(temp_new, dtype=torch.float32)

    # YODEL Calculation
    phi_c = calc_phi_c(t_d50, t_sigma)
    phi_m0 = 0.65 + 0.1 * (t_sigma - 1.2)

    # Point 1
    g_max_1 = 80000.0 * (1 + 0.05 * (t_temp_1 - 25.0))
    m1_1 = calc_m1(t_d50, g_max_1)
    phi_m_1 = calc_phi_m_dynamic(phi_m0, t_emix_1)
    tau0_1 = yodel_mechanism(t_phi_1, phi_m_1, phi_c, m1_1)

    # Point New (83 min)
    g_max_new = 80000.0 * (1 + 0.05 * (t_temp_new - 25.0))
    m1_new = calc_m1(t_d50, g_max_new)
    phi_m_new = calc_phi_m_dynamic(phi_m0, t_emix_new) # 能量增加，Phi_m 增加
    tau0_new = yodel_mechanism(t_phi_2, phi_m_new, phi_c, m1_new) # 固含量降低

    # Analysis
    tau0_1_np = tau0_1.numpy()
    tau0_new_np = tau0_new.numpy()

    valid = (tau0_1_np >= 70) & (tau0_1_np <= 120)
    n_valid = np.sum(valid)

    ratio_drop = tau0_new_np / (tau0_1_np + 1e-6)

    print(f"Valid samples (Tau0_1 in 70-120): {n_valid}/{n} ({n_valid/n*100:.1f}%)")
    print(f"Tau0_1 stats: Mean={tau0_1_np.mean():.1f}")
    print(f"Tau0_new stats: Mean={tau0_new_np.mean():.1f}")
    print(f"Drop Ratio (New/1): Mean={ratio_drop.mean():.2f}")
    print(f"Phi_1 vs Phi_2: {phi_1.mean():.4f} -> {phi_2.mean():.4f}")

if __name__ == "__main__":
    check_params()

