import numpy as np
import torch
from src.physics.yodel import calc_phi_c, calc_m1, calc_phi_m_dynamic, yodel_mechanism

def check_params():
    # 模拟参数
    n = 1000
    d50 = np.random.uniform(25.0, 35.0, n)
    sigma = np.random.uniform(1.5, 1.6, n)
    phi_2 = np.random.uniform(0.62, 0.65, n)
    ratio = np.random.uniform(0.15, 0.20, n)

    # 计算 Phi_1
    phi_1 = phi_2 / (phi_2 + (1 - ratio) * (1 - phi_2))

    # 模拟 Emix (Point 1)
    # 粗略估计: 48 min, speed 24, torque 450
    # E = 2 * pi * 24 * 450 * 48 ~ 3.2e6
    emix_1 = np.random.uniform(3e6, 4e6, n)
    temp_1 = np.random.uniform(47, 50, n)

    # 转 Tensor
    t_d50 = torch.tensor(d50, dtype=torch.float32)
    t_sigma = torch.tensor(sigma, dtype=torch.float32)
    t_phi_1 = torch.tensor(phi_1, dtype=torch.float32)
    t_emix_1 = torch.tensor(emix_1, dtype=torch.float32)
    t_temp_1 = torch.tensor(temp_1, dtype=torch.float32)

    # YODEL
    phi_c = calc_phi_c(t_d50, t_sigma)
    g_max = 80000.0 * (1 + 0.05 * (t_temp_1 - 25.0))
    m1 = calc_m1(t_d50, g_max)
    phi_m0 = 0.65 + 0.1 * (t_sigma - 1.2)
    phi_m = calc_phi_m_dynamic(phi_m0, t_emix_1)

    tau0 = yodel_mechanism(t_phi_1, phi_m, phi_c, m1)

    tau0_np = tau0.numpy()
    valid = (tau0_np >= 70) & (tau0_np <= 120)
    n_valid = np.sum(valid)

    print(f"Valid samples: {n_valid}/{n} ({n_valid/n*100:.1f}%)")
    print(f"Tau0 stats: Min={tau0_np.min():.1f}, Max={tau0_np.max():.1f}, Mean={tau0_np.mean():.1f}")
    print(f"Phi_1 stats: Mean={phi_1.mean():.3f}")
    print(f"Phi_m stats: Mean={phi_m.mean():.3f}")

if __name__ == "__main__":
    check_params()

