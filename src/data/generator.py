# src/data/generator.py

import os

import numpy as np
import pandas as pd
import torch

from src.physics.yodel import calc_phi_c, calc_m1, calc_phi_m_dynamic, yodel_mechanism


class SyntheticDataGenerator:
    def __init__(self, real_data_df=None):
        """
        Args:
            real_data_df: 包含真实 Emix 和 Temp 分布的 DataFrame
        """
        self.real_data = real_data_df

    def generate(self, n_samples_per_real=10, save_path="data/synthetic/dataset.csv"):
        """
        生成合成数据集
        策略：遍历真实数据的每一行 (Emix, Temp)，为其生成 n_samples_per_real 组随机的 (Phi, PSD)
        这样保证工况分布与真实数据完全一致。

        Args:
            n_samples_per_real: 每个真实样本生成的虚拟配方数量
        """
        if self.real_data is None or self.real_data.empty:
            raise ValueError("Real data is required to generate synthetic dataset based on actual process conditions.")

        # 1. 扩展真实数据
        # 重复真实数据行
        df_expanded = self.real_data.loc[self.real_data.index.repeat(n_samples_per_real)].reset_index(drop=True)
        n_total = len(df_expanded)

        emix = df_expanded['Emix'].values
        temp = df_expanded['Temp_end'].values

        # 2. 随机生成缺失的物性参数
        # Phi: 固含量 [0.60, 0.75]
        phi = np.random.uniform(0.60, 0.75, n_total)

        # PSD: d50 [5, 50] um, sigma [1.2, 2.0]
        d50 = np.random.uniform(5.0, 50.0, n_total)
        sigma = np.random.uniform(1.2, 2.0, n_total)

        # 3. 转为 Tensor 进行机理计算 (Ground Truth Generation)
        t_phi = torch.tensor(phi, dtype=torch.float32)
        t_d50 = torch.tensor(d50, dtype=torch.float32)
        t_sigma = torch.tensor(sigma, dtype=torch.float32)
        t_emix = torch.tensor(emix, dtype=torch.float32)
        t_temp = torch.tensor(temp, dtype=torch.float32)

        # 4. 计算中间物理量
        # Phi_c
        phi_c = calc_phi_c(t_d50, t_sigma)

        # G_max (Temp dependent)
        # G_max = G0 * exp(Ea/RT)
        # 简化：G_max 随温度升高而降低 (假设热运动破坏结构) 或升高 (固化)
        # 这里假设固化主导：温度高 -> 固化快 -> G_max 大
        # 由于 calc_m1 引入了 1.8/pi^4 (~0.0185) 的常数，这里 G_max 需要相应放大
        g_max_base = 5000.0 # 基础值 (原 100.0 / 0.0185 ≈ 5400)
        g_max = g_max_base * (1 + 0.05 * (t_temp - 25.0)) # 简单线性假设用于生成数据

        # m1
        m1 = calc_m1(t_d50, g_max)

        # Phi_m (Dynamic)
        # 基础 Phi_m 与 PSD 有关 (宽分布堆积更密)
        phi_m0 = 0.65 + 0.1 * (t_sigma - 1.2)
        phi_m = calc_phi_m_dynamic(phi_m0, t_emix)

        # 5. 计算 Label (Tau0)
        tau0 = yodel_mechanism(t_phi, phi_m, phi_c, m1)

        # 6. 组装 DataFrame
        data = {
            'Phi(固含量)': phi,
            'd50(中位径_um)': d50,
            'sigma(几何标准差)': sigma,
            'Emix(混合功_J)': emix,
            'Temp(温度_C)': temp,
            'Phi_c_true(渗透阈值)': phi_c.numpy(),
            'Phi_m_true(最大堆积)': phi_m.numpy(),
            'm1_true(强度因子_Pa)': m1.numpy(),
            'Tau0(屈服应力_Pa)': tau0.numpy()
        }

        df = pd.DataFrame(data)

        # 过滤无效数据 (Tau0 = 0 或 NaN)
        df = df[df['Tau0(屈服应力_Pa)'] > 1e-3].dropna()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Saved {len(df)} samples to {save_path}")

        return df

if __name__ == "__main__":
    from src.data.loader import load_excel_data

    # 1. 加载真实数据
    print("Loading real data...")
    real_df = load_excel_data()
    print(f"Loaded {len(real_df)} real process samples.")

    # 2. 基于真实工况生成合成数据
    gen = SyntheticDataGenerator(real_df)
    gen.generate(n_samples_per_real=50) # 每个真实工况生成 50 种虚拟配方

