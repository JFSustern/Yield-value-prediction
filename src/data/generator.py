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
        # 严格使用真实温度，不引入人为扰动
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
        # Phi_c (Log-Normal corrected)
        phi_c = calc_phi_c(t_d50, t_sigma)

        # G_max (Temp dependent)
        # g_max_base = 80000.0 对应真实物理力约 80 nN (纳牛)，符合范德华力/液桥力范围
        g_max_base = 80000.0
        g_max = g_max_base * (1 + 0.05 * (t_temp - 25.0))

        # m1 (Restored geometric constant)
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

        # 7. 物理约束过滤 (关键步骤)
        initial_len = len(df)

        # 过滤 Phi >= Phi_m (防止堵塞/负值)
        df = df[df['Phi(固含量)'] < df['Phi_m_true(最大堆积)']]

        # 过滤 Phi <= Phi_c_true (防止无屈服/分母为0)
        df = df[df['Phi(固含量)'] > df['Phi_c_true(渗透阈值)']]

        # 过滤 Tau0 异常值 (例如 > 5000 Pa 或 < 0)
        df = df[(df['Tau0(屈服应力_Pa)'] > 0) & (df['Tau0(屈服应力_Pa)'] < 5000)]

        final_len = len(df)
        print(f"Filtered {initial_len - final_len} physically invalid samples.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 随机打乱
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # 划分训练集和测试集 (8:2)
            n_train = int(len(df) * 0.8)
            train_df = df.iloc[:n_train]
            test_df = df.iloc[n_train:]

            # 保存
            train_path = os.path.join(os.path.dirname(save_path), "train_data.csv")
            test_path = os.path.join(os.path.dirname(save_path), "test_data.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            # 为了兼容旧代码，也保存一份完整的
            df.to_csv(save_path, index=False)

            print(f"Generated {len(df)} samples.")
            print(f"Saved Train: {len(train_df)} samples -> {train_path}")
            print(f"Saved Test:  {len(test_df)} samples -> {test_path}")

        return df

if __name__ == "__main__":
    from src.data.loader import load_excel_data

    # 1. 加载真实数据
    print("Loading real data...")
    real_df = load_excel_data()
    print(f"Loaded {len(real_df)} real process samples.")

    # 2. 基于真实工况生成合成数据
    gen = SyntheticDataGenerator(real_df)
    gen.generate(n_samples_per_real=50)

