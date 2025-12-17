
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
        生成合成数据集 (双阶段：Peak -> Final)

        物理过程：
        1. Peak阶段 (加氧化剂后)：液体较少(无固化剂)，固含量高 -> 高屈服值
        2. Final阶段 (加固化剂后)：液体增加，固含量降低 -> 低屈服值
        """
        if self.real_data is None or self.real_data.empty:
            raise ValueError("Real data is required to generate synthetic dataset based on actual process conditions.")

        # 1. 扩展真实数据
        df_expanded = self.real_data.loc[self.real_data.index.repeat(n_samples_per_real)].reset_index(drop=True)
        n_total = len(df_expanded)

        emix = df_expanded['Emix'].values
        temp = df_expanded['Temp_end'].values

        # 2. 随机生成配方参数
        # Phi_final: 最终固含量 [0.60, 0.75]
        phi_final = np.random.uniform(0.60, 0.75, n_total)

        # PSD: d50 [5, 50] um, sigma [1.2, 2.0]
        d50 = np.random.uniform(5.0, 50.0, n_total)
        sigma = np.random.uniform(1.2, 2.0, n_total)

        # Curing Agent Ratio: 固化剂在液相中的体积占比 [0.10, 0.25]
        # 这决定了从 Peak 到 Final 的稀释程度
        ratio_curing = np.random.uniform(0.10, 0.25, n_total)

        # 3. 转为 Tensor 进行机理计算
        t_phi_final = torch.tensor(phi_final, dtype=torch.float32)
        t_d50 = torch.tensor(d50, dtype=torch.float32)
        t_sigma = torch.tensor(sigma, dtype=torch.float32)
        t_emix = torch.tensor(emix, dtype=torch.float32)
        t_temp = torch.tensor(temp, dtype=torch.float32)
        t_ratio = torch.tensor(ratio_curing, dtype=torch.float32)

        # 4. 计算中间物理量
        # Phi_c
        phi_c = calc_phi_c(t_d50, t_sigma)

        # G_max (Temp dependent)
        g_max_base = 80000.0
        g_max = g_max_base * (1 + 0.05 * (t_temp - 25.0))

        # m1
        m1 = calc_m1(t_d50, g_max)

        # Phi_m (Dynamic)
        phi_m0 = 0.65 + 0.1 * (t_sigma - 1.2)
        phi_m = calc_phi_m_dynamic(phi_m0, t_emix)

        # 5. 计算两个阶段的固含量
        # Final 阶段: t_phi_final

        # Peak 阶段: 移除固化剂后的固含量
        # V_total = V_solid + V_liquid_total
        # V_liquid_total = V_liquid_other + V_curing
        # ratio = V_curing / V_liquid_total
        # Phi_final = V_solid / (V_solid + V_liquid_total)
        # Phi_peak = V_solid / (V_solid + V_liquid_other)
        #          = V_solid / (V_solid + V_liquid_total * (1 - ratio))
        #          = Phi_final / (Phi_final + (1-ratio)*(1-Phi_final))

        # 推导:
        # V_solid = Phi_final * V_total
        # V_liquid_total = (1 - Phi_final) * V_total
        # V_liquid_other = (1 - ratio) * V_liquid_total
        # V_peak_total = V_solid + V_liquid_other
        # Phi_peak = V_solid / V_peak_total

        v_solid = t_phi_final
        v_liquid_total = 1.0 - t_phi_final
        v_liquid_other = v_liquid_total * (1.0 - t_ratio)
        t_phi_peak = v_solid / (v_solid + v_liquid_other)

        # 6. 计算 Label (Tau0)
        tau0_final = yodel_mechanism(t_phi_final, phi_m, phi_c, m1)
        tau0_peak = yodel_mechanism(t_phi_peak, phi_m, phi_c, m1)

        # 7. 组装 DataFrame
        data = {
            'Phi_final(固含量)': phi_final,
            'd50(中位径_um)': d50,
            'sigma(几何标准差)': sigma,
            'Emix(混合功_J)': emix,
            'Temp(温度_C)': temp,
            'Curing_Ratio(固化剂比例)': ratio_curing,
            'Phi_peak(高峰固含量)': t_phi_peak.numpy(),
            'Phi_m_true(最大堆积)': phi_m.numpy(),
            'Tau0_peak(高峰屈服_Pa)': tau0_peak.numpy(),
            'Tau0_final(最终屈服_Pa)': tau0_final.numpy()
        }

        df = pd.DataFrame(data)

        # 8. 物理约束过滤
        initial_len = len(df)

        # 过滤 Phi < Phi_m
        df = df[df['Phi_peak(高峰固含量)'] < df['Phi_m_true(最大堆积)']]

        # 过滤 Phi > Phi_c
        df = df[df['Phi_final(固含量)'] > 0.2] # 简单过滤，具体由Yodel内部处理

        # 过滤 Tau0 有效性
        df = df[(df['Tau0_final(最终屈服_Pa)'] > 0) & (df['Tau0_final(最终屈服_Pa)'] < 10000)]
        df = df[(df['Tau0_peak(高峰屈服_Pa)'] > 0) & (df['Tau0_peak(高峰屈服_Pa)'] < 20000)]

        final_len = len(df)
        print(f"Filtered {initial_len - final_len} physically invalid samples.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            n_train = int(len(df) * 0.8)
            train_df = df.iloc[:n_train]
            test_df = df.iloc[n_train:]

            train_path = os.path.join(os.path.dirname(save_path), "train_data.csv")
            test_path = os.path.join(os.path.dirname(save_path), "test_data.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            df.to_csv(save_path, index=False)

            print(f"Generated {len(df)} samples.")
            print(f"Saved Train: {len(train_df)} samples -> {train_path}")
            print(f"Saved Test:  {len(test_df)} samples -> {test_path}")

        return df

if __name__ == "__main__":
    from src.data.loader import load_excel_data
    print("Loading real data...")
    real_df = load_excel_data()
    print(f"Loaded {len(real_df)} real process samples.")

    gen = SyntheticDataGenerator(real_df)
    gen.generate(n_samples_per_real=50)

