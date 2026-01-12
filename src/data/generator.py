
import os

import numpy as np
import pandas as pd
import torch

from src.physics.yodel import calc_phi_c, calc_m1, calc_phi_m_dynamic, yodel_mechanism


class ProcessBasedGenerator:
    def __init__(self):
        # 工序定义
        # 阶段: (duration_min, speed_range, torque_range, temp_range)
        self.stages = [
            # 1. Initial Mix
            {'duration': 10, 'speed': (22, 26), 'torque': (400, 500), 'temp': (42, 44)},
            # 2. Add Oxidizer 1
            {'duration': 10, 'speed': (22, 26), 'torque': (450, 550), 'temp': (43.5, 44.5)},
            # 3. Add Oxidizer 2
            {'duration': 10, 'speed': (22, 26), 'torque': (600, 700), 'temp': (47, 48)},
            # 4. Add Oxidizer 3
            {'duration': 10, 'speed': (22, 26), 'torque': (900, 1000), 'temp': (47, 49)},
            # 5. Mix (End of T1 = 48 min)
            {'duration': 8,  'speed': (22, 26), 'torque': (450, 500), 'temp': (47, 50)},
            # 6. High Speed Mix
            {'duration': 35, 'speed': (28, 32), 'torque': (1950, 2050), 'temp': (52, 53)},
            # 7. Add Curing Agent
            {'duration': 3,  'speed': (10, 14), 'torque': (1750, 1850), 'temp': (50, 51)},
            # 8. Final Mix (End of T2 = 111 min)
            {'duration': 25, 'speed': (28, 32), 'torque': (1300, 1500), 'temp': (50, 51)},
            # 9. Resting (15 min) - Not used for energy calculation as speed is 0
            {'duration': 15, 'speed': (0, 0),   'torque': (0, 0),       'temp': (25, 25)}
        ]

    def _simulate_single_process(self):
        """模拟单个批次的工序参数"""
        emix_accum = 0.0

        # 记录关键时间点的状态
        state_t1 = {} # 48 min
        state_t2 = {} # 111 min

        current_time = 0

        for i, stage in enumerate(self.stages):
            duration = stage['duration'] # min

            # 随机采样该阶段的平均参数
            speed = np.random.uniform(*stage['speed']) # rpm
            torque = np.random.uniform(*stage['torque']) # Nm
            temp = np.random.uniform(*stage['temp']) # C

            # 计算该阶段混合功 (Joules)
            # Power (W) = 2 * pi * (speed / 60) * torque
            # Energy (J) = Power * (duration * 60)
            # Simplified: 2 * pi * speed * torque * duration
            energy = 2 * np.pi * speed * torque * duration

            emix_accum += energy
            current_time += duration

            # Checkpoints
            # T1: End of Stage 5 (10+10+10+10+8 = 48)
            if i == 4:
                state_t1['Emix'] = emix_accum
                state_t1['Temp'] = temp

            # T2: End of Stage 8 (48 + 35 + 3 + 25 = 111)
            if i == 7:
                state_t2['Emix'] = emix_accum
                state_t2['Temp'] = temp

        return state_t1, state_t2

    def generate(self, n_samples=2000, save_path="data/synthetic/dataset.csv"):
        print(f"Generating {n_samples} samples based on process chart...")

        data_list = []

        for _ in range(n_samples):
            # 1. 基础物性 (PSD)
            # 缩小方差：d50 [20, 30], sigma [1.4, 1.6]
            d50 = np.random.uniform(20.0, 30.0)
            sigma = np.random.uniform(1.4, 1.6)

            # 2. 固含量 (Phi)
            # T2 (Final) 固含量
            phi_2 = np.random.uniform(0.60, 0.75)

            # T1 (Peak) 固含量 - 未加固化剂，所以固含量更高
            # 假设固化剂占液相体积的 10% - 25%
            ratio_curing = np.random.uniform(0.10, 0.25)

            # Phi_1 计算 (稀释逆推)
            # V_solid = Phi_2 * V_total_2
            # V_liquid_2 = (1 - Phi_2) * V_total_2
            # V_liquid_1 = V_liquid_2 * (1 - ratio_curing)
            # V_total_1 = V_solid + V_liquid_1
            # Phi_1 = V_solid / V_total_1

            # 简化公式: Phi_1 = Phi_2 / (Phi_2 + (1-ratio)*(1-Phi_2))
            phi_1 = phi_2 / (phi_2 + (1 - ratio_curing) * (1 - phi_2))

            # 3. 工序模拟 (Emix, Temp)
            state_t1, state_t2 = self._simulate_single_process()

            # 4. 物理计算 (YODEL)
            # 转换为 Tensor
            t_d50 = torch.tensor(d50, dtype=torch.float32)
            t_sigma = torch.tensor(sigma, dtype=torch.float32)

            # Phi_c (Constant)
            phi_c = calc_phi_c(t_d50, t_sigma)

            # --- Point 1 (48 min) ---
            t_phi_1 = torch.tensor(phi_1, dtype=torch.float32)
            t_emix_1 = torch.tensor(state_t1['Emix'], dtype=torch.float32)
            t_temp_1 = torch.tensor(state_t1['Temp'], dtype=torch.float32)

            # G_max 1
            g_max_base = 80000.0
            g_max_1 = g_max_base * (1 + 0.05 * (t_temp_1 - 25.0))

            # m1 1
            m1_1 = calc_m1(t_d50, g_max_1)

            # Phi_m 1
            phi_m0 = 0.65 + 0.1 * (t_sigma - 1.2)
            phi_m_1 = calc_phi_m_dynamic(phi_m0, t_emix_1)

            # Tau0 1
            tau0_1 = yodel_mechanism(t_phi_1, phi_m_1, phi_c, m1_1)

            # --- Point 2 (111 min) ---
            t_phi_2 = torch.tensor(phi_2, dtype=torch.float32)
            t_emix_2 = torch.tensor(state_t2['Emix'], dtype=torch.float32)
            t_temp_2 = torch.tensor(state_t2['Temp'], dtype=torch.float32)

            # G_max 2
            g_max_2 = g_max_base * (1 + 0.05 * (t_temp_2 - 25.0))

            # m1 2
            m1_2 = calc_m1(t_d50, g_max_2)

            # Phi_m 2
            phi_m_2 = calc_phi_m_dynamic(phi_m0, t_emix_2)

            # Tau0 2
            tau0_2 = yodel_mechanism(t_phi_2, phi_m_2, phi_c, m1_2)

            # 5. 收集数据
            row = {
                'd50(中位径_um)': d50,
                'sigma(几何标准差)': sigma,

                # Point 1
                'Phi_1(固含量)': phi_1,
                'Phi_m_1(最大堆积)': phi_m_1.item(),
                'Emix_1(混合功_J)': state_t1['Emix'],
                'Temp_1(温度_C)': state_t1['Temp'],
                'Tau0_1(屈服应力_Pa)': tau0_1.item(),

                # Point 2
                'Phi_2(固含量)': phi_2,
                'Phi_m_2(最大堆积)': phi_m_2.item(),
                'Emix_2(混合功_J)': state_t2['Emix'],
                'Temp_2(温度_C)': state_t2['Temp'],
                'Tau0_2(屈服应力_Pa)': tau0_2.item(),
            }
            data_list.append(row)

        df = pd.DataFrame(data_list)

        # 6. 物理约束过滤
        # 过滤无效值
        df = df[
            (df['Tau0_1(屈服应力_Pa)'] > 0) & (df['Tau0_1(屈服应力_Pa)'] < 5000) &
            (df['Tau0_2(屈服应力_Pa)'] > 0) & (df['Tau0_2(屈服应力_Pa)'] < 5000) &
            (df['Phi_1(固含量)'] < df['Phi_m_1(最大堆积)']) &
            (df['Phi_2(固含量)'] < df['Phi_m_2(最大堆积)'])
        ]

        # Round to 4 decimal places
        df = df.round(4)

        print(f"Generated {len(df)} valid samples after filtering.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 保存宽格式 (Raw)
            df.to_csv(save_path, index=False)
            print(f"Saved raw wide-format data to {save_path}")

            # 转换为长格式用于训练 (兼容旧代码)
            # 格式: Phi, d50, sigma, Emix, Temp -> Tau0

            # Part 1
            df1 = df[['Phi_1(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix_1(混合功_J)', 'Temp_1(温度_C)', 'Tau0_1(屈服应力_Pa)']].copy()
            df1.columns = ['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)', 'Tau0(屈服应力_Pa)']

            # Part 2
            df2 = df[['Phi_2(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix_2(混合功_J)', 'Temp_2(温度_C)', 'Tau0_2(屈服应力_Pa)']].copy()
            df2.columns = ['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)', 'Tau0(屈服应力_Pa)']

            # 合并
            df_long = pd.concat([df1, df2], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

            # 划分训练/测试
            n_train = int(len(df_long) * 0.8)
            train_df = df_long.iloc[:n_train]
            test_df = df_long.iloc[n_train:]

            train_path = os.path.join(os.path.dirname(save_path), "train_data.csv")
            test_path = os.path.join(os.path.dirname(save_path), "test_data.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(f"Saved Train (Long Format): {len(train_df)} samples -> {train_path}")
            print(f"Saved Test (Long Format):  {len(test_df)} samples -> {test_path}")

        return df

if __name__ == "__main__":
    gen = ProcessBasedGenerator()
    gen.generate(n_samples=2000)

