"""
基于论文 Table 6 参数范围的合成数据生成器

论文: Constitutive Modeling of Rheological Behavior of Cement Paste
      Based on Material Composition (Materials 2025, 18, 2983)

Table 6 实验设置:
  - 材料: 普通硅酸盐水泥 + 20 vol.% 粉煤灰 + PCE 增塑剂
  - φ 范围: 45.8% - 50.4%
  - SP 掺量: 0.4% - 1.0% (by binder weight)
  - Tau0 范围: 0.19 - 1.95 Pa

论文使用的模型公式 (Eq.2 / Eq.4):
  τ = m1 * φ³ / [φ_max(φ_max - φ)]
  注意: 无 φ_c 项, 与 YODEL 原始公式不同

论文标定参数 (Table 6 体系, af 和 m1 固定):
  af = 0.034
  m1 = 0.72 Pa
  φ_max = φ_max(SP)  随增塑剂掺量变化

粒径参数 (论文 Figure 1-2):
  水泥 C:    D50=17.86 μm
  粉煤灰 FA: D50=11.41 μm
  混合体系 (80%C + 20%FA): D50 ≈ 16.6 μm
"""

import os
import numpy as np
import pandas as pd


def tau0_formula(phi, phi_max, m1):
    """
    论文 Eq.2/Eq.4 的屈服应力公式
    τ = m1 * φ³ / [φ_max * (φ_max - φ)]

    Args:
        phi: 固含量 (scalar)
        phi_max: 最大堆积分数 (scalar)
        m1: 屈服应力系数 Pa (scalar)
    Returns:
        tau0 (Pa)
    """
    denom = phi_max * (phi_max - phi)
    if denom <= 0:
        return np.nan
    return m1 * phi**3 / denom


def generate_table6_aligned(n_target=2000, save_dir="data/synthetic_table6", seed=42):
    """
    生成与论文 Table 6 参数范围对齐的合成数据

    生成策略:
      1. 在 Table 6 的参数范围内随机采样 φ 和 SP
      2. 根据 SP 掌量计算 φ_max (线性关系, 从论文反推)
      3. 用论文公式计算 Tau0
      4. 对 m1 和 φ_max 加入小幅随机扰动, 模拟材料批次差异
      5. 过滤保留 Tau0 在合理范围内的样本
    """
    np.random.seed(seed)

    print("=" * 60)
    print("生成与 Table 6 对齐的合成数据")
    print("=" * 60)
    print(f"\n目标样本数: {n_target}")
    print("""
参数范围 (对齐 Table 6):
  φ:      0.45 - 0.51
  SP%:    0.35 - 1.10
  d50:    14.0 - 19.0 μm  (80%水泥+20%FA混合体系)
  sigma:  1.5  - 1.9
  m1:     0.60 - 0.85 Pa  (论文值 0.72, 加入扰动)
  φ_max:  由 SP 决定 + 扰动
  目标 Tau0: 0.1 - 2.5 Pa
""")

    # φ_max 与 SP 的关系 (从 Table 6 反推的线性拟合)
    # SP=0.4% → φ_max≈0.59, SP=0.6% → φ_max≈0.70, SP=1.0% → φ_max≈0.88
    # 线性: φ_max = 0.44 + 0.44 * SP
    # 但限制在物理合理范围 [0.55, 0.95]
    def phi_max_from_sp(sp_percent):
        return np.clip(0.44 + 0.44 * sp_percent, 0.55, 0.92)

    data_list = []
    n_generated = 0
    n_attempts = 0
    max_attempts = n_target * 50

    while n_generated < n_target and n_attempts < max_attempts:
        n_attempts += 1

        # 1. 采样 φ (固含量)
        phi = np.random.uniform(0.45, 0.51)

        # 2. 采样 SP 掺量
        sp = np.random.uniform(0.35, 1.10)

        # 3. 计算 φ_max (加入 ±3% 随机扰动模拟批次差异)
        phi_max_base = phi_max_from_sp(sp)
        phi_max = phi_max_base * np.random.uniform(0.97, 1.03)

        # 物理约束: φ_max 必须大于 φ
        if phi_max <= phi + 0.01:
            continue

        # 4. 采样 m1 (论文值 0.72 Pa, 加入扰动模拟材料差异)
        m1 = np.random.uniform(0.60, 0.85)

        # 5. 用论文公式计算 Tau0
        tau0 = tau0_formula(phi, phi_max, m1)

        # 6. 过滤: 保留 Tau0 在 Table 6 范围内 (0.1-2.5 Pa)
        if tau0 is np.nan or tau0 < 0.10 or tau0 > 2.5:
            continue

        # 7. 采样粒径参数 (混合粉体: 80%水泥+20%FA)
        d50 = np.random.uniform(14.0, 19.0)
        sigma = np.random.uniform(1.5, 1.9)

        data_list.append({
            'Phi': round(phi, 4),
            'd50_um': round(d50, 3),
            'sigma': round(sigma, 4),
            'SP_percent': round(sp, 3),
            'phi_max': round(phi_max, 4),
            'm1_Pa': round(m1, 4),
            'Tau0_Pa': round(tau0, 4),
        })
        n_generated += 1

    df = pd.DataFrame(data_list)

    print(f"生成尝试次数: {n_attempts:,}")
    print(f"有效样本数: {len(df):,}")
    print(f"过滤率: {len(df)/n_attempts*100:.1f}%")

    print(f"\n生成数据统计:")
    print(f"  φ:      {df['Phi'].min():.3f} - {df['Phi'].max():.3f}  (均值 {df['Phi'].mean():.3f})")
    print(f"  SP%:    {df['SP_percent'].min():.2f} - {df['SP_percent'].max():.2f}")
    print(f"  d50:    {df['d50_um'].min():.1f} - {df['d50_um'].max():.1f} μm")
    print(f"  φ_max:  {df['phi_max'].min():.3f} - {df['phi_max'].max():.3f}")
    print(f"  m1:     {df['m1_Pa'].min():.3f} - {df['m1_Pa'].max():.3f} Pa")
    print(f"  Tau0:   {df['Tau0_Pa'].min():.3f} - {df['Tau0_Pa'].max():.3f} Pa  (均值 {df['Tau0_Pa'].mean():.3f} Pa)")

    print(f"\n对比 Table 6 真实数据:")
    print(f"  φ:     0.458 - 0.504  (均值 0.486)")
    print(f"  Tau0:  0.19  - 1.95 Pa (均值 0.73 Pa)")

    # 保存
    os.makedirs(save_dir, exist_ok=True)

    # 训练/测试划分 8:2
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * 0.8)
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    df.to_csv(os.path.join(save_dir, "dataset.csv"), index=False)
    df_train.to_csv(os.path.join(save_dir, "train_data.csv"), index=False)
    df_test.to_csv(os.path.join(save_dir, "test_data.csv"), index=False)

    print(f"\n已保存:")
    print(f"  全量: {save_dir}/dataset.csv  ({len(df)} 样本)")
    print(f"  训练: {save_dir}/train_data.csv  ({len(df_train)} 样本)")
    print(f"  测试: {save_dir}/test_data.csv  ({len(df_test)} 样本)")
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = generate_table6_aligned(n_target=2000)
