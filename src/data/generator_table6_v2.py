"""
数据生成器 v2 - 严格对齐论文公式

论文: Lian et al., Materials 2025, 18, 2983
公式 (Eq.4): τ = m1 * φ³ / [φ_max(SP) * (φ_max(SP) - φ)]

输入变量:
  φ       : 固含量 (可测量)
  SP%     : 增塑剂掺量 (可测量)

中间变量 (神经网络预测):
  φ_max   : 最大堆积分数 (不可直接测量, 由 SP 决定)

固定参数 (论文 Table 6 标定):
  m1 = 0.72 Pa  (对 Table 6 体系固定)

生成策略:
  1. 在 Table 6 参数范围内采样 φ 和 SP
  2. 用论文反推的线性关系计算 φ_max(SP)，加小扰动模拟批次差异
  3. 用论文公式计算 τ₀
  4. 过滤保留 τ₀ 在 Table 6 范围内的样本
"""

import os
import numpy as np
import pandas as pd


def phi_max_from_sp(sp):
    """
    φ_max 与 SP 掺量的关系 (从论文 Table 6 反推线性拟合)
    SP=0.4% → φ_max≈0.582, SP=0.6% → φ_max≈0.684, SP=1.0% → φ_max≈0.877
    拟合: φ_max = 0.44 + 0.44 * SP
    限制: [0.55, 0.92]
    """
    return np.clip(0.44 + 0.44 * sp, 0.55, 0.92)


def tau0_paper(phi, phi_max, m1=0.72):
    """
    论文 Eq.4 屈服应力公式
    τ = m1 * φ³ / [φ_max * (φ_max - φ)]
    """
    denom = phi_max * (phi_max - phi)
    if denom <= 1e-8:
        return np.nan
    return m1 * phi**3 / denom


def generate(n_target=2000, save_dir="data/synthetic_table6_v2", seed=42):
    np.random.seed(seed)

    print("=" * 60)
    print("数据生成 v2 - 严格对齐论文公式")
    print("=" * 60)
    print("""
论文公式: τ = m1 * φ³ / [φ_max(SP) * (φ_max(SP) - φ)]
输入特征: [Phi, SP_percent]  (2维)
固定参数: m1 = 0.72 Pa

参数范围 (对齐 Table 6):
  φ:    0.45 - 0.51
  SP%:  0.35 - 1.10
  目标 τ₀: 0.10 - 2.50 Pa
""")

    records = []
    attempts = 0

    while len(records) < n_target and attempts < n_target * 20:
        attempts += 1

        # 采样 φ 和 SP
        phi = np.random.uniform(0.45, 0.51)
        sp  = np.random.uniform(0.35, 1.10)

        # 计算 φ_max，加 ±3% 扰动模拟批次差异
        phi_max = phi_max_from_sp(sp) * np.random.uniform(0.97, 1.03)

        # 物理约束: φ_max 必须大于 φ
        if phi_max <= phi + 0.01:
            continue

        # 用论文公式计算 τ₀ (m1 固定 0.72 Pa)
        tau0 = tau0_paper(phi, phi_max, m1=0.72)

        # 过滤范围
        if tau0 is np.nan or not (0.10 <= tau0 <= 2.50):
            continue

        records.append({
            'Phi':        round(phi, 4),
            'SP_percent': round(sp, 4),
            'phi_max':    round(phi_max, 4),   # 中间量，仅用于验证
            'Tau0_Pa':    round(tau0, 4),
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = int(len(df) * 0.8)
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/dataset.csv",    index=False)
    df_train.to_csv(f"{save_dir}/train_data.csv", index=False)
    df_test.to_csv(f"{save_dir}/test_data.csv",   index=False)

    print(f"有效样本: {len(df)}  (尝试 {attempts} 次)")
    print(f"\n生成数据统计:")
    print(f"  Phi:  {df.Phi.min():.3f} – {df.Phi.max():.3f}  (均值 {df.Phi.mean():.3f})")
    print(f"  SP%:  {df.SP_percent.min():.2f} – {df.SP_percent.max():.2f}")
    print(f"  τ₀:   {df.Tau0_Pa.min():.3f} – {df.Tau0_Pa.max():.3f} Pa  (均值 {df.Tau0_Pa.mean():.3f})")
    print(f"\n对比 Table 6 真实数据:")
    print(f"  Phi:  0.458 – 0.504  (均值 0.486)")
    print(f"  SP%:  0.40  – 1.00")
    print(f"  τ₀:   0.19  – 1.95 Pa (均值 0.73 Pa)")
    print(f"\n已保存至 {save_dir}/")
    print("=" * 60)
    return df


if __name__ == "__main__":
    generate(n_target=2000)
