"""
YODEL 陶瓷悬浮液数据生成器

参考:  Flatt & Bowen (2006) YODEL, J. Am. Ceram. Soc. 89(4): 1244–1256
体系:  Al₂O₃ 无机陶瓷悬浮液（水溶性分散剂体系）
用途:  PI-MFNN 跨体系泛化验证实验（第二验证体系）

---------------------------------------------------------------------------
物理方程 (YODEL 简化形式):
    τ₀ = m_y * φ² / [φ_max(c_d) * (φ_max(c_d) − φ)²]

    m_y = 0.12 Pa   (陶瓷体系标定常数)
    φ   : 固体体积分数       (可直接测量)
    c_d : 分散剂掺量 %       (可直接测量)
    φ_max: 有效最大堆积分数  (隐变量，由 c_d 决定，NPE 反演)

与 Lian 2025 水泥浆方程的对比:
    Lian:  τ₀ = 0.72 * φ³ / [φ_max * (φ_max − φ)]    (线性分母,立方固含量)
    YODEL: τ₀ = 0.12 * φ² / [φ_max * (φ_max − φ)²]   (平方分母,平方固含量)
---------------------------------------------------------------------------

生成策略:
  低保真 (LF):  宽参数空间均匀采样，2000 条
  高保真 (HF):  模拟实验操作区间蒙特卡洛采样，400 条
                (对应 φ_max-c_d 关系加入批次噪声，模拟真实测量误差)
"""

import os
import numpy as np
import pandas as pd


# ── 物理方程 ──────────────────────────────────────────────────────────────

M_Y = 0.12   # Pa, YODEL 陶瓷体系标定常数


def phi_max_from_cd(c_d):
    """
    φ_max 与分散剂掺量 c_d 的映射关系 (线性拟合，陶瓷文献参考值)

    c_d = 0.20% → φ_max ≈ 0.512
    c_d = 0.80% → φ_max ≈ 0.608
    c_d = 1.50% → φ_max ≈ 0.720

    拟合: φ_max = 0.480 + 0.160 * c_d
    物理限制: [0.50, 0.75]
    """
    return np.clip(0.480 + 0.160 * c_d, 0.50, 0.75)


def tau0_yodel(phi, phi_max, m_y=M_Y):
    """
    YODEL 屈服应力方程:
        τ₀ = m_y * φ² / [φ_max * (φ_max − φ)²]

    Returns:
        屈服应力 (Pa)，当 phi_max <= phi 时返回 nan
    """
    denom = phi_max * (phi_max - phi) ** 2
    if np.ndim(phi) == 0:  # scalar
        if denom <= 1e-10:
            return np.nan
        return m_y * phi**2 / denom
    else:
        result = np.where(denom > 1e-10, m_y * phi**2 / denom, np.nan)
        return result


# ── 数据生成函数 ──────────────────────────────────────────────────────────

def generate_lf(n_target: int = 2000,
                save_dir: str = "data/yodel_lf",
                seed: int = 42) -> pd.DataFrame:
    """
    生成低保真合成数据 (宽参数空间，用于预训练)

    参数空间:
      φ     ∈ [0.30, 0.48]
      c_d   ∈ [0.20, 1.50]%
      φ_max ∈ phi_max_from_cd(c_d) × U(0.97, 1.03)  (±3% 批次扰动)
      τ₀   ∈ [0.08, 3.50] Pa  (过滤范围)
    """
    np.random.seed(seed)

    print("=" * 60)
    print("YODEL 低保真合成数据生成")
    print("=" * 60)
    print(f"物理方程: τ₀ = {M_Y} * φ² / [φ_max * (φ_max − φ)²]")
    print(f"参数空间: φ ∈ [0.30, 0.48], c_d ∈ [0.20, 1.50]%")
    print()

    records = []
    attempts = 0

    while len(records) < n_target and attempts < n_target * 20:
        attempts += 1

        phi = np.random.uniform(0.30, 0.48)
        c_d = np.random.uniform(0.20, 1.50)

        # φ_max: 由 c_d 决定的基线 + ±3% 批次扰动
        phi_max = phi_max_from_cd(c_d) * np.random.uniform(0.97, 1.03)

        # 物理约束
        if phi_max <= phi + 0.02:
            continue

        tau0 = tau0_yodel(phi, phi_max)

        if tau0 is np.nan or not (0.08 <= tau0 <= 3.50):
            continue

        records.append({
            'phi':     round(phi,     4),
            'c_d':     round(c_d,     4),
            'phi_max': round(phi_max, 4),   # 中间量，仅用于验证
            'tau0_Pa': round(tau0,    4),
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = int(len(df) * 0.8)
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/dataset.csv",       index=False)
    df_train.to_csv(f"{save_dir}/train.csv",   index=False)
    df_test.to_csv(f"{save_dir}/test.csv",     index=False)

    print(f"有效样本: {len(df)} (尝试 {attempts} 次)")
    print(f"  phi:    {df.phi.min():.3f} – {df.phi.max():.3f}  均值 {df.phi.mean():.3f}")
    print(f"  c_d:    {df.c_d.min():.2f} – {df.c_d.max():.2f}%")
    print(f"  phi_max:{df.phi_max.min():.3f} – {df.phi_max.max():.3f}")
    print(f"  tau0:   {df.tau0_Pa.min():.3f} – {df.tau0_Pa.max():.3f} Pa  "
          f"均值 {df.tau0_Pa.mean():.3f}")
    print(f"\n已保存至 {save_dir}/")
    print("=" * 60)
    return df


def generate_hf(n_total: int = 400,
                save_dir: str = "data/yodel_hf",
                seed: int = 42) -> pd.DataFrame:
    """
    生成高保真代理数据 (限制于实验操作区间，模拟真实陶瓷实验)

    模拟设置:
      - 限制于文献报告的实验操作区间
      - φ_max 使用基线关系 + 略小扰动 (±1.5%) 模拟更精确的表征
      - τ₀ 范围缩窄至 0.10–2.50 Pa

    数据划分 (与 Lian 2025 体系保持一致):
      train (稀缺场景): 30 条   (模拟高成本测量稀缺)
      eval:             10 条   (早停验证)
      test:            360 条   (独立测试集)

    另有数据充足场景划分:
      train (充足场景): 320 条 (80%)
      eval+test:         80 条
    """
    np.random.seed(seed)

    print("=" * 60)
    print("YODEL 高保真代理数据生成 (实验操作区间)")
    print("=" * 60)

    records = []
    attempts = 0

    while len(records) < n_total and attempts < n_total * 30:
        attempts += 1

        # 实验操作区间 (对应陶瓷悬浮液典型实验范围)
        phi = np.random.uniform(0.33, 0.46)
        c_d = np.random.uniform(0.30, 1.20)

        # 高保真数据: 批次扰动更小 (±1.5%)
        phi_max = phi_max_from_cd(c_d) * np.random.uniform(0.985, 1.015)

        if phi_max <= phi + 0.05:
            continue

        tau0 = tau0_yodel(phi, phi_max)

        # 高保真数据: τ₀ 范围更窄 (过滤极端值)
        if tau0 is np.nan or not (0.10 <= tau0 <= 2.50):
            continue

        records.append({
            'phi':     round(phi,     4),
            'c_d':     round(c_d,     4),
            'phi_max': round(phi_max, 4),
            'tau0_Pa': round(tau0,    4),
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── 稀缺场景划分 (主实验, 与 Lian 2025 对称) ──
    idx_train_scarce = df.index[:30]
    idx_eval         = df.index[30:40]
    idx_test         = df.index[40:]

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/dataset.csv",                  index=False)
    df.iloc[idx_train_scarce].to_csv(f"{save_dir}/train_scarce.csv",  index=False)
    df.iloc[idx_eval        ].to_csv(f"{save_dir}/eval.csv",          index=False)
    df.iloc[idx_test        ].to_csv(f"{save_dir}/test.csv",          index=False)

    # ── 充足场景划分 ──
    n_rich = min(400, len(df))
    df_rich = df.iloc[:n_rich]
    df_rich.to_csv(f"{save_dir}/train_rich.csv", index=False)

    print(f"有效样本: {len(df)} (尝试 {attempts} 次)")
    print(f"  phi:    {df.phi.min():.3f} – {df.phi.max():.3f}  均值 {df.phi.mean():.3f}")
    print(f"  c_d:    {df.c_d.min():.2f} – {df.c_d.max():.2f}%")
    print(f"  phi_max:{df.phi_max.min():.3f} – {df.phi_max.max():.3f}")
    print(f"  tau0:   {df.tau0_Pa.min():.3f} – {df.tau0_Pa.max():.3f} Pa  "
          f"均值 {df.tau0_Pa.mean():.3f}")
    print(f"\n数据划分 (稀缺场景): train=30, eval=10, test={len(df)-40}")
    print(f"已保存至 {save_dir}/")
    print("=" * 60)
    return df


if __name__ == "__main__":
    import sys
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)

    print("生成 YODEL 两级数据集 ...")
    generate_lf(n_target=2000, save_dir="data/yodel_lf", seed=42)
    print()
    generate_hf(n_total=400,   save_dir="data/yodel_hf", seed=42)
    print("\n全部完成。")
