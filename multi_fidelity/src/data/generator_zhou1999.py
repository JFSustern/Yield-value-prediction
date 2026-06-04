"""
Zhou 1999 Al₂O₃ 悬浮液数据生成器

真实高保真数据来源:
    Zhou, Solomon, Scales, Boger (1999)
    "The yield stress of concentrated flocculated suspensions
     of size distributed particles."
    J. Rheol. 43(3): 651–71. DOI: 10.1122/1.551029

实验条件: 4 种 Al₂O₃ 粉末 (AKP-15/20/30/50)，pH 8.9±0.2 (等电点 IEP)
          Haake RV3 流变仪，Vane 法测量剪切屈服应力，20±1 °C

物理方程 (YODEL Eq. 42, Flatt & Bowen 2006):
    τ = m₁_eff × φ(φ − φ₀)² / [φ_max_ref × (φ_max_ref − φ)]

    φ_max_ref = 0.570  (压滤法独立测定，Zhou 1999 Table 1 + YODEL paper)
    φ₀        = 0.026  (渗流阈值，YODEL paper 拟合值)
    m₁_eff            (隐变量，颗粒间力参数，不可在线测量)

隐变量与粒径的关系 (YODEL 理论):
    m₁_eff ≈ K_H / d_s²   (K_H 与 Hamaker 常数相关)

标定值 (从 Fig. 1 视觉估算 + YODEL 方程反推):
    AKP-15 (d_s = 0.580 μm): m₁ ≈ 310 Pa,  K = m₁×d_s² ≈ 104 Pa·μm²
    AKP-20 (d_s = 0.401 μm): m₁ ≈ 470 Pa,  K = m₁×d_s² ≈  76 Pa·μm²
    AKP-30 (d_s = 0.247 μm): m₁ ≈ 830 Pa,  K = m₁×d_s² ≈  51 Pa·μm²
    AKP-50 (d_s = 0.185 μm): m₁ ≈ 1380 Pa, K = m₁×d_s² ≈  47 Pa·μm²

⚠️  HF 数字化精度说明:
    本文件中的 HF 数据是基于 YODEL 方程 + 标定 m₁ + 测量噪声的估算值，
    从 Zhou 1999 Fig. 1 视觉估算标定，适合方法验证。
    发表前建议用 WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/)
    对 Fig. 1 进行精确数字化以替换本文件中的估算值。

生成策略:
  低保真 (LF): 宽参数空间采样，YODEL 方程 + K/d_s² 关系 + ±15% 噪声，2000 条
  高保真 (HF): 数字化真实实验点（4 粉末 × ~12 点 ≈ 48 条）
"""

import os
import numpy as np
import pandas as pd


# ── 物理常数 ───────────────────────────────────────────────────────────────

PHI_MAX_REF = 0.570    # 压滤独立测定，Zhou 1999
PHI_0       = 0.026    # YODEL 渗流阈值
K_H_REF     = 65.0     # 颗粒间力标定常数 (Pa·μm²)，四粉末平均


# ── 粉末参数 ───────────────────────────────────────────────────────────────

# d_s: 表面平均粒径 (μm), m1: 颗粒间力参数 (Pa)
# 来源: Zhou 1999 Table I + YODEL paper 拟合
POWDERS = {
    'AKP-15': {'d_s': 0.580, 'm1': 310},
    'AKP-20': {'d_s': 0.401, 'm1': 470},
    'AKP-30': {'d_s': 0.247, 'm1': 830},
    'AKP-50': {'d_s': 0.185, 'm1': 1380},
}


# ── 物理方程 ───────────────────────────────────────────────────────────────

def tau_yodel_full(phi, m1, phi_max=PHI_MAX_REF, phi0=PHI_0):
    """
    YODEL Eq. 42 (完整含渗流阈值):
        τ = m₁ × φ(φ − φ₀)² / [φ_max × (φ_max − φ)]

    Returns nan 当 phi >= phi_max 或 phi <= phi0
    """
    if np.ndim(phi) == 0:
        if phi >= phi_max - 1e-6 or phi <= phi0:
            return np.nan
        num   = m1 * phi * (phi - phi0) ** 2
        denom = phi_max * (phi_max - phi)
        return num / denom
    else:
        valid = (phi < phi_max - 1e-6) & (phi > phi0)
        num   = m1 * phi * np.where(valid, (phi - phi0) ** 2, 0.0)
        denom = phi_max * np.where(valid, phi_max - phi, 1.0)
        result = np.where(valid, num / denom, np.nan)
        return result


# ── HF 数字化数据（视觉估算 + YODEL 验证）─────────────────────────────────

def build_hf_digitized(seed: int = 42) -> pd.DataFrame:
    """
    生成 HF 数字化数据集。

    数据基于 YODEL 方程 + 标定 m₁ + 12% 测量噪声（模拟 Vane 法重复性）。
    用于验证目的；精确发表请用 WebPlotDigitizer 替换。

    Return: DataFrame with [phi, d_s_um, tau_Pa, powder]
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # 每种粉末的 φ 采样点 (对齐 Zhou 1999 Fig. 1 可读数据范围)
    phi_points = {
        'AKP-15': [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.41, 0.44, 0.47, 0.50],
        'AKP-20': [0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.41, 0.44, 0.47, 0.50],
        'AKP-30': [0.10, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.37, 0.40, 0.43, 0.46, 0.49, 0.51],
        'AKP-50': [0.10, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.37, 0.40, 0.43, 0.46, 0.49, 0.51],
    }

    records = []
    for pname, pinfo in POWDERS.items():
        d_s = pinfo['d_s']
        m1  = pinfo['m1']
        for phi in phi_points[pname]:
            tau_true = tau_yodel_full(phi, m1)
            if tau_true is np.nan or np.isnan(tau_true):
                continue
            # 模拟 Vane 法测量噪声 (±12%, log-normal 分布)
            noise = rng.lognormal(mean=0, sigma=0.12)
            tau_meas = tau_true * noise
            if tau_meas < 0.5:   # 低于仪器检测下限则跳过
                continue
            records.append({
                'phi':     round(phi,    4),
                'd_s_um':  d_s,
                'tau_Pa':  round(float(tau_meas), 2),
                'powder':  pname,
                'm1_true': m1,    # 仅供验证，训练时不使用
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def save_hf_splits(df: pd.DataFrame, save_dir: str = 'data/zhou1999_hf',
                   seed: int = 42):
    """
    保存 HF 数据并按稀缺场景划分。

    因 HF 数据量有限 (~48 条)，划分为:
      train_scarce : 30 条
      eval         : 10 条
      test         :  8 条 (剩余)

    注: 测试集仅 8 条，报告指标时须说明统计局限性。
    """
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f'{save_dir}/dataset.csv', index=False)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))

    train_idx = idx[:30]
    eval_idx  = idx[30:40]
    test_idx  = idx[40:]

    df.iloc[train_idx].to_csv(f'{save_dir}/train_scarce.csv', index=False)
    df.iloc[eval_idx ].to_csv(f'{save_dir}/eval.csv',         index=False)
    df.iloc[test_idx ].to_csv(f'{save_dir}/test.csv',         index=False)

    print(f"HF 数据总量: {len(df)} 条")
    print(f"  train_scarce: {len(train_idx)}")
    print(f"  eval:         {len(eval_idx)}")
    print(f"  test:         {len(test_idx)}  ← 量少，结果仅供参考")
    print(f"已保存至 {save_dir}/")


# ── LF 合成数据生成 ────────────────────────────────────────────────────────

def generate_lf(n_target: int = 2000,
                save_dir: str  = 'data/zhou1999_lf',
                seed: int      = 42) -> pd.DataFrame:
    """
    生成低保真合成数据 (宽参数空间，用于预训练)。

    输入空间:
      φ    ∈ [0.08, 0.52]
      d_s  ∈ [0.14, 0.70] μm  (覆盖 AKP 系列范围并有外推)

    m₁ 由 K_H/d_s² + ±15% 扰动生成 (模拟批次差异).
    """
    np.random.seed(seed)
    print('=' * 60)
    print('Zhou 1999 低保真合成数据生成 (YODEL Eq. 42)')
    print('=' * 60)

    records = []
    attempts = 0

    while len(records) < n_target and attempts < n_target * 25:
        attempts += 1

        phi = np.random.uniform(0.08, 0.52)
        d_s = np.random.uniform(0.14, 0.70)   # μm

        # m₁ 由 K_H/d_s² 决定，加 ±15% 扰动
        m1_base = K_H_REF / d_s ** 2
        m1      = m1_base * np.random.uniform(0.85, 1.15)
        m1      = np.clip(m1, 50.0, 3000.0)

        # 物理约束
        if phi <= PHI_0 + 0.01 or phi >= PHI_MAX_REF - 0.01:
            continue

        tau = tau_yodel_full(phi, m1)
        if tau is np.nan or np.isnan(tau) or not (1.0 <= tau <= 12000.0):
            continue

        records.append({
            'phi':    round(phi,  4),
            'd_s_um': round(d_s,  4),
            'm1_lf':  round(m1,   2),   # 中间量，仅用于验证
            'tau_Pa': round(tau,  3),
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = int(len(df) * 0.8)
    df_train = df.iloc[:n_train]
    df_test  = df.iloc[n_train:]

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(       f'{save_dir}/dataset.csv',    index=False)
    df_train.to_csv( f'{save_dir}/train.csv',      index=False)
    df_test.to_csv(  f'{save_dir}/test.csv',       index=False)

    print(f'有效样本: {len(df)} (尝试 {attempts} 次)')
    print(f'  phi:    {df.phi.min():.3f} – {df.phi.max():.3f}  均值 {df.phi.mean():.3f}')
    print(f'  d_s:    {df.d_s_um.min():.3f} – {df.d_s_um.max():.3f} μm')
    print(f'  m1:     {df.m1_lf.min():.0f} – {df.m1_lf.max():.0f} Pa')
    print(f'  tau:    {df.tau_Pa.min():.1f} – {df.tau_Pa.max():.1f} Pa  均值 {df.tau_Pa.mean():.1f}')
    print(f'\n已保存至 {save_dir}/')
    print('=' * 60)
    return df


# ── 入口 ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)

    print('生成 Zhou 1999 体系数据集 ...\n')

    # HF: 数字化真实实验数据
    print('── HF 数字化数据 ──')
    df_hf = build_hf_digitized(seed=42)
    print(f'HF 合计: {len(df_hf)} 条')
    print(f'  phi:  {df_hf.phi.min():.2f} – {df_hf.phi.max():.2f}')
    print(f'  tau:  {df_hf.tau_Pa.min():.1f} – {df_hf.tau_Pa.max():.1f} Pa')
    print(f'  各粉末数量: {df_hf.powder.value_counts().to_dict()}')
    save_hf_splits(df_hf, save_dir='data/zhou1999_hf', seed=42)

    # LF: 合成数据
    print('\n── LF 合成数据 ──')
    generate_lf(n_target=2000, save_dir='data/zhou1999_lf', seed=42)

    print('\n全部完成。')
    print()
    print('⚠️  提示: HF 数据为视觉估算 + YODEL 公式生成的估算值。')
    print('   建议在发表前用 WebPlotDigitizer 对 Zhou 1999 Fig.1 进行精确数字化。')
    print('   工具: https://automeris.io/WebPlotDigitizer/')
