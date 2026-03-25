"""
整理高保真度数据文件
来源: 论文 Table 3 (纯水泥) 和 Table 6 (含增塑剂)
输出列: Phi, d50_um, sigma, SP_percent, Tau0_Pa, Source

关于 sigma (几何标准差) 的计算:
  Log-Normal 分布中，几何标准差 σ_g 定义为:
    σ_g = exp(std(ln(d)))
  由 D10, D50, D90 估算:
    ln(σ_g) = (ln(D90) - ln(D10)) / (2 * 1.282)
            = ln(D90/D10) / 2.564
  其中 1.282 是标准正态分布 90% 分位点

论文测量值 (Figure 1-4):
  水泥 C:    D10=2.844, D50=17.86, D90=47.15  → σ_g = exp(ln(47.15/2.844)/2.564) ≈ 2.99
  粉煤灰 FA: D10=2.333, D50=11.41, D90=37.16  → σ_g = exp(ln(37.16/2.333)/2.564) ≈ 2.94

注意: 生成器中 sigma 范围 1.5-1.9 是错误的，真实值约 2.9-3.0
这里使用论文真实值。
"""

import math
import numpy as np
import pandas as pd

# ── 粒径参数计算 ──────────────────────────────────────────
def calc_sigma_g(d10, d90):
    """由 D10, D90 计算几何标准差"""
    return math.exp(math.log(d90 / d10) / 2.564)

# 水泥
sigma_cement = calc_sigma_g(d10=2.844, d90=47.15)
d50_cement   = 17.86

# 粉煤灰
sigma_fa = calc_sigma_g(d10=2.333, d90=37.16)
d50_fa   = 11.41

# 混合体系: 80 vol.% 水泥 + 20 vol.% 粉煤灰 (Table 6)
d50_mix   = 0.80 * d50_cement + 0.20 * d50_fa
sigma_mix = 0.80 * sigma_cement + 0.20 * sigma_fa

print(f"水泥:   d50={d50_cement:.2f} μm, sigma={sigma_cement:.4f}")
print(f"粉煤灰: d50={d50_fa:.2f} μm, sigma={sigma_fa:.4f}")
print(f"混合:   d50={d50_mix:.2f} μm, sigma={sigma_mix:.4f}")

# ── Table 3: 纯水泥，5个真实样本 ─────────────────────────
table3 = pd.DataFrame({
    'Source':      ['Table3'] * 5,
    'Phi':         [0.527, 0.488, 0.455, 0.426, 0.401],
    'd50_um':      [round(d50_cement, 3)] * 5,
    'sigma':       [round(sigma_cement, 4)] * 5,
    'SP_percent':  [0.0] * 5,
    'Tau0_Pa':     [25.8, 8.7, 4.1, 2.3, 1.6],
})

# ── Table 6: 含增塑剂，16个真实样本 ──────────────────────
table6 = pd.DataFrame({
    'Source':     ['Table6'] * 16,
    'Phi':        [0.458, 0.459, 0.459, 0.502, 0.503, 0.503, 0.504,
                   0.478, 0.479, 0.503, 0.504, 0.479, 0.479, 0.504, 0.479, 0.479],
    'd50_um':     [round(d50_mix, 3)] * 16,
    'sigma':      [round(sigma_mix, 4)] * 16,
    'SP_percent': [0.80, 0.90, 1.00, 0.40, 0.50, 0.60, 0.60,
                   0.40, 0.50, 0.50, 0.70, 0.40, 0.50, 0.60, 0.50, 0.60],
    'Tau0_Pa':    [0.35, 0.23, 0.19, 1.95, 0.97, 0.74, 0.67,
                   1.14, 0.66, 1.29, 0.39, 0.86, 0.67, 0.44, 0.60, 0.46],
})

df = pd.concat([table3, table6], ignore_index=True)

print(f"\n合并后: {len(df)} 样本")
print(f"  Phi:  {df.Phi.min():.3f} – {df.Phi.max():.3f}")
print(f"  d50:  {df.d50_um.min():.2f} – {df.d50_um.max():.2f} μm")
print(f"  sigma:{df.sigma.min():.4f} – {df.sigma.max():.4f}")
print(f"  SP%:  {df.SP_percent.min():.2f} – {df.SP_percent.max():.2f}")
print(f"  Tau0: {df.Tau0_Pa.min():.3f} – {df.Tau0_Pa.max():.3f} Pa")
print(f"\n按 Source 统计:")
print(df.groupby('Source')['Tau0_Pa'].describe().round(3))

df.to_csv('data/high_fidelity/hifi_table3_table6.csv', index=False)
print("\n已保存至 data/high_fidelity/hifi_table3_table6.csv")

# 单独保存 Table 6
df6 = df[df.Source == 'Table6'].reset_index(drop=True)
df6.to_csv('data/high_fidelity/hifi_table6_only.csv', index=False)
print("已保存至 data/high_fidelity/hifi_table6_only.csv")
