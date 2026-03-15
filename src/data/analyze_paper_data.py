"""
高保真度数据分析脚本
分析论文提取的水泥浆料数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    data_path = '/Users/sunjifei/Desktop/work/Project/data/high_fidelity/cement_paste_data.csv'
    df = pd.read_csv(data_path)
    print(f"加载数据: {len(df)} 样本")
    return df


def analyze_distribution(df):
    """分析数据分布"""
    print("\n" + "="*60)
    print("数据分布分析")
    print("="*60)

    # 基本统计
    print("\n特征统计:")
    stats = df[['Phi', 'd50_um', 'sigma', 'Emix_J', 'Tau0_Pa']].describe()
    print(stats)

    # 相关性分析
    print("\n相关性矩阵:")
    corr = df[['Phi', 'd50_um', 'sigma', 'Emix_J', 'Tau0_Pa']].corr()
    print(corr)

    return stats, corr


def compare_with_synthetic(df_paper):
    """对比论文数据与合成数据"""
    print("\n" + "="*60)
    print("论文数据 vs 合成数据对比")
    print("="*60)

    # 尝试加载合成数据
    synthetic_path = '/Users/sunjifei/Desktop/work/Project/data/synthetic/dataset.csv'
    if Path(synthetic_path).exists():
        df_synthetic = pd.read_csv(synthetic_path)
        print(f"\n合成数据: {len(df_synthetic)} 样本")

        # 对比统计 (使用中文列名)
        comparison = pd.DataFrame({
            '特征': ['Phi', 'd50_um', 'sigma', 'Tau0_Pa'],
            '论文_最小': [
                df_paper['Phi'].min(),
                df_paper['d50_um'].min(),
                df_paper['sigma'].min(),
                df_paper['Tau0_Pa'].min()
            ],
            '论文_最大': [
                df_paper['Phi'].max(),
                df_paper['d50_um'].max(),
                df_paper['sigma'].max(),
                df_paper['Tau0_Pa'].max()
            ],
            '论文_均值': [
                df_paper['Phi'].mean(),
                df_paper['d50_um'].mean(),
                df_paper['sigma'].mean(),
                df_paper['Tau0_Pa'].mean()
            ],
            '合成_最小': [
                df_synthetic['Phi_2(固含量)'].min(),
                df_synthetic['d50(中位径_um)'].min(),
                df_synthetic['sigma(几何标准差)'].min(),
                df_synthetic['Tau0_2(屈服应力_Pa)'].min()
            ],
            '合成_最大': [
                df_synthetic['Phi_2(固含量)'].max(),
                df_synthetic['d50(中位径_um)'].max(),
                df_synthetic['sigma(几何标准差)'].max(),
                df_synthetic['Tau0_2(屈服应力_Pa)'].max()
            ],
            '合成_均值': [
                df_synthetic['Phi_2(固含量)'].mean(),
                df_synthetic['d50(中位径_um)'].mean(),
                df_synthetic['sigma(几何标准差)'].mean(),
                df_synthetic['Tau0_2(屈服应力_Pa)'].mean()
            ]
        })
        print("\n对比表:")
        print(comparison.to_string(index=False))

        return df_synthetic, comparison
    else:
        print("\n未找到合成数据文件")
        return None, None


def plot_analysis(df, df_synthetic=None):
    """绘制分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('高保真度数据分析 (论文数据)', fontsize=16, fontweight='bold')

    # 1. 固含量分布
    ax = axes[0, 0]
    ax.hist(df['Phi'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    if df_synthetic is not None:
        ax.hist(df_synthetic['Phi_2(固含量)'], bins=15, alpha=0.5, color='orange', edgecolor='black')
        ax.legend(['Paper Data', 'Synthetic Data'])
    ax.set_xlabel('Phi (Solid Volume Fraction)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Phi Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 2. 粒径分布
    ax = axes[0, 1]
    ax.hist(df['d50_um'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    if df_synthetic is not None:
        ax.hist(df_synthetic['d50(中位径_um)'], bins=15, alpha=0.5, color='orange', edgecolor='black')
    ax.set_xlabel('d50 (μm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Particle Size Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. 屈服应力分布
    ax = axes[0, 2]
    ax.hist(df['Tau0_Pa'], bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    if df_synthetic is not None:
        ax.hist(df_synthetic['Tau0_2(屈服应力_Pa)'], bins=15, alpha=0.5, color='orange', edgecolor='black')
    ax.set_xlabel('Yield Stress (Pa)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Yield Stress Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Phi vs Tau0
    ax = axes[1, 0]
    ax.scatter(df['Phi'], df['Tau0_Pa'], alpha=0.7, s=80, color='steelblue', edgecolor='black')
    if df_synthetic is not None:
        ax.scatter(df_synthetic['Phi_2(固含量)'], df_synthetic['Tau0_2(屈服应力_Pa)'], alpha=0.3, s=30, color='orange')
    ax.set_xlabel('Phi', fontsize=11)
    ax.set_ylabel('Tau0 (Pa)', fontsize=11)
    ax.set_title('Phi vs Yield Stress', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. d50 vs Tau0
    ax = axes[1, 1]
    ax.scatter(df['d50_um'], df['Tau0_Pa'], alpha=0.7, s=80, color='steelblue', edgecolor='black')
    if df_synthetic is not None:
        ax.scatter(df_synthetic['d50(中位径_um)'], df_synthetic['Tau0_2(屈服应力_Pa)'], alpha=0.3, s=30, color='orange')
    ax.set_xlabel('d50 (μm)', fontsize=11)
    ax.set_ylabel('Tau0 (Pa)', fontsize=11)
    ax.set_title('d50 vs Yield Stress', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6. 相关性热图
    ax = axes[1, 2]
    corr = df[['Phi', 'd50_um', 'sigma', 'Emix_J', 'Tau0_Pa']].corr()
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    ax.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

    # 添加相关系数值
    for i in range(len(corr)):
        for j in range(len(corr)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    output_path = '/Users/sunjifei/Desktop/work/Project/data/high_fidelity/data_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    plt.close()


def plot_material_comparison(df):
    """按材料类型对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('不同材料类型对比', fontsize=14, fontweight='bold')

    # 1. 材料类型分布
    ax = axes[0]
    material_counts = df['Material_Type'].value_counts()
    colors = plt.cm.Set3(range(len(material_counts)))
    ax.bar(range(len(material_counts)), material_counts.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(material_counts)))
    ax.set_xticklabels(material_counts.index, rotation=45, ha='right')
    ax.set_ylabel('Sample Count', fontsize=11)
    ax.set_title('Material Type Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. 不同材料的屈服应力分布
    ax = axes[1]
    materials = df['Material_Type'].unique()
    positions = range(len(materials))
    data_to_plot = [df[df['Material_Type'] == mat]['Tau0_Pa'].values for mat in materials]

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'))

    ax.set_xticks(positions)
    ax.set_xticklabels(materials, rotation=45, ha='right')
    ax.set_ylabel('Yield Stress (Pa)', fontsize=11)
    ax.set_title('Yield Stress by Material Type', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = '/Users/sunjifei/Desktop/work/Project/data/high_fidelity/material_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"材料对比图已保存至: {output_path}")
    plt.close()


def generate_report(df, stats, corr):
    """生成分析报告"""
    report = f"""# 论文数据提取与分析报告

## 数据来源
**论文**: Constitutive Modeling of Rheological Behavior of Cement Paste Based on Material Composition (2025)
**期刊**: Materials (MDPI)

## 数据概览

### 样本统计
- **总样本数**: {len(df)}
- **数据来源分布**:
{df['Source'].value_counts().to_string()}

- **材料类型分布**:
{df['Material_Type'].value_counts().to_string()}

### 特征统计

```
{stats.to_string()}
```

### 关键发现

1. **固含量范围**: {df['Phi'].min():.3f} - {df['Phi'].max():.3f}
   - 平均值: {df['Phi'].mean():.3f}
   - 中位数: {df['Phi'].median():.3f}

2. **粒径范围**: {df['d50_um'].min():.2f} - {df['d50_um'].max():.2f} μm
   - 平均值: {df['d50_um'].mean():.2f} μm

3. **屈服应力范围**: {df['Tau0_Pa'].min():.2f} - {df['Tau0_Pa'].max():.2f} Pa
   - 平均值: {df['Tau0_Pa'].mean():.2f} Pa
   - 中位数: {df['Tau0_Pa'].median():.2f} Pa
   - **注意**: 论文数据的屈服应力普遍较低 (< 50 Pa)

### 相关性分析

```
{corr.to_string()}
```

**关键相关性**:
- Phi ↔ Tau0: {corr.loc['Phi', 'Tau0_Pa']:.3f}
- d50 ↔ Tau0: {corr.loc['d50_um', 'Tau0_Pa']:.3f}
- sigma ↔ Tau0: {corr.loc['sigma', 'Tau0_Pa']:.3f}

## 数据质量评估

### 优点
✅ **数据来源可靠**: 来自高质量期刊论文
✅ **参数完整**: 包含 PSD、固含量、屈服应力等关键参数
✅ **材料多样**: 涵盖纯水泥、粉煤灰、矿渣、硅灰等多种材料

### 局限性
⚠️ **样本量有限**: 仅 {len(df)} 个样本
⚠️ **固含量偏低**: Phi < 0.5,而项目目标 Phi > 0.63
⚠️ **屈服应力偏低**: 大部分 < 50 Pa,项目目标 70-120 Pa
⚠️ **材料差异**: 水泥浆料 vs 火箭推进剂

## 与合成数据对比

论文数据与项目合成数据的参数范围差异较大:
- **Phi**: 论文 0.35-0.50 vs 项目 0.60-0.74
- **d50**: 论文 7-15 μm vs 项目 5-50 μm
- **Tau0**: 论文 1-50 Pa vs 项目 0-3500 Pa

## 多保真度学习策略建议

### 阶段 1: 低保真度训练
- 使用合成数据 (~2700 样本)
- 训练完整的 PINN 模型
- 目标: Loss < 20 Pa, R² > 0.8

### 阶段 2: 高保真度微调
- 使用论文数据 ({len(df)} 样本)
- 冻结神经网络层
- 仅微调物理层参数 (Φ_m, G_max)
- 小学习率: 1e-5 ~ 1e-4
- Epochs: 50-100

### 预期效果
- 提升模型在真实数据上的泛化能力
- 验证 YODEL 物理机理的普适性
- 为后续真实推进剂数据做准备

## 风险与缓解

### 风险 1: 参数范围不匹配
**缓解**:
- 在微调时使用较小的学习率
- 监控验证集性能,防止过拟合
- 保留低保真度模型作为 baseline

### 风险 2: 材料物理差异
**缓解**:
- 将微调视为"迁移学习"而非"精确拟合"
- 关注物理参数的合理性而非绝对精度
- 后续获取真实推进剂数据进行验证

## 下一步行动

1. ✅ 数据提取与分析 (已完成)
2. ⏭️ 验证/训练低保真度模型
3. ⏭️ 实现微调脚本
4. ⏭️ 模型评估与对比

---

**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    report_path = '/Users/sunjifei/Desktop/work/Project/docs/论文数据提取报告.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n分析报告已保存至: {report_path}")


def main():
    """主函数"""
    print("="*60)
    print("高保真度数据分析")
    print("="*60)

    # 加载数据
    df = load_data()

    # 分析分布
    stats, corr = analyze_distribution(df)

    # 对比合成数据
    df_synthetic, comparison = compare_with_synthetic(df)

    # 绘制分析图表
    plot_analysis(df, df_synthetic)

    # 材料对比
    plot_material_comparison(df)

    # 生成报告
    generate_report(df, stats, corr)

    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)


if __name__ == '__main__':
    main()
