"""
详细特征分析脚本 - 生成单个特征的深度分析
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# 设置中文字体 - macOS系统
plt.rcParams['font.sans-serif'] = ['Heiti SC', 'Songti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class DetailedFeatureAnalyzer:
    """详细特征分析类"""

    def __init__(self, csv_path):
        """初始化分析器"""
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        self.output_dir = self.csv_path.parent / 'analysis_results'
        self.output_dir.mkdir(exist_ok=True)

    def analyze_feature(self, feature_name):
        """分析单个特征"""
        if feature_name not in self.df.columns:
            print(f"错误: 特征 '{feature_name}' 不存在")
            return

        data = self.df[feature_name]

        print("\n" + "=" * 80)
        print(f"特征详细分析: {feature_name}")
        print("=" * 80)

        # 基本统计
        print("\n【基本统计信息】")
        print(f"  数据类型: {data.dtype}")
        print(f"  样本数: {len(data)}")
        print(f"  缺失值: {data.isnull().sum()}")
        print(f"  唯一值数: {data.nunique()}")

        # 位置统计
        print("\n【位置统计】")
        print(f"  最小值: {data.min():.6f}")
        print(f"  最大值: {data.max():.6f}")
        print(f"  均值: {data.mean():.6f}")
        print(f"  中位数: {data.median():.6f}")
        print(f"  众数: {data.mode().values[0] if len(data.mode()) > 0 else 'N/A':.6f}")

        # 离散度统计
        print("\n【离散度统计】")
        print(f"  标准差: {data.std():.6f}")
        print(f"  方差: {data.var():.6f}")
        print(f"  极差: {data.max() - data.min():.6f}")
        print(f"  四分位距(IQR): {data.quantile(0.75) - data.quantile(0.25):.6f}")
        print(f"  变异系数: {(data.std() / data.mean()):.6f}")

        # 分位数
        print("\n【分位数】")
        for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
            print(f"  {q*100:5.0f}% 分位数: {data.quantile(q):.6f}")

        # 形状统计
        print("\n【分布形状】")
        skewness = data.skew()
        kurtosis = data.kurtosis()
        print(f"  偏度(Skewness): {skewness:.6f}")
        if abs(skewness) < 0.5:
            print(f"    → 分布形状: 对称")
        elif skewness > 0:
            print(f"    → 分布形状: 右偏(正偏)")
        else:
            print(f"    → 分布形状: 左偏(负偏)")

        print(f"  峰度(Kurtosis): {kurtosis:.6f}")
        if abs(kurtosis) < 0.5:
            print(f"    → 峰度: 正常")
        elif kurtosis > 0:
            print(f"    → 峰度: 尖峰(重尾)")
        else:
            print(f"    → 峰度: 平峰(轻尾)")

        # 正态性检验
        print("\n【正态性检验】")
        stat, p_value = stats.shapiro(data)
        print(f"  Shapiro-Wilk 检验:")
        print(f"    统计量: {stat:.6f}")
        print(f"    p值: {p_value:.6f}")
        if p_value > 0.05:
            print(f"    → 结论: 数据符合正态分布 (p > 0.05)")
        else:
            print(f"    → 结论: 数据不符合正态分布 (p < 0.05)")

        # 异常值检测
        print("\n【异常值检测】")
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        print(f"  IQR方法:")
        print(f"    下界: {lower_bound:.6f}")
        print(f"    上界: {upper_bound:.6f}")
        print(f"    异常值数量: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
        if len(outliers) > 0:
            print(f"    异常值范围: {outliers.min():.6f} ~ {outliers.max():.6f}")

        # 绘制详细分析图
        self._plot_detailed_analysis(feature_name, data)

    def _plot_detailed_analysis(self, feature_name, data):
        """绘制详细分析图"""
        fig = plt.figure(figsize=(16, 12))

        # 1. 直方图 + KDE
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(data, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        ax1_twin = ax1.twinx()
        data.plot(kind='kde', ax=ax1_twin, color='red', linewidth=2)
        ax1.set_title('直方图 + KDE', fontweight='bold')
        ax1.set_xlabel('值')
        ax1.set_ylabel('频数')
        ax1_twin.set_ylabel('密度')
        ax1.grid(True, alpha=0.3)

        # 2. 箱线图
        ax2 = plt.subplot(3, 3, 2)
        bp = ax2.boxplot(data, vert=True, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax2.set_title('箱线图', fontweight='bold')
        ax2.set_ylabel('值')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 小提琴图
        ax3 = plt.subplot(3, 3, 3)
        parts = ax3.violinplot([data.values], positions=[0], showmeans=True, showmedians=True)
        ax3.set_title('小提琴图', fontweight='bold')
        ax3.set_ylabel('值')
        ax3.set_xticks([])
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Q-Q图
        ax4 = plt.subplot(3, 3, 4)
        stats.probplot(data, dist="norm", plot=ax4)
        ax4.set_title('Q-Q图 (正态性检验)', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. 累积分布函数
        ax5 = plt.subplot(3, 3, 5)
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax5.plot(sorted_data, cumulative, linewidth=2, color='blue')
        ax5.fill_between(sorted_data, cumulative, alpha=0.3, color='blue')
        ax5.set_title('累积分布函数 (CDF)', fontweight='bold')
        ax5.set_xlabel('值')
        ax5.set_ylabel('累积概率')
        ax5.grid(True, alpha=0.3)

        # 6. 对数变换后的直方图
        ax6 = plt.subplot(3, 3, 6)
        if (data > 0).all():
            log_data = np.log(data)
            ax6.hist(log_data, bins=40, alpha=0.7, color='lightgreen', edgecolor='black')
            ax6.set_title('对数变换后的直方图', fontweight='bold')
            ax6.set_xlabel('log(值)')
            ax6.set_ylabel('频数')
        else:
            ax6.text(0.5, 0.5, '数据包含非正值\n无法进行对数变换',
                    ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('对数变换 (不适用)', fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # 7. 标准化后的直方图
        ax7 = plt.subplot(3, 3, 7)
        normalized = (data - data.mean()) / data.std()
        ax7.hist(normalized, bins=40, alpha=0.7, color='lightyellow', edgecolor='black')
        ax7.axvline(0, color='red', linestyle='--', linewidth=2, label='均值')
        ax7.axvline(-1, color='orange', linestyle='--', linewidth=1, label='±1σ')
        ax7.axvline(1, color='orange', linestyle='--', linewidth=1)
        ax7.set_title('标准化后的直方图', fontweight='bold')
        ax7.set_xlabel('标准化值')
        ax7.set_ylabel('频数')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 时间序列图 (按索引)
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(data.values, linewidth=0.5, alpha=0.7, color='blue')
        ax8.axhline(data.mean(), color='red', linestyle='--', linewidth=2, label='均值')
        ax8.axhline(data.median(), color='green', linestyle='--', linewidth=2, label='中位数')
        ax8.set_title('数据序列图', fontweight='bold')
        ax8.set_xlabel('样本索引')
        ax8.set_ylabel('值')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. 统计信息文本
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        stats_text = f"""
        统计信息摘要

        样本数: {len(data)}
        均值: {data.mean():.4f}
        中位数: {data.median():.4f}
        标准差: {data.std():.4f}

        最小值: {data.min():.4f}
        最大值: {data.max():.4f}
        极差: {data.max() - data.min():.4f}

        偏度: {data.skew():.4f}
        峰度: {data.kurtosis():.4f}

        Q1: {data.quantile(0.25):.4f}
        Q3: {data.quantile(0.75):.4f}
        IQR: {data.quantile(0.75) - data.quantile(0.25):.4f}
        """
        ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

        plt.suptitle(f'特征详细分析: {feature_name}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_path = self.output_dir / f'详细分析_{feature_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 详细分析图已保存: {output_path}")
        plt.close()

    def compare_features(self, feature_list=None):
        """比较多个特征"""
        if feature_list is None:
            feature_list = self.df.columns.tolist()

        # 过滤存在的特征
        feature_list = [f for f in feature_list if f in self.df.columns]

        print("\n" + "=" * 80)
        print("特征对比分析")
        print("=" * 80)

        # 创建对比表
        comparison_data = []
        for feature in feature_list:
            data = self.df[feature]
            comparison_data.append({
                '特征': feature,
                '均值': f"{data.mean():.4f}",
                '中位数': f"{data.median():.4f}",
                '标准差': f"{data.std():.4f}",
                '偏度': f"{data.skew():.4f}",
                '峰度': f"{data.kurtosis():.4f}",
                '最小值': f"{data.min():.4f}",
                '最大值': f"{data.max():.4f}",
            })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))


def main():
    """主函数"""
    csv_path = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'dataset.csv'

    if not csv_path.exists():
        print(f"错误: 找不到文件 {csv_path}")
        return

    analyzer = DetailedFeatureAnalyzer(csv_path)

    # 分析所有特征
    print("\n开始详细特征分析...\n")
    for feature in analyzer.df.columns:
        analyzer.analyze_feature(feature)

    # 特征对比
    analyzer.compare_features()

    print("\n" + "=" * 80)
    print("✓ 详细分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

