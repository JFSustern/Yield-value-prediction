"""
特征分析模块：对生成的数据集进行全面的特征分析和可视化
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置中文字体 - macOS系统
plt.rcParams['font.sans-serif'] = ['Heiti SC', 'Songti SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class FeatureAnalyzer:
    """特征分析类"""

    def __init__(self, csv_path):
        """
        初始化分析器

        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        self.output_dir = self.csv_path.parent / 'analysis_results'
        self.output_dir.mkdir(exist_ok=True)

    def get_basic_statistics(self):
        """获取基本统计信息"""
        print("=" * 80)
        print("数据集基本信息")
        print("=" * 80)
        print(f"数据集形状: {self.df.shape}")
        print(f"特征数量: {len(self.df.columns)}")
        print(f"样本数量: {len(self.df)}")
        print(f"\n缺失值统计:\n{self.df.isnull().sum()}")

        print("\n" + "=" * 80)
        print("特征统计描述")
        print("=" * 80)
        stats = self.df.describe().T
        stats['skewness'] = self.df.skew()
        stats['kurtosis'] = self.df.kurtosis()
        print(stats)

        return stats

    def plot_distribution(self):
        """绘制特征分布图"""
        print("\n绘制特征分布图...")

        n_features = len(self.df.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(self.df.columns):
            ax = axes[idx]

            # 绘制直方图和KDE
            self.df[col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
            ax2 = ax.twinx()
            self.df[col].plot(kind='kde', ax=ax2, color='red', linewidth=2)

            ax.set_title(f'{col}\n(均值: {self.df[col].mean():.4f}, 标准差: {self.df[col].std():.4f})',
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('值')
            ax.set_ylabel('频数')
            ax2.set_ylabel('密度')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / '01_特征分布图.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

    def plot_boxplot(self):
        """绘制箱线图"""
        print("绘制箱线图...")

        fig, axes = plt.subplots(2, 5, figsize=(18, 8))
        axes = axes.flatten()

        for idx, col in enumerate(self.df.columns):
            ax = axes[idx]
            bp = ax.boxplot(self.df[col], vert=True, patch_artist=True)

            # 设置颜色
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')

            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.set_ylabel('值')
            ax.grid(True, alpha=0.3, axis='y')

            # 添加统计信息
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            median = self.df[col].median()
            ax.text(1.15, median, f'中位数: {median:.2f}', fontsize=8)

        plt.tight_layout()
        output_path = self.output_dir / '02_箱线图.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

    def plot_correlation_heatmap(self):
        """绘制相关性热力图"""
        print("绘制相关性热力图...")

        # 计算相关系数矩阵
        corr_matrix = self.df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1)

        ax.set_title('特征相关性矩阵', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = self.output_dir / '03_相关性热力图.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

        return corr_matrix

    def plot_scatter_matrix(self):
        """绘制散点矩阵图（选择主要特征）"""
        print("绘制散点矩阵图...")

        # 选择前5个特征进行散点矩阵分析
        selected_cols = self.df.columns[:5]

        fig = plt.figure(figsize=(14, 12))

        n_features = len(selected_cols)
        for i, col1 in enumerate(selected_cols):
            for j, col2 in enumerate(selected_cols):
                ax = plt.subplot(n_features, n_features, i * n_features + j + 1)

                if i == j:
                    # 对角线上绘制直方图
                    ax.hist(self.df[col1], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_ylabel('频数')
                else:
                    # 绘制散点图
                    ax.scatter(self.df[col2], self.df[col1], alpha=0.5, s=10, color='blue')
                    ax.set_ylabel(col1 if j == 0 else '')

                ax.set_xlabel(col2 if i == n_features - 1 else '')

                if i == 0:
                    ax.set_title(col2, fontsize=9, fontweight='bold')

                ax.grid(True, alpha=0.3)

        plt.suptitle('特征散点矩阵图（前5个特征）', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_path = self.output_dir / '04_散点矩阵图.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

    def plot_violin_plot(self):
        """绘制小提琴图"""
        print("绘制小提琴图...")

        # 标准化数据用于比较
        df_normalized = (self.df - self.df.mean()) / self.df.std()

        fig, ax = plt.subplots(figsize=(16, 6))

        # 准备数据
        data_for_violin = []
        labels = []
        for col in self.df.columns:
            data_for_violin.append(df_normalized[col].values)
            labels.append(col)

        parts = ax.violinplot(data_for_violin, positions=range(len(labels)),
                             showmeans=True, showmedians=True)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('标准化值')
        ax.set_title('特征分布小提琴图（标准化）', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = self.output_dir / '05_小提琴图.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

    def plot_qq_plot(self):
        """绘制Q-Q图检验正态性"""
        print("绘制Q-Q图...")

        from scipy import stats

        n_features = len(self.df.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(self.df.columns):
            ax = axes[idx]
            stats.probplot(self.df[col], dist="norm", plot=ax)
            ax.set_title(f'{col} Q-Q图', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / '06_Q-Q图.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

    def plot_cumulative_distribution(self):
        """绘制累积分布函数"""
        print("绘制累积分布函数...")

        n_features = len(self.df.columns)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()

        for idx, col in enumerate(self.df.columns):
            ax = axes[idx]

            # 计算累积分布
            sorted_data = np.sort(self.df[col])
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

            ax.plot(sorted_data, cumulative, linewidth=2, color='blue')
            ax.fill_between(sorted_data, cumulative, alpha=0.3, color='blue')

            ax.set_title(f'{col} 累积分布函数', fontsize=10, fontweight='bold')
            ax.set_xlabel('值')
            ax.set_ylabel('累积概率')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / '07_累积分布函数.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {output_path}")
        plt.close()

    def generate_summary_report(self, corr_matrix):
        """生成汇总报告"""
        print("\n生成汇总报告...")

        report = []
        report.append("=" * 80)
        report.append("数据集特征分析汇总报告")
        report.append("=" * 80)
        report.append(f"\n数据集路径: {self.csv_path}")
        report.append(f"数据集形状: {self.df.shape[0]} 行 × {self.df.shape[1]} 列")

        report.append("\n" + "-" * 80)
        report.append("特征统计信息")
        report.append("-" * 80)

        for col in self.df.columns:
            report.append(f"\n{col}:")
            report.append(f"  最小值: {self.df[col].min():.6f}")
            report.append(f"  最大值: {self.df[col].max():.6f}")
            report.append(f"  均值: {self.df[col].mean():.6f}")
            report.append(f"  中位数: {self.df[col].median():.6f}")
            report.append(f"  标准差: {self.df[col].std():.6f}")
            report.append(f"  偏度: {self.df[col].skew():.6f}")
            report.append(f"  峰度: {self.df[col].kurtosis():.6f}")
            report.append(f"  四分位距(IQR): {self.df[col].quantile(0.75) - self.df[col].quantile(0.25):.6f}")

        report.append("\n" + "-" * 80)
        report.append("特征相关性分析")
        report.append("-" * 80)

        # 找出最强的相关性对
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        report.append("\n最强的10个相关性对:")
        for i, (feat1, feat2, corr) in enumerate(corr_pairs[:10], 1):
            report.append(f"  {i}. {feat1} <-> {feat2}: {corr:.4f}")

        report.append("\n" + "-" * 80)
        report.append("数据质量评估")
        report.append("-" * 80)
        report.append(f"缺失值总数: {self.df.isnull().sum().sum()}")
        report.append(f"重复行数: {self.df.duplicated().sum()}")
        report.append(f"数据完整性: {(1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100:.2f}%")

        report_text = "\n".join(report)
        print(report_text)

        # 保存报告
        report_path = self.output_dir / '00_分析报告.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ 报告已保存: {report_path}")

        return report_text

    def run_full_analysis(self):
        """运行完整分析"""
        print("\n开始进行特征分析...\n")

        # 基本统计
        self.get_basic_statistics()

        # 绘制各种图表
        self.plot_distribution()
        self.plot_boxplot()
        corr_matrix = self.plot_correlation_heatmap()
        self.plot_scatter_matrix()
        self.plot_violin_plot()
        self.plot_qq_plot()
        self.plot_cumulative_distribution()

        # 生成报告
        self.generate_summary_report(corr_matrix)

        print("\n" + "=" * 80)
        print(f"✓ 分析完成！所有结果已保存到: {self.output_dir}")
        print("=" * 80)


def main():
    """主函数"""
    csv_path = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'dataset.csv'

    if not csv_path.exists():
        print(f"错误: 找不到文件 {csv_path}")
        return

    analyzer = FeatureAnalyzer(csv_path)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()

