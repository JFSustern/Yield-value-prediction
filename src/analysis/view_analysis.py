"""
查看特征分析结果的脚本
"""

import subprocess
import sys
from pathlib import Path


def main():
    """打开分析结果目录"""
    analysis_dir = Path(__file__).parent.parent.parent / 'data' / 'synthetic' / 'analysis_results'

    if not analysis_dir.exists():
        print(f"错误: 分析结果目录不存在: {analysis_dir}")
        return

    print("=" * 80)
    print("特征分析结果")
    print("=" * 80)
    print(f"\n分析结果保存位置: {analysis_dir}\n")

    # 列出所有生成的文件
    files = sorted(analysis_dir.glob('*'))

    print("生成的文件列表:")
    print("-" * 80)
    for i, file in enumerate(files, 1):
        size = file.stat().st_size
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.2f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.2f} KB"
        else:
            size_str = f"{size} B"

        print(f"{i}. {file.name:<30} ({size_str})")

    print("\n" + "-" * 80)
    print("文件说明:")
    print("-" * 80)
    descriptions = {
        "00_分析报告.txt": "详细的统计分析报告，包含所有特征的统计信息和相关性分析",
        "01_特征分布图.png": "每个特征的直方图和核密度估计(KDE)图",
        "02_箱线图.png": "所有特征的箱线图，显示四分位数和异常值",
        "03_相关性热力图.png": "特征之间的相关系数矩阵热力图",
        "04_散点矩阵图.png": "前5个特征的散点矩阵图，显示特征间的关系",
        "05_小提琴图.png": "标准化后的特征分布小提琴图",
        "06_Q-Q图.png": "Q-Q图用于检验特征的正态性",
        "07_累积分布函数.png": "每个特征的累积分布函数(CDF)图"
    }

    for file in files:
        if file.name in descriptions:
            print(f"\n✓ {file.name}")
            print(f"  {descriptions[file.name]}")

    print("\n" + "=" * 80)
    print("✓ 所有分析完成！")
    print("=" * 80)

    # 尝试打开文件夹
    try:
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', str(analysis_dir)])
        elif sys.platform == 'win32':  # Windows
            subprocess.run(['explorer', str(analysis_dir)])
        elif sys.platform == 'linux':  # Linux
            subprocess.run(['xdg-open', str(analysis_dir)])
    except Exception as e:
        print(f"\n提示: 无法自动打开文件夹，请手动访问: {analysis_dir}")

if __name__ == '__main__':
    main()

