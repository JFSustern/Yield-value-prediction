# src/data/loader.py

import glob
import os

import numpy as np
import pandas as pd


def load_excel_data(data_dir="data/20251121处理后/"):
    """
    加载所有 Excel 文件并计算混合功 Emix

    Returns:
        DataFrame: 包含 ['Emix', 'Temp_mean', 'Temp_end', 'Duration']
    """
    file_pattern = os.path.join(data_dir, "*.xlsx")
    files = glob.glob(file_pattern)

    results = []

    for f in files:
        try:
            df = pd.read_excel(f)

            # 简单的列名映射 (根据之前 read_excel 的结果)
            # ['日期', '时间', '锅内压力', '夹套温度', '扭矩', '水箱水温', '药浆温度', '转速']
            if '扭矩' not in df.columns or '转速' not in df.columns:
                print(f"Skipping {f}: Missing columns")
                continue

            # 预处理：填充 0 或 NaN
            df['扭矩'] = pd.to_numeric(df['扭矩'], errors='coerce').fillna(0)
            df['转速'] = pd.to_numeric(df['转速'], errors='coerce').fillna(0)
            df['药浆温度'] = pd.to_numeric(df['药浆温度'], errors='coerce')

            # 计算功率 P = Torque * Speed * (2*pi/60)
            # 假设扭矩单位 N.m, 转速 rpm -> Power in Watts
            # 如果扭矩是百分比，这里计算的是相对功，对于归一化输入也是可以的
            power = df['扭矩'] * df['转速']

            # 积分计算功 (假设采样间隔为 1分钟 = 60s)
            # 实际上应该解析 '时间' 列，这里简化假设每行代表 1 分钟
            dt = 60.0
            emix = np.sum(power) * dt

            temp_mean = df['药浆温度'].mean()
            temp_end = df['药浆温度'].iloc[-1] if not df['药浆温度'].empty else 25.0
            duration = len(df) * dt

            results.append({
                'filename': os.path.basename(f),
                'Emix': emix,
                'Temp_mean': temp_mean,
                'Temp_end': temp_end,
                'Duration': duration
            })

        except Exception as e:
            print(f"Error processing {f}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_excel_data()
    print(df.head())
    print(f"Total samples: {len(df)}")

