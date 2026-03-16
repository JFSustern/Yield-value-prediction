"""
评估低保真度和高保真度模型的性能
"""

import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multi_fidelity.src.model.pinn import YodelPINN


def evaluate_model(model_path, data_path, model_name):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估 {model_name}")
    print('='*60)

    # 加载模型
    device = torch.device('cpu')
    model = YodelPINN(input_dim=5, hidden_dim=128).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载数据
    df = pd.read_csv(data_path)
    print(f"数据样本数: {len(df)}")

    # 准备输入
    X = torch.tensor(
        df[['Phi', 'd50_um', 'sigma', 'Emix_J', 'Temp_C']].values,
        dtype=torch.float32
    ).to(device)
    y_true = df['Tau0_Pa'].values

    # 预测
    with torch.no_grad():
        y_pred, (phi_m, m1, g_max) = model(X)
        y_pred = y_pred.cpu().numpy()

    # 统计
    print(f"\n真实值范围: {y_true.min():.2f} - {y_true.max():.2f} Pa")
    print(f"预测值范围: {y_pred.min():.2f} - {y_pred.max():.2f} Pa")
    print(f"真实值均值: {y_true.mean():.2f} Pa")
    print(f"预测值均值: {y_pred.mean():.2f} Pa")

    # 误差
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # R²
    y_mean = y_true.mean()
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\nMAE: {mae:.2f} Pa")
    print(f"RMSE: {rmse:.2f} Pa")
    print(f"R²: {r2:.4f}")

    return y_true, y_pred, {'mae': mae, 'rmse': rmse, 'r2': r2}


def plot_comparison():
    """绘制对比图"""
    print("\n" + "="*60)
    print("生成对比图")
    print("="*60)

    # 高保真度数据路径
    high_fidelity_data = project_root / 'data/high_fidelity/cement_paste_data.csv'

    # 模型路径
    low_fidelity_model = project_root / 'multi_fidelity/models/low_fidelity/pinn_low.pth'
    high_fidelity_model = project_root / 'multi_fidelity/models/high_fidelity/pinn_high.pth'

    # 评估
    y_true_low, y_pred_low, metrics_low = evaluate_model(
        low_fidelity_model, high_fidelity_data, "低保真度模型"
    )

    y_true_high, y_pred_high, metrics_high = evaluate_model(
        high_fidelity_model, high_fidelity_data, "高保真度模型"
    )

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 低保真度模型
    ax = axes[0]
    ax.scatter(y_true_low, y_pred_low, alpha=0.6, s=50)
    ax.plot([y_true_low.min(), y_true_low.max()],
            [y_true_low.min(), y_true_low.max()],
            'r--', linewidth=2, label='Ideal')
    ax.set_xlabel('True Tau0 (Pa)', fontsize=12)
    ax.set_ylabel('Predicted Tau0 (Pa)', fontsize=12)
    ax.set_title(f'Low Fidelity Model\nR²={metrics_low["r2"]:.4f}, MAE={metrics_low["mae"]:.2f} Pa',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 高保真度模型
    ax = axes[1]
    ax.scatter(y_true_high, y_pred_high, alpha=0.6, s=50, color='orange')
    ax.plot([y_true_high.min(), y_true_high.max()],
            [y_true_high.min(), y_true_high.max()],
            'r--', linewidth=2, label='Ideal')
    ax.set_xlabel('True Tau0 (Pa)', fontsize=12)
    ax.set_ylabel('Predicted Tau0 (Pa)', fontsize=12)
    ax.set_title(f'High Fidelity Model (Fine-tuned)\nR²={metrics_high["r2"]:.4f}, MAE={metrics_high["mae"]:.2f} Pa',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    save_path = project_root / 'multi_fidelity/results/plots/model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图已保存至: {save_path}")
    plt.close()

    # 打印对比总结
    print("\n" + "="*60)
    print("模型对比总结")
    print("="*60)
    print(f"{'指标':<15} {'低保真度':<15} {'高保真度':<15} {'改进':<15}")
    print("-"*60)
    print(f"{'MAE (Pa)':<15} {metrics_low['mae']:<15.2f} {metrics_high['mae']:<15.2f} {metrics_low['mae']-metrics_high['mae']:<15.2f}")
    print(f"{'RMSE (Pa)':<15} {metrics_low['rmse']:<15.2f} {metrics_high['rmse']:<15.2f} {metrics_low['rmse']-metrics_high['rmse']:<15.2f}")
    print(f"{'R²':<15} {metrics_low['r2']:<15.4f} {metrics_high['r2']:<15.4f} {metrics_high['r2']-metrics_low['r2']:<15.4f}")
    print("="*60)


if __name__ == '__main__':
    plot_comparison()
