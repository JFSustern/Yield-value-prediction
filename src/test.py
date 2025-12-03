# src/test.py

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.model.pinn import YodelPINN


def test():
    # 1. 加载测试数据
    test_path = "data/synthetic/test_data.csv"
    if not os.path.exists(test_path):
        print(f"Error: Test data not found at {test_path}")
        return

    print(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)

    X = df[['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)']].values.astype(np.float32)
    y_true = df['Tau0(屈服应力_Pa)'].values.astype(np.float32)

    X_tensor = torch.from_numpy(X)

    # 2. 加载模型
    model_path = "models/yodel_pinn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # 设备选择
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    model = YodelPINN().to(device)
    # 加载权重时映射到对应设备
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 预测
    # 移动数据到设备
    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        y_pred_tensor, (phi_m, m1, g_max) = model(X_tensor)
        # 转回 CPU 进行 numpy 操作
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        phi_m = phi_m.cpu()
        g_max = g_max.cpu()

    # 4. 评估指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + "="*40)
    print("Test Results")
    print("="*40)
    print(f"Samples: {len(y_true)}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f} Pa")
    print(f"MAE:      {mae:.4f} Pa")
    print("-" * 40)
    print("Physical Parameters (Mean):")
    print(f"Phi_m: {np.mean(phi_m.numpy()):.4f}")
    print(f"G_max: {np.mean(g_max.numpy()):.0f}")
    print("="*40)

    # 5. 保存预测结果
    df['Tau0_Pred'] = y_pred
    df['Error'] = y_pred - y_true
    df['Abs_Error'] = np.abs(df['Error'])

    result_path = "data/synthetic/test_results.csv"
    df.to_csv(result_path, index=False)
    print(f"\nDetailed results saved to {result_path}")

    # 6. 绘制评估图表
    plot_test_results(y_true, y_pred)

def plot_test_results(y_true, y_pred):
    """绘制测试集评估图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. True vs Pred (Scatter)
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, color='blue', s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_title('True vs Predicted (Scatter)')
    axes[0, 0].set_xlabel('True Tau0 (Pa)')
    axes[0, 0].set_ylabel('Predicted Tau0 (Pa)')
    axes[0, 0].grid(True)

    # 2. True vs Pred (Hexbin Heatmap)
    # 聚焦于 90% 的数据范围，以便看清密集区
    limit = np.percentile(y_true, 90)

    hb = axes[0, 1].hexbin(y_true, y_pred, gridsize=30, cmap='Blues', mincnt=1,
                           extent=[0, limit, 0, limit]) # 限制 hexbin 范围

    axes[0, 1].plot([0, limit], [0, limit], 'r--', lw=2)
    axes[0, 1].set_title(f'True vs Predicted (Density, 0-{limit:.0f} Pa)')
    axes[0, 1].set_xlabel('True Tau0 (Pa)')
    axes[0, 1].set_ylabel('Predicted Tau0 (Pa)')
    axes[0, 1].set_xlim(0, limit)
    axes[0, 1].set_ylim(0, limit)
    cb = fig.colorbar(hb, ax=axes[0, 1])
    cb.set_label('Count')

    # 3. Residuals vs True Value
    residuals = y_pred - y_true
    axes[1, 0].scatter(y_true, residuals, alpha=0.5, color='purple', s=10)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('Residuals vs True Value')
    axes[1, 0].set_xlabel('True Tau0 (Pa)')
    axes[1, 0].set_ylabel('Residual (Pred - True)')
    axes[1, 0].grid(True)

    # 4. Residual Distribution
    axes[1, 1].hist(residuals, bins=30, color='green', alpha=0.7, density=True)
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].set_xlabel('Prediction Error (Pa)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True)

    plt.tight_layout()

    # 保存到 plots 目录
    plot_dir = "data/synthetic/plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "test_evaluation_advanced.png")

    plt.savefig(save_path)
    print(f"Advanced evaluation plots saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    test()

