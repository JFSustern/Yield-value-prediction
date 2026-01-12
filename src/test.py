
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.model.pinn import YodelPINN


def test():
    # 1. 加载测试数据 (Long Format)
    test_path = "data/synthetic/test_data.csv"
    if not os.path.exists(test_path):
        print(f"Error: Test data not found at {test_path}")
        return

    print(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)

    # 输入特征: [Phi, d50, sigma, Emix, Temp]
    feature_cols = ['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)']
    target_col = 'Tau0(屈服应力_Pa)'

    X = df[feature_cols].values.astype(np.float32)
    y_true = df[target_col].values.astype(np.float32)

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
    model = YodelPINN(input_dim=5, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 预测 (Long Format)
    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        y_pred_tensor, params = model(X_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        phi_m = params[0].cpu().numpy().flatten()
        g_max = params[2].cpu().numpy().flatten()

    # 4. 评估指标 (Long Format)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("\n" + "="*40)
    print("Test Results (Mixed Stages)")
    print("="*40)
    print(f"Samples: {len(y_true)}")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f} Pa")
    print(f"MAE:      {mae:.4f} Pa")
    print("-" * 40)
    print("Physical Parameters (Mean):")
    print(f"Phi_m: {np.mean(phi_m):.4f}")
    print(f"G_max: {np.mean(g_max):.0f}")
    print("="*40)

    # 5. 保存预测结果
    df['Tau0_Pred'] = y_pred.round(4)
    df['Error'] = (y_pred - y_true).round(4)

    result_path = "data/synthetic/test_results.csv"
    df.to_csv(result_path, index=False)
    print(f"\nDetailed results saved to {result_path}")

    # 6. 绘制评估图表
    plot_test_results(y_true, y_pred)

    # 7. 双阶段评估 (Dual Stage Evaluation)
    # 加载原始 Wide Format 数据
    raw_path = "data/synthetic/dataset.csv"
    if os.path.exists(raw_path):
        evaluate_dual_stage(model, raw_path, device)

def plot_test_results(y_true, y_pred):
    """绘制测试集评估图表"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # True vs Pred
    axes[0].scatter(y_true, y_pred, alpha=0.5, color='blue', s=10)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_title('True vs Predicted')
    axes[0].set_xlabel('True Tau0 (Pa)')
    axes[0].set_ylabel('Predicted Tau0 (Pa)')
    axes[0].grid(True)
    # Log scale for better visibility
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    # Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=30, color='green', alpha=0.7)
    axes[1].set_title('Residual Distribution')
    axes[1].set_xlabel('Error (Pa)')
    axes[1].grid(True)

    plt.tight_layout()
    plot_dir = "data/synthetic/plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "test_evaluation.png")
    plt.savefig(save_path)
    print(f"Evaluation plots saved to {save_path}")
    plt.close()

def evaluate_dual_stage(model, data_path, device):
    """评估模型在双阶段预测任务上的表现"""
    print("\n" + "="*40)
    print("Dual Stage Evaluation (Process Simulation)")
    print("="*40)

    df = pd.read_csv(data_path)
    # 取一部分作为测试 (例如最后 20%)
    n_test = int(len(df) * 0.2)
    df_test = df.iloc[-n_test:].copy()

    print(f"Evaluating on {len(df_test)} process batches...")

    # 准备 Point 1 数据
    X1 = df_test[['Phi_1(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix_1(混合功_J)', 'Temp_1(温度_C)']].values.astype(np.float32)
    y1_true = df_test['Tau0_1(屈服应力_Pa)'].values.astype(np.float32)

    # 准备 Point 2 数据
    X2 = df_test[['Phi_2(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix_2(混合功_J)', 'Temp_2(温度_C)']].values.astype(np.float32)
    y2_true = df_test['Tau0_2(屈服应力_Pa)'].values.astype(np.float32)

    # 预测
    model.eval()
    with torch.no_grad():
        # Point 1
        pred1, _ = model(torch.from_numpy(X1).to(device))
        y1_pred = pred1.cpu().numpy().flatten()

        # Point 2
        pred2, _ = model(torch.from_numpy(X2).to(device))
        y2_pred = pred2.cpu().numpy().flatten()

    # 计算指标
    r2_1 = r2_score(y1_true, y1_pred)
    r2_2 = r2_score(y2_true, y2_pred)

    print(f"Point 1 (48 min) R²: {r2_1:.4f}")
    print(f"Point 2 (111 min) R²: {r2_2:.4f}")

    # 绘制对比图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y1_true, y1_pred, color='red', alpha=0.5, label='Point 1 (Peak)', s=15)
    ax.scatter(y2_true, y2_pred, color='blue', alpha=0.5, label='Point 2 (Final)', s=15)

    # 统一坐标轴范围
    all_min = min(y1_true.min(), y2_true.min())
    all_max = max(y1_true.max(), y2_true.max())
    ax.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2)

    ax.set_title('Dual Stage Prediction Accuracy')
    ax.set_xlabel('True Tau0 (Pa)')
    ax.set_ylabel('Predicted Tau0 (Pa)')
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_yscale('log')

    save_path = "data/synthetic/plots/dual_stage_eval.png"
    plt.savefig(save_path)
    print(f"Dual stage plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    test()

