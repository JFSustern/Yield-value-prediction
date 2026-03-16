"""
Multi-Fidelity Neural Network Testing Script
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.model.mfnn import YodelMFNN


def load_test_data(data_path="data/synthetic/test_data.csv"):
    """加载测试数据"""
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)

    feature_cols = ['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)']
    target_col = 'Tau0(屈服应力_Pa)'

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    print(f"Loaded {len(df)} test samples")
    return X, y, df


def evaluate_model(model, X, y, device, use_hf=True):
    """评估模型"""
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        y_pred_hf, y_pred_lf, alpha, (phi_m, m1, g_max) = model(X_tensor, use_hf=use_hf)

        y_pred_hf = y_pred_hf.cpu().numpy().flatten()
        y_pred_lf = y_pred_lf.cpu().numpy().flatten()

        if alpha is not None:
            alpha = alpha.cpu().numpy().flatten()

        phi_m = phi_m.cpu().numpy().flatten()
        g_max = g_max.cpu().numpy().flatten()

    # 计算指标
    if use_hf:
        r2 = r2_score(y, y_pred_hf)
        rmse = np.sqrt(mean_squared_error(y, y_pred_hf))
        mae = mean_absolute_error(y, y_pred_hf)
        y_pred = y_pred_hf
    else:
        r2 = r2_score(y, y_pred_lf)
        rmse = np.sqrt(mean_squared_error(y, y_pred_lf))
        mae = mean_absolute_error(y, y_pred_lf)
        y_pred = y_pred_lf

    results = {
        'y_true': y,
        'y_pred_hf': y_pred_hf,
        'y_pred_lf': y_pred_lf,
        'alpha': alpha,
        'phi_m': phi_m,
        'g_max': g_max,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }

    return results


def plot_evaluation(results_lf, results_hf, save_path="data/synthetic/plots/mfnn_evaluation.png"):
    """绘制评估结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    y_true = results_lf['y_true']
    y_pred_lf = results_lf['y_pred_lf']
    y_pred_hf = results_hf['y_pred_hf']

    # 1. 低保真度预测 vs 真实值
    axes[0, 0].scatter(y_true, y_pred_lf, alpha=0.5, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Tau0 (Pa)')
    axes[0, 0].set_ylabel('Predicted Tau0 (Pa)')
    axes[0, 0].set_title(f'Low-Fidelity Prediction\nR²={results_lf["r2"]:.4f}, RMSE={results_lf["rmse"]:.2f}')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 高保真度预测 vs 真实值
    axes[0, 1].scatter(y_true, y_pred_hf, alpha=0.5, s=20, color='orange')
    axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Tau0 (Pa)')
    axes[0, 1].set_ylabel('Predicted Tau0 (Pa)')
    axes[0, 1].set_title(f'High-Fidelity Prediction\nR²={results_hf["r2"]:.4f}, RMSE={results_hf["rmse"]:.2f}')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 低保真度 vs 高保真度
    axes[0, 2].scatter(y_pred_lf, y_pred_hf, alpha=0.5, s=20, color='green')
    axes[0, 2].plot([y_pred_lf.min(), y_pred_lf.max()], [y_pred_lf.min(), y_pred_lf.max()], 'r--', lw=2)
    axes[0, 2].set_xlabel('Low-Fidelity Prediction (Pa)')
    axes[0, 2].set_ylabel('High-Fidelity Prediction (Pa)')
    axes[0, 2].set_title('LF vs HF Predictions')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 误差分布 (低保真度)
    error_lf = y_pred_lf - y_true
    axes[1, 0].hist(error_lf, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Prediction Error (Pa)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'LF Error Distribution\nMAE={results_lf["mae"]:.2f} Pa')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 误差分布 (高保真度)
    error_hf = y_pred_hf - y_true
    axes[1, 1].hist(error_hf, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Prediction Error (Pa)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'HF Error Distribution\nMAE={results_hf["mae"]:.2f} Pa')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 缩放因子 α 分布
    if results_hf['alpha'] is not None:
        axes[1, 2].hist(results_hf['alpha'], bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 2].axvline(results_hf['alpha'].mean(), color='red', linestyle='--', linewidth=2,
                          label=f'Mean={results_hf["alpha"].mean():.3f}')
        axes[1, 2].set_xlabel('Scaling Factor α')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Alpha Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'No Alpha\n(LF only)', ha='center', va='center', fontsize=14)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nEvaluation plot saved to {save_path}")
    plt.close()


def plot_correction_analysis(results, save_path="data/synthetic/plots/mfnn_correction.png"):
    """分析高保真度修正的效果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    y_true = results['y_true']
    y_pred_lf = results['y_pred_lf']
    y_pred_hf = results['y_pred_hf']
    alpha = results['alpha']

    if alpha is None:
        print("No alpha available, skipping correction analysis")
        return

    correction = y_pred_hf - y_pred_lf

    # 1. 修正量 vs 真实值
    axes[0, 0].scatter(y_true, correction, alpha=0.5, s=20)
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('True Tau0 (Pa)')
    axes[0, 0].set_ylabel('Correction (HF - LF) (Pa)')
    axes[0, 0].set_title('Correction vs True Value')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 修正量 vs 低保真度预测
    axes[0, 1].scatter(y_pred_lf, correction, alpha=0.5, s=20, color='orange')
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('LF Prediction (Pa)')
    axes[0, 1].set_ylabel('Correction (HF - LF) (Pa)')
    axes[0, 1].set_title('Correction vs LF Prediction')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Alpha vs 低保真度预测
    axes[1, 0].scatter(y_pred_lf, alpha, alpha=0.5, s=20, color='green')
    axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('LF Prediction (Pa)')
    axes[1, 0].set_ylabel('Scaling Factor α')
    axes[1, 0].set_title('Alpha vs LF Prediction')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 误差改善
    error_lf = np.abs(y_pred_lf - y_true)
    error_hf = np.abs(y_pred_hf - y_true)
    improvement = error_lf - error_hf

    axes[1, 1].scatter(error_lf, improvement, alpha=0.5, s=20, color='purple')
    axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('LF Absolute Error (Pa)')
    axes[1, 1].set_ylabel('Error Improvement (Pa)')
    axes[1, 1].set_title(f'Error Improvement\nMean={improvement.mean():.2f} Pa')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Correction analysis saved to {save_path}")
    plt.close()


def test_mfnn():
    """主测试函数"""
    print("=" * 60)
    print("Multi-Fidelity Neural Network Testing")
    print("=" * 60)

    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # 加载测试数据
    X_test, y_test, df_test = load_test_data()

    # 初始化模型
    model = YodelMFNN(input_dim=5, lf_hidden_dim=128, alpha_hidden_dim=64).to(device)

    # 加载训练好的模型
    model_path = "models/yodel_mfnn.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_mfnn.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}\n")

    # 评估低保真度预测
    print("=" * 60)
    print("Low-Fidelity Evaluation")
    print("=" * 60)
    results_lf = evaluate_model(model, X_test, y_test, device, use_hf=False)

    print(f"Samples: {len(y_test)}")
    print(f"R² Score: {results_lf['r2']:.4f}")
    print(f"RMSE:     {results_lf['rmse']:.4f} Pa")
    print(f"MAE:      {results_lf['mae']:.4f} Pa")
    print("-" * 60)
    print(f"Physical Parameters (Mean):")
    print(f"Phi_m: {results_lf['phi_m'].mean():.4f}")
    print(f"G_max: {results_lf['g_max'].mean():.0f}")

    # 评估高保真度预测
    print("\n" + "=" * 60)
    print("High-Fidelity Evaluation")
    print("=" * 60)
    results_hf = evaluate_model(model, X_test, y_test, device, use_hf=True)

    print(f"Samples: {len(y_test)}")
    print(f"R² Score: {results_hf['r2']:.4f}")
    print(f"RMSE:     {results_hf['rmse']:.4f} Pa")
    print(f"MAE:      {results_hf['mae']:.4f} Pa")

    if results_hf['alpha'] is not None:
        print("-" * 60)
        print(f"Scaling Factor α (Mean): {results_hf['alpha'].mean():.4f}")
        print(f"Scaling Factor α (Std):  {results_hf['alpha'].std():.4f}")

        # 计算改善
        improvement = results_lf['rmse'] - results_hf['rmse']
        improvement_pct = (improvement / results_lf['rmse']) * 100
        print("-" * 60)
        print(f"RMSE Improvement: {improvement:.4f} Pa ({improvement_pct:.2f}%)")

    # 保存结果
    results_df = df_test.copy()
    results_df['Tau0_Pred_LF'] = results_lf['y_pred_lf']
    results_df['Tau0_Pred_HF'] = results_hf['y_pred_hf']
    results_df['Error_LF'] = results_lf['y_pred_lf'] - y_test
    results_df['Error_HF'] = results_hf['y_pred_hf'] - y_test

    if results_hf['alpha'] is not None:
        results_df['Alpha'] = results_hf['alpha']
        results_df['Correction'] = results_hf['y_pred_hf'] - results_lf['y_pred_lf']

    results_path = "data/synthetic/mfnn_test_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # 绘制评估图
    plot_evaluation(results_lf, results_hf)

    # 绘制修正分析
    if results_hf['alpha'] is not None:
        plot_correction_analysis(results_hf)

    print("\n" + "=" * 60)
    print("Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_mfnn()
