
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

    # 输入特征: [Phi_final, d50, sigma, Emix, Temp, Ratio_curing]
    feature_cols = ['Phi_final(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)',
                   'Emix(混合功_J)', 'Temp(温度_C)', 'Curing_Ratio(固化剂比例)']

    # 目标值: [Tau0_peak, Tau0_final]
    target_cols = ['Tau0_peak(高峰屈服_Pa)', 'Tau0_final(最终屈服_Pa)']

    X = df[feature_cols].values.astype(np.float32)
    y_true = df[target_cols].values.astype(np.float32) # [N, 2]

    X_tensor = torch.from_numpy(X)

    # 2. 加载模型
    model_path = "models/yodel_pinn_dual.pth"
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
    model = YodelPINN(input_dim=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 预测
    X_tensor = X_tensor.to(device)

    with torch.no_grad():
        # pred: [N, 2], params: (phi_m, m1, g_max, phi_peak, delta)
        y_pred_tensor, params = model(X_tensor)

        y_pred = y_pred_tensor.cpu().numpy()
        phi_m = params[0].cpu().numpy().flatten()
        phi_peak = params[3].cpu().numpy().flatten()
        delta = params[4].cpu().numpy().flatten()

    # 分离 Peak 和 Final
    y_true_peak = y_true[:, 0]
    y_true_final = y_true[:, 1]
    y_pred_peak = y_pred[:, 0]
    y_pred_final = y_pred[:, 1]

    # 4. 评估指标
    def evaluate(name, true, pred):
        # Standard Metrics
        r2 = r2_score(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)

        # Log-Space Metrics (Better for wide-range data)
        log_true = np.log1p(true)
        log_pred = np.log1p(pred)

        log_rmse = np.sqrt(mean_squared_error(log_true, log_pred))
        log_r2 = r2_score(log_true, log_pred)

        return r2, rmse, mae, log_rmse, log_r2

    r2_p, rmse_p, mae_p, log_rmse_p, log_r2_p = evaluate("Peak", y_true_peak, y_pred_peak)
    r2_f, rmse_f, mae_f, log_rmse_f, log_r2_f = evaluate("Final", y_true_final, y_pred_final)

    print("\n" + "="*75)
    print(f"{'Metric':<12} | {'Peak Stage':<25} | {'Final Stage':<25}")
    print("-" * 75)
    print(f"{'R² (Linear)':<12} | {r2_p:.4f}{'':<19} | {r2_f:.4f}")
    print(f"{'R² (Log)':<12} | {log_r2_p:.4f} (Recommended){'':<5} | {log_r2_f:.4f}")
    print("-" * 75)
    print(f"{'RMSE':<12} | {rmse_p:.2f} Pa{'':<16} | {rmse_f:.2f} Pa")
    print(f"{'MAE':<12} | {mae_p:.2f} Pa{'':<16} | {mae_f:.2f} Pa")
    print(f"{'Log-RMSE':<12} | {log_rmse_p:.4f}{'':<19} | {log_rmse_f:.4f}")
    print("="*75)

    print("\nPhysical Parameters (Mean):")
    print(f"Phi_m:    {np.mean(phi_m):.4f}")
    print(f"Phi_peak: {np.mean(phi_peak):.4f} (vs Phi_final: {np.mean(X[:,0]):.4f})")
    print(f"Delta:    {np.mean(delta):.4f}")

    # 5. 保存预测结果
    df['Tau0_Peak_Pred'] = y_pred_peak
    df['Tau0_Final_Pred'] = y_pred_final
    df['Phi_peak_Pred'] = phi_peak
    df['Phi_m_Pred'] = phi_m

    result_path = "data/synthetic/test_results_dual.csv"
    df.to_csv(result_path, index=False)
    print(f"\nDetailed results saved to {result_path}")

    # 6. 绘制评估图表
    plot_dual_results(y_true_peak, y_pred_peak, y_true_final, y_pred_final)

def plot_dual_results(true_p, pred_p, true_f, pred_f):
    """绘制双阶段评估图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Peak Stage: True vs Pred
    axes[0, 0].scatter(true_p, pred_p, alpha=0.6, color='red', s=15)
    axes[0, 0].plot([true_p.min(), true_p.max()], [true_p.min(), true_p.max()], 'k--', lw=2)
    axes[0, 0].set_title('Peak Stage: True vs Predicted')
    axes[0, 0].set_xlabel('True Tau0 (Pa)')
    axes[0, 0].set_ylabel('Predicted Tau0 (Pa)')
    axes[0, 0].grid(True)
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')

    # 2. Final Stage: True vs Pred
    axes[0, 1].scatter(true_f, pred_f, alpha=0.6, color='blue', s=15)
    axes[0, 1].plot([true_f.min(), true_f.max()], [true_f.min(), true_f.max()], 'k--', lw=2)
    axes[0, 1].set_title('Final Stage: True vs Predicted')
    axes[0, 1].set_xlabel('True Tau0 (Pa)')
    axes[0, 1].set_ylabel('Predicted Tau0 (Pa)')
    axes[0, 1].grid(True)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')

    # 3. Peak Residuals
    res_p = (pred_p - true_p) / (true_p + 1e-6) * 100 # Relative Error %
    axes[1, 0].hist(res_p, bins=30, color='red', alpha=0.7)
    axes[1, 0].set_title('Peak Stage: Relative Error Distribution')
    axes[1, 0].set_xlabel('Relative Error (%)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True)

    # 4. Final Residuals
    res_f = (pred_f - true_f) / (true_f + 1e-6) * 100
    axes[1, 1].hist(res_f, bins=30, color='blue', alpha=0.7)
    axes[1, 1].set_title('Final Stage: Relative Error Distribution')
    axes[1, 1].set_xlabel('Relative Error (%)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True)

    plt.tight_layout()

    plot_dir = "data/synthetic/plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "test_evaluation_dual.png")
    plt.savefig(save_path)
    print(f"Evaluation plots saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    test()

