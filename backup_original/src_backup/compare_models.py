import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error

from src.model.baseline import PureNN
from src.model.pinn import YodelPINN


def compare_models():
    print("=== Comparing PINN vs Pure NN ===")

    # 1. 加载数据
    test_path = "data/synthetic/test_data.csv"
    df = pd.read_csv(test_path)

    feature_cols = ['Phi_final(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)',
                   'Emix(混合功_J)', 'Temp(温度_C)', 'Curing_Ratio(固化剂比例)']
    target_cols = ['Tau0_peak(高峰屈服_Pa)', 'Tau0_final(最终屈服_Pa)']

    X = torch.from_numpy(df[feature_cols].values.astype(np.float32))
    y_true = df[target_cols].values.astype(np.float32)

    # 2. 加载模型
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # PINN
    pinn = YodelPINN(input_dim=6, hidden_dim=128).to(device)
    pinn.load_state_dict(torch.load("models/yodel_pinn_dual.pth", map_location=device))
    pinn.eval()

    # Pure NN
    baseline = PureNN(input_dim=6, hidden_dim=128).to(device)
    baseline.load_state_dict(torch.load("models/pure_nn.pth", map_location=device))
    baseline.eval()

    # 3. 预测
    X = X.to(device)
    with torch.no_grad():
        pred_pinn, _ = pinn(X)
        pred_baseline = baseline(X)

        pred_pinn = pred_pinn.cpu().numpy()
        pred_baseline = pred_baseline.cpu().numpy()

    # 4. 评估指标
    def get_metrics(true, pred, name):
        log_rmse = np.sqrt(mean_squared_error(np.log1p(true), np.log1p(pred)))
        log_r2 = r2_score(np.log1p(true), np.log1p(pred))
        return log_rmse, log_r2

    print("\n" + "="*60)
    print(f"{'Model':<10} | {'Stage':<10} | {'Log-RMSE (Lower is better)':<25} | {'Log-R2'}")
    print("-" * 60)

    # PINN Metrics
    rmse_p_peak, r2_p_peak = get_metrics(y_true[:,0], pred_pinn[:,0], "PINN Peak")
    rmse_p_final, r2_p_final = get_metrics(y_true[:,1], pred_pinn[:,1], "PINN Final")

    # Baseline Metrics
    rmse_b_peak, r2_b_peak = get_metrics(y_true[:,0], pred_baseline[:,0], "Base Peak")
    rmse_b_final, r2_b_final = get_metrics(y_true[:,1], pred_baseline[:,1], "Base Final")

    print(f"{'PINN':<10} | {'Peak':<10} | {rmse_p_peak:.4f}{'':<21} | {r2_p_peak:.4f}")
    print(f"{'Pure NN':<10} | {'Peak':<10} | {rmse_b_peak:.4f}{'':<21} | {r2_b_peak:.4f}")
    print("-" * 60)
    print(f"{'PINN':<10} | {'Final':<10} | {rmse_p_final:.4f}{'':<21} | {r2_p_final:.4f}")
    print(f"{'Pure NN':<10} | {'Final':<10} | {rmse_b_final:.4f}{'':<21} | {r2_b_final:.4f}")
    print("="*60)

    # 5. 绘图对比
    plt.figure(figsize=(12, 5))

    # Peak Stage Comparison
    plt.subplot(1, 2, 1)
    plt.scatter(y_true[:,0], pred_pinn[:,0], alpha=0.5, label=f'PINN (R2={r2_p_peak:.2f})', c='blue')
    plt.scatter(y_true[:,0], pred_baseline[:,0], alpha=0.5, label=f'Pure NN (R2={r2_b_peak:.2f})', c='red', marker='x')
    plt.plot([y_true[:,0].min(), y_true[:,0].max()], [y_true[:,0].min(), y_true[:,0].max()], 'k--')
    plt.xscale('log'); plt.yscale('log')
    plt.title('Peak Stage Comparison')
    plt.xlabel('True Tau0'); plt.ylabel('Predicted Tau0')
    plt.legend()

    # Final Stage Comparison
    plt.subplot(1, 2, 2)
    plt.scatter(y_true[:,1], pred_pinn[:,1], alpha=0.5, label=f'PINN (R2={r2_p_final:.2f})', c='blue')
    plt.scatter(y_true[:,1], pred_baseline[:,1], alpha=0.5, label=f'Pure NN (R2={r2_b_final:.2f})', c='red', marker='x')
    plt.plot([y_true[:,1].min(), y_true[:,1].max()], [y_true[:,1].min(), y_true[:,1].max()], 'k--')
    plt.xscale('log'); plt.yscale('log')
    plt.title('Final Stage Comparison')
    plt.xlabel('True Tau0'); plt.ylabel('Predicted Tau0')
    plt.legend()

    plt.tight_layout()
    plt.savefig("data/synthetic/plots/model_comparison.png")
    print("\nComparison plot saved to data/synthetic/plots/model_comparison.png")

if __name__ == "__main__":
    compare_models()

