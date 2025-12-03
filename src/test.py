# src/test.py

import os

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

if __name__ == "__main__":
    test()

