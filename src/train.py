
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.model.pinn import YodelPINN


def train():
    # 1. 数据准备
    print("Step 1: Loading Training Data...")
    data_path = "data/synthetic/train_data.csv"

    if not os.path.exists(data_path):
        data_path = "data/synthetic/dataset.csv"

    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please run 'python main.py generate' first.")
        return

    try:
        train_df = pd.read_csv(data_path)
        print(f"Loaded {len(train_df)} samples from {data_path}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 输入特征: [Phi_final, d50, sigma, Emix, Temp, Ratio_curing]
    feature_cols = ['Phi_final(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)',
                   'Emix(混合功_J)', 'Temp(温度_C)', 'Curing_Ratio(固化剂比例)']

    # 目标值: [Tau0_peak, Tau0_final]
    target_cols = ['Tau0_peak(高峰屈服_Pa)', 'Tau0_final(最终屈服_Pa)']

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_cols].values.astype(np.float32)

    # 转换为 Tensor
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y) # [batch, 2]

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. 模型初始化
    print("Step 2: Initializing Model...")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = YodelPINN(input_dim=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. 训练循环
    print("Step 3: Starting Training...")
    epochs = 100

    history = {
        'loss': [],
        'loss_peak': [],
        'loss_final': [],
        'phi_m': [],
        'g_max': []
    }

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_loss_peak = 0
        epoch_loss_final = 0

        phi_m_stats = []
        g_max_stats = []

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward -> [batch, 2]
            pred_tau0, (pred_phi_m, pred_m1, pred_g_max, pred_phi_peak) = model(batch_X)

            phi_m_stats.extend(pred_phi_m.detach().cpu().numpy().flatten())
            g_max_stats.extend(pred_g_max.detach().cpu().numpy().flatten())

            # Loss Calculation
            # Scale targets to keep loss in reasonable range (e.g. / 100.0)
            # Peak values are larger (~2000), Final (~500)

            pred_peak = pred_tau0[:, 0]
            pred_final = pred_tau0[:, 1]
            true_peak = batch_y[:, 0]
            true_final = batch_y[:, 1]

            loss_peak = criterion(pred_peak / 100.0, true_peak / 100.0)
            loss_final = criterion(pred_final / 100.0, true_final / 100.0)

            # 物理约束: Phi_m > Phi_peak (已经在模型内部通过结构保证，这里作为辅助)
            loss_reg = torch.mean(torch.relu(pred_phi_peak - pred_phi_m)) * 100.0

            loss = loss_peak + loss_final + loss_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_peak += loss_peak.item()
            epoch_loss_final += loss_final.item()

        avg_loss = epoch_loss / len(loader)
        avg_loss_peak = epoch_loss_peak / len(loader)
        avg_loss_final = epoch_loss_final / len(loader)

        phi_m_mean = np.mean(phi_m_stats)
        g_max_mean = np.mean(g_max_stats)

        history['loss'].append(avg_loss)
        history['loss_peak'].append(avg_loss_peak)
        history['loss_final'].append(avg_loss_final)
        history['phi_m'].append(phi_m_mean)
        history['g_max'].append(g_max_mean)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (P:{avg_loss_peak:.2f}, F:{avg_loss_final:.2f}) | "
                  f"Phi_m: {phi_m_mean:.3f}")

    # 4. 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/yodel_pinn_dual.pth")
    print("Model saved to models/yodel_pinn_dual.pth")

    # 5. 绘制训练历史
    plot_training_history(history)

    # 6. 简单验证
    model.eval()
    with torch.no_grad():
        sample_X = X_tensor[:5].to(device)
        sample_y = y_tensor[:5].to(device)
        pred, params = model(sample_X)

        print("\nValidation Samples (True vs Pred):")
        print("   Peak(True) | Peak(Pred) || Final(True) | Final(Pred)")
        for i in range(5):
            print(f"   {sample_y[i,0]:.0f}       | {pred[i,0]:.0f}       || {sample_y[i,1]:.0f}        | {pred[i,1]:.0f}")

def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['loss'], label='Total Loss', color='black')
    axes[0].plot(history['loss_peak'], label='Peak Loss', color='red', linestyle='--')
    axes[0].plot(history['loss_final'], label='Final Loss', color='blue', linestyle='--')
    axes[0].set_title('Loss History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Scaled MSE Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Phi_m
    axes[1].plot(history['phi_m'], label='Avg Phi_m', color='green')
    axes[1].set_title('Physical Parameter: Phi_m')
    axes[1].grid(True)

    # G_max
    axes[2].plot(history['g_max'], label='Avg G_max', color='purple')
    axes[2].set_title('Physical Parameter: G_max')
    axes[2].grid(True)

    plt.tight_layout()
    plot_dir = "data/synthetic/plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "training_history_dual.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    train()

