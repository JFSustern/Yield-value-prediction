
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
        print(f"Error: Training data not found at {data_path}")
        return

    try:
        train_df = pd.read_csv(data_path)
        print(f"Loaded {len(train_df)} samples from {data_path}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 输入特征: [Phi, d50, sigma, Emix, Temp]
    feature_cols = ['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)']
    # 目标值: [Tau0]
    target_col = 'Tau0(屈服应力_Pa)'

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_col].values.astype(np.float32)

    # 转换为 Tensor
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).unsqueeze(1) # [batch, 1]

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

    # input_dim=5 (Phi, d50, sigma, Emix, Temp)
    model = YodelPINN(input_dim=5, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. 训练循环
    print("Step 3: Starting Training...")
    epochs = 200

    history = {
        'loss': [],
        'phi_m': [],
        'g_max': []
    }

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        phi_m_stats = []
        g_max_stats = []

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward
            pred_tau0, (pred_phi_m, pred_m1, pred_g_max) = model(batch_X)

            # 收集统计信息
            phi_m_stats.extend(pred_phi_m.detach().cpu().numpy().flatten())
            g_max_stats.extend(pred_g_max.detach().cpu().numpy().flatten())

            if pred_tau0.dim() == 1:
                pred_tau0 = pred_tau0.unsqueeze(1)

            # Loss Calculation
            # Log-MSE: Better for long-tail distribution (Tau0 ranges from 10 to 5000)
            loss_mse = torch.mean((torch.log1p(pred_tau0) - torch.log1p(batch_y)) ** 2)

            # 物理约束: Phi_m > Phi (Implicitly handled by model structure, but can add reg if needed)
            # loss_reg = torch.mean(torch.relu(batch_X[:, 0] - pred_phi_m)) * 10.0

            loss = loss_mse

            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        phi_m_mean = np.mean(phi_m_stats)
        g_max_mean = np.mean(g_max_stats)

        history['loss'].append(avg_loss)
        history['phi_m'].append(phi_m_mean)
        history['g_max'].append(g_max_mean)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Phi_m: {phi_m_mean:.3f} | G_max: {g_max_mean:.0f}")

    # 4. 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/yodel_pinn.pth")
    print("Model saved to models/yodel_pinn.pth")

    # 5. 绘制训练历史
    plot_training_history(history)

def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['loss'], label='Training Loss', color='blue')
    axes[0].set_title('Scaled MSE Loss History')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True)

    # Phi_m
    axes[1].plot(history['phi_m'], label='Avg Phi_m', color='green')
    axes[1].set_title('Physical Parameter: Phi_m')
    axes[1].grid(True)

    # G_max
    axes[2].plot(history['g_max'], label='Avg G_max', color='red')
    axes[2].set_title('Physical Parameter: G_max')
    axes[2].grid(True)

    plt.tight_layout()
    plot_dir = "data/synthetic/plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "training_history.png")
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    train()

