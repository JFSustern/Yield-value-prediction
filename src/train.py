# src/train.py

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
        # 如果没有拆分好的训练集，尝试读取完整数据集
        data_path = "data/synthetic/dataset.csv"

    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        print("Please run 'python main.py generate' first to create the dataset.")
        return

    try:
        train_df = pd.read_csv(data_path)
        print(f"Loaded {len(train_df)} samples from {data_path}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    X = train_df[['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)']].values.astype(np.float32)
    y = train_df['Tau0(屈服应力_Pa)'].values.astype(np.float32)

    # 转换为 Tensor
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. 模型初始化
    print("Step 2: Initializing Model...")

    # 设备选择: MPS (Mac) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    model = YodelPINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. 训练循环
    print("Step 3: Starting Training...")
    epochs = 100

    # 记录历史数据
    history = {
        'loss': [],
        'phi_m': [],
        'g_max': []
    }

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0

        # 监控物理参数的统计值
        phi_m_stats = []
        g_max_stats = []

        for batch_X, batch_y in loader:
            # 移动数据到设备
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Forward
            pred_tau0, (pred_phi_m, pred_m1, pred_g_max) = model(batch_X)

            # 收集统计信息 (detach并转为numpy)
            phi_m_stats.extend(pred_phi_m.detach().cpu().numpy().flatten())
            g_max_stats.extend(pred_g_max.detach().cpu().numpy().flatten())

            # 确保维度匹配 [batch, 1]
            if pred_tau0.dim() == 1:
                pred_tau0 = pred_tau0.unsqueeze(1)

            # Loss 1: 预测误差 (Scaled MSE to prevent gradient explosion)
            # Tau0 range is 0-5000, so we scale by 100.0 to keep loss in reasonable range
            loss_mse = criterion(pred_tau0 / 100.0, batch_y / 100.0)

            # Loss 2: 物理约束 (可选)
            loss_reg = torch.mean(torch.relu(pred_phi_m - 0.74)) * 10.0

            loss = loss_mse + loss_reg

            loss.backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        # 计算物理参数的统计特征
        phi_m_mean = np.mean(phi_m_stats)
        g_max_mean = np.mean(g_max_stats)

        # 记录
        history['loss'].append(avg_loss)
        history['phi_m'].append(phi_m_mean)
        history['g_max'].append(g_max_mean)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Phi_m: {phi_m_mean:.3f} | G_max: {g_max_mean:.0f}")

    # 4. 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/yodel_pinn.pth")
    print("Model saved to models/yodel_pinn.pth")

    # 5. 绘制训练历史
    plot_training_history(history)

    # 6. 简单验证 (使用部分训练数据)
    model.eval()
    with torch.no_grad():
        # 取几个样本看预测
        sample_X = X_tensor[:5].to(device)
        sample_y = y_tensor[:5].to(device)
        pred, params = model(sample_X)

        print("\nValidation Samples (from Train Set):")
        for i in range(5):
            print(f"True: {sample_y[i].item():.2f}, Pred: {pred[i].item():.2f}, "
                  f"Phi_m: {params[0][i].item():.3f}, G_max: {params[2][i].item():.0f}")

def plot_training_history(history):
    """绘制训练过程中的 Loss 和物理参数变化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history['loss'], label='Training Loss', color='blue')
    axes[0].set_title('Loss History')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss (Pa^2)')
    axes[0].grid(True)

    # Phi_m
    axes[1].plot(history['phi_m'], label='Avg Phi_m', color='green')
    axes[1].set_title('Physical Parameter: Phi_m')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Max Packing Fraction')
    axes[1].grid(True)

    # G_max
    axes[2].plot(history['g_max'], label='Avg G_max', color='red')
    axes[2].set_title('Physical Parameter: G_max')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Interaction Force Parameter')
    axes[2].grid(True)

    plt.tight_layout()

    # 保存到 plots 目录
    plot_dir = "data/synthetic/plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "training_history.png")

    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    train()

