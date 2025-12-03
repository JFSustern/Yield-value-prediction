# src/train.py

import os

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
    data_path = "data/synthetic/train_data.csv" # 修改为读取训练集

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

    # 转换为 Tensor (保持原始物理量，归一化在 Model 内部处理)
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
    epochs = 50
    loss_history = []

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

            # Loss 1: 预测误差
            loss_mse = criterion(pred_tau0, batch_y)

            # Loss 2: 物理约束 (可选，因为我们在 forward 里已经强制了硬约束)
            # 这里可以加一些软约束，比如希望 Phi_m 不要太大
            loss_reg = torch.mean(torch.relu(pred_phi_m - 0.74)) * 10.0

            loss = loss_mse + loss_reg

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)

        # 计算物理参数的统计特征
        phi_m_mean = np.mean(phi_m_stats)
        g_max_mean = np.mean(g_max_stats)

        if (epoch + 1) % 5 == 0: # 每5轮打印一次
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Phi_m: {phi_m_mean:.3f} | G_max: {g_max_mean:.0f}")

    # 4. 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/yodel_pinn.pth")
    print("Model saved to models/yodel_pinn.pth")

    # 5. 简单验证
    model.eval()
    with torch.no_grad():
        # 取几个样本看预测
        sample_X = X_tensor[:5]
        sample_y = y_tensor[:5]
        pred, params = model(sample_X)

        print("\nValidation Samples:")
        for i in range(5):
            print(f"True: {sample_y[i].item():.2f}, Pred: {pred[i].item():.2f}, "
                  f"Phi_m: {params[0][i].item():.3f}, G_max: {params[2][i].item():.0f}")

if __name__ == "__main__":
    train()

