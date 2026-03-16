import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.model.baseline import PureNN


def train_baseline():
    print("=== Training Pure NN Baseline ===")

    # 1. 数据准备
    data_path = "data/synthetic/train_data.csv"
    if not os.path.exists(data_path):
        print("Error: Train data not found.")
        return

    train_df = pd.read_csv(data_path)

    feature_cols = ['Phi_final(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)',
                   'Emix(混合功_J)', 'Temp(温度_C)', 'Curing_Ratio(固化剂比例)']
    target_cols = ['Tau0_peak(高峰屈服_Pa)', 'Tau0_final(最终屈服_Pa)']

    X = train_df[feature_cols].values.astype(np.float32)
    y = train_df[target_cols].values.astype(np.float32)

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. 模型初始化
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = PureNN(input_dim=6, hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3. 训练循环
    epochs = 300
    history = {'loss': []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            pred = model(batch_X)

            # Log-MSE Loss (与 PINN 保持一致)
            loss = torch.mean((torch.log1p(pred) - torch.log1p(batch_y)) ** 2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    # 4. 保存
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pure_nn.pth")
    print("Baseline model saved to models/pure_nn.pth")

    # 5. 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(history['loss'], label='Pure NN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log-MSE Loss')
    plt.title('Baseline Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig("data/synthetic/plots/baseline_history.png")

if __name__ == "__main__":
    train_baseline()

