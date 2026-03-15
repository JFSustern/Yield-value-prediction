"""
Multi-Fidelity Neural Network Training Script

训练策略:
    Stage 1: 用低保真度数据(合成数据)训练 LowFidelityNet
    Stage 2: 冻结LF网络,用高保真度数据(真实数据)训练 NonlinearScalingNet
    Stage 3 (可选): 联合微调
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from src.model.mfnn import YodelMFNN


def load_low_fidelity_data(data_path="data/synthetic/train_data.csv"):
    """加载低保真度数据(合成数据)"""
    print("Loading low-fidelity data (synthetic)...")
    df = pd.read_csv(data_path)

    feature_cols = ['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)']
    target_col = 'Tau0(屈服应力_Pa)'

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    print(f"Loaded {len(df)} low-fidelity samples")
    return X, y


def load_high_fidelity_data(data_path="data/20251121处理后"):
    """
    加载高保真度数据(真实实验数据)

    TODO: 根据实际的真实数据格式进行调整
    这里提供一个示例框架
    """
    print("Loading high-fidelity data (real experiments)...")

    # 示例: 如果真实数据在Excel文件中
    # 需要根据实际情况调整
    try:
        # 查找所有Excel文件
        import glob
        excel_files = glob.glob(os.path.join(data_path, "*.xlsx"))

        if not excel_files:
            print(f"Warning: No Excel files found in {data_path}")
            return None, None

        # 读取并合并所有Excel文件
        dfs = []
        for file in excel_files:
            try:
                df = pd.read_excel(file)
                # 这里需要根据实际列名进行调整
                # 假设列名与合成数据一致
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if not dfs:
            print("Warning: No valid data loaded from Excel files")
            return None, None

        df_combined = pd.concat(dfs, ignore_index=True)

        # 提取特征和目标
        feature_cols = ['Phi(固含量)', 'd50(中位径_um)', 'sigma(几何标准差)', 'Emix(混合功_J)', 'Temp(温度_C)']
        target_col = 'Tau0(屈服应力_Pa)'

        # 检查列是否存在
        missing_cols = [col for col in feature_cols + [target_col] if col not in df_combined.columns]
        if missing_cols:
            print(f"Warning: Missing columns in real data: {missing_cols}")
            print(f"Available columns: {df_combined.columns.tolist()}")
            return None, None

        X = df_combined[feature_cols].values.astype(np.float32)
        y = df_combined[target_col].values.astype(np.float32)

        print(f"Loaded {len(df_combined)} high-fidelity samples")
        return X, y

    except Exception as e:
        print(f"Error loading high-fidelity data: {e}")
        return None, None


def train_stage1_low_fidelity(model, X_lf, y_lf, device, epochs=200, batch_size=64, lr=1e-3):
    """
    Stage 1: 训练低保真度网络

    Args:
        model: YodelMFNN模型
        X_lf: 低保真度输入
        y_lf: 低保真度目标
        device: 训练设备
        epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
    """
    print("\n" + "=" * 60)
    print("Stage 1: Training Low-Fidelity Network")
    print("=" * 60)

    # 准备数据
    X_tensor = torch.from_numpy(X_lf)
    y_tensor = torch.from_numpy(y_lf).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 仅优化低保真度网络
    optimizer = optim.Adam(model.lf_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'loss': [], 'phi_m': [], 'g_max': []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        phi_m_stats = []
        g_max_stats = []

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 仅使用低保真度网络
            pred, _, _, (pred_phi_m, _, pred_g_max) = model(batch_X, use_hf=False)

            # 收集统计
            phi_m_stats.extend(pred_phi_m.detach().cpu().numpy().flatten())
            g_max_stats.extend(pred_g_max.detach().cpu().numpy().flatten())

            # Log-MSE Loss
            loss = torch.mean((torch.log1p(pred) - torch.log1p(batch_y)) ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lf_net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        phi_m_mean = np.mean(phi_m_stats)
        g_max_mean = np.mean(g_max_stats)

        history['loss'].append(avg_loss)
        history['phi_m'].append(phi_m_mean)
        history['g_max'].append(g_max_mean)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Phi_m: {phi_m_mean:.3f} | G_max: {g_max_mean:.0f}")

    # 计算并设置低保真度均值
    model.eval()
    with torch.no_grad():
        X_all = torch.from_numpy(X_lf).to(device)
        y_lf_pred, _, _, _ = model(X_all, use_hf=False)
        y_lf_mean = y_lf_pred.mean().item()
        model.set_y_lf_mean(y_lf_mean)
        print(f"\nLow-fidelity mean set to: {y_lf_mean:.2f} Pa")

    return history


def train_stage2_high_fidelity(model, X_hf, y_hf, device, epochs=100, batch_size=16, lr=1e-4):
    """
    Stage 2: 训练高保真度缩放网络

    Args:
        model: YodelMFNN模型
        X_hf: 高保真度输入
        y_hf: 高保真度目标
        device: 训练设备
        epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
    """
    print("\n" + "=" * 60)
    print("Stage 2: Training High-Fidelity Scaling Network")
    print("=" * 60)

    # 冻结低保真度网络
    model.freeze_lf_net()
    print("Low-fidelity network frozen")

    # 准备数据
    X_tensor = torch.from_numpy(X_hf)
    y_tensor = torch.from_numpy(y_hf).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 仅优化缩放网络
    optimizer = optim.Adam(model.alpha_net.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'loss': [], 'alpha': []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        alpha_stats = []

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 使用高保真度修正
            pred_hf, pred_lf, alpha, _ = model(batch_X, use_hf=True)

            # 收集统计
            alpha_stats.extend(alpha.detach().cpu().numpy().flatten())

            # MSE Loss
            loss = criterion(pred_hf, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.alpha_net.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        alpha_mean = np.mean(alpha_stats)

        history['loss'].append(avg_loss)
        history['alpha'].append(alpha_mean)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Alpha: {alpha_mean:.3f}")

    return history


def train_stage3_joint_finetuning(model, X_hf, y_hf, device, epochs=50, batch_size=16, lr=1e-5):
    """
    Stage 3 (可选): 联合微调

    Args:
        model: YodelMFNN模型
        X_hf: 高保真度输入
        y_hf: 高保真度目标
        device: 训练设备
        epochs: 训练轮数
        batch_size: 批量大小
        lr: 学习率
    """
    print("\n" + "=" * 60)
    print("Stage 3: Joint Fine-tuning")
    print("=" * 60)

    # 解冻低保真度网络
    model.unfreeze_lf_net()
    print("Low-fidelity network unfrozen")

    # 准备数据
    X_tensor = torch.from_numpy(X_hf)
    y_tensor = torch.from_numpy(y_hf).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 优化整个模型
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {'loss': []}

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 使用高保真度修正
            pred_hf, _, _, _ = model(batch_X, use_hf=True)

            # MSE Loss
            loss = criterion(pred_hf, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    return history


def plot_training_history(history_stage1, history_stage2=None, history_stage3=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Stage 1 Loss
    axes[0].plot(history_stage1['loss'], label='Stage 1 (LF)', color='blue')
    if history_stage2:
        offset = len(history_stage1['loss'])
        axes[0].plot(range(offset, offset + len(history_stage2['loss'])),
                     history_stage2['loss'], label='Stage 2 (HF)', color='orange')
    if history_stage3:
        offset = len(history_stage1['loss']) + (len(history_stage2['loss']) if history_stage2 else 0)
        axes[0].plot(range(offset, offset + len(history_stage3['loss'])),
                     history_stage3['loss'], label='Stage 3 (Joint)', color='green')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Phi_m
    axes[1].plot(history_stage1['phi_m'], label='Phi_m', color='green')
    axes[1].set_title('Physical Parameter: Phi_m (Stage 1)')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True)

    # G_max
    axes[2].plot(history_stage1['g_max'], label='G_max', color='red')
    axes[2].set_title('Physical Parameter: G_max (Stage 1)')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(True)

    plt.tight_layout()
    plot_dir = "data/synthetic/plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, "mfnn_training_history.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nTraining history saved to {save_path}")
    plt.close()


def train_mfnn():
    """主训练函数"""
    print("=" * 60)
    print("Multi-Fidelity Neural Network Training")
    print("=" * 60)

    # 设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # 加载数据
    X_lf, y_lf = load_low_fidelity_data()
    X_hf, y_hf = load_high_fidelity_data()

    # 初始化模型
    model = YodelMFNN(input_dim=5, lf_hidden_dim=128, alpha_hidden_dim=64).to(device)

    # Stage 1: 训练低保真度网络
    history_stage1 = train_stage1_low_fidelity(
        model, X_lf, y_lf, device,
        epochs=200, batch_size=64, lr=1e-3
    )

    # 保存Stage 1模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mfnn_stage1.pth")
    print("\nStage 1 model saved to models/mfnn_stage1.pth")

    # Stage 2: 训练高保真度缩放网络 (如果有真实数据)
    history_stage2 = None
    history_stage3 = None

    if X_hf is not None and y_hf is not None:
        history_stage2 = train_stage2_high_fidelity(
            model, X_hf, y_hf, device,
            epochs=100, batch_size=16, lr=1e-4
        )

        # 保存Stage 2模型
        torch.save(model.state_dict(), "models/mfnn_stage2.pth")
        print("\nStage 2 model saved to models/mfnn_stage2.pth")

        # Stage 3 (可选): 联合微调
        # history_stage3 = train_stage3_joint_finetuning(
        #     model, X_hf, y_hf, device,
        #     epochs=50, batch_size=16, lr=1e-5
        # )
        #
        # torch.save(model.state_dict(), "models/mfnn_stage3.pth")
        # print("\nStage 3 model saved to models/mfnn_stage3.pth")
    else:
        print("\nWarning: No high-fidelity data available, skipping Stage 2 and Stage 3")
        print("Model will only use low-fidelity predictions")

    # 保存最终模型
    torch.save(model.state_dict(), "models/yodel_mfnn.pth")
    print("\nFinal model saved to models/yodel_mfnn.pth")

    # 绘制训练历史
    plot_training_history(history_stage1, history_stage2, history_stage3)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    train_mfnn()
