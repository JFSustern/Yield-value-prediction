"""
低保真度模型训练脚本
使用对齐后的合成数据训练 PINN 模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multi_fidelity.src.model.pinn import YodelPINN


class LowFidelityTrainer:
    """低保真度模型训练器"""

    def __init__(self,
                 train_data_path='data/synthetic_aligned/train_data.csv',
                 test_data_path='data/synthetic_aligned/test_data.csv',
                 model_save_path='multi_fidelity/models/low_fidelity/pinn_low.pth',
                 device='cpu'):
        """
        Args:
            train_data_path: 训练数据路径
            test_data_path: 测试数据路径
            model_save_path: 模型保存路径
            device: 训练设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 路径设置
        self.train_data_path = project_root / train_data_path
        self.test_data_path = project_root / test_data_path
        self.model_save_path = project_root / model_save_path

        # 加载数据
        self.load_data()

        # 创建模型
        self.model = YodelPINN(input_dim=5, hidden_dim=128).to(self.device)
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'lr': []
        }

    def load_data(self):
        """加载训练和测试数据"""
        print("\n" + "="*60)
        print("加载数据")
        print("="*60)

        # 加载训练数据
        df_train = pd.read_csv(self.train_data_path)
        print(f"训练集: {len(df_train)} 样本")
        print(f"  Phi: {df_train['Phi'].min():.3f} - {df_train['Phi'].max():.3f}")
        print(f"  d50: {df_train['d50'].min():.2f} - {df_train['d50'].max():.2f} μm")
        print(f"  Tau0: {df_train['Tau0'].min():.2f} - {df_train['Tau0'].max():.2f} Pa")

        # 加载测试数据
        df_test = pd.read_csv(self.test_data_path)
        print(f"测试集: {len(df_test)} 样本")

        # 转换为 Tensor
        self.X_train = torch.tensor(
            df_train[['Phi', 'd50', 'sigma', 'Emix', 'Temp']].values,
            dtype=torch.float32
        ).to(self.device)

        self.y_train = torch.tensor(
            df_train['Tau0'].values,
            dtype=torch.float32
        ).to(self.device)

        self.X_test = torch.tensor(
            df_test[['Phi', 'd50', 'sigma', 'Emix', 'Temp']].values,
            dtype=torch.float32
        ).to(self.device)

        self.y_test = torch.tensor(
            df_test['Tau0'].values,
            dtype=torch.float32
        ).to(self.device)

        print("✅ 数据加载完成")

    def log_mse_loss(self, pred, target):
        """Log-MSE Loss 函数"""
        return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)

    def train_epoch(self, batch_size=64):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        # 随机打乱
        indices = torch.randperm(len(self.X_train))

        for i in range(0, len(self.X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = self.X_train[batch_indices]
            y_batch = self.y_train[batch_indices]

            # 前向传播
            self.optimizer.zero_grad()
            tau0_pred, physics_params = self.model(X_batch)

            # 计算 Loss
            loss = self.log_mse_loss(tau0_pred, y_batch)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            tau0_pred, _ = self.model(self.X_test)
            loss = self.log_mse_loss(tau0_pred, self.y_test)

            # 计算 R²
            y_mean = self.y_test.mean()
            ss_tot = torch.sum((self.y_test - y_mean) ** 2)
            ss_res = torch.sum((self.y_test - tau0_pred) ** 2)
            r2 = 1 - ss_res / ss_tot

            # 计算 MAE
            mae = torch.mean(torch.abs(tau0_pred - self.y_test))

        return loss.item(), r2.item(), mae.item()

    def train(self, epochs=200, batch_size=64, early_stop_patience=20):
        """训练模型"""
        print("\n" + "="*60)
        print("开始训练低保真度模型")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Early Stop Patience: {early_stop_patience}")
        print("="*60)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(batch_size)

            # 评估
            test_loss, r2, mae = self.evaluate()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # 学习率调度
            self.scheduler.step(test_loss)

            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Test Loss: {test_loss:.4f} | "
                      f"R²: {r2:.4f} | "
                      f"MAE: {mae:.2f} Pa | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Early Stopping
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print("\n" + "="*60)
        print("✅ 训练完成!")
        print(f"最佳测试 Loss: {best_loss:.4f}")
        print(f"模型已保存至: {self.model_save_path}")
        print("="*60)

    def save_model(self):
        """保存模型"""
        os.makedirs(self.model_save_path.parent, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, self.model_save_path)

    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss 曲线
        ax = axes[0]
        ax.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(self.history['test_loss'], label='Test Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Log-MSE Loss', fontsize=12)
        ax.set_title('Training History (Low Fidelity)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 学习率曲线
        ax = axes[1]
        ax.plot(self.history['lr'], linewidth=2, color='orange')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plot_path = project_root / 'multi_fidelity/results/plots/low_fidelity_training.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存至: {plot_path}")
        plt.close()


def main():
    """主函数"""
    print("="*60)
    print("低保真度 PINN 模型训练")
    print("="*60)

    # 创建训练器
    trainer = LowFidelityTrainer(
        train_data_path='data/synthetic_aligned/train_data.csv',
        test_data_path='data/synthetic_aligned/test_data.csv',
        model_save_path='multi_fidelity/models/low_fidelity/pinn_low.pth',
        device='cpu'
    )

    # 训练模型
    trainer.train(epochs=200, batch_size=64, early_stop_patience=20)

    # 绘制训练历史
    trainer.plot_training_history()

    print("\n✅ 低保真度模型训练完成!")
    print("下一步: 使用高保真度数据进行微调")


if __name__ == '__main__':
    main()
