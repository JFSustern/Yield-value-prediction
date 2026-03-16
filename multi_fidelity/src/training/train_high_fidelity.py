"""
高保真度微调脚本
加载低保真度模型,冻结神经网络层,仅微调物理层参数
使用论文 Table 5 数据进行微调
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


class HighFidelityTrainer:
    """高保真度微调训练器"""

    def __init__(self,
                 low_fidelity_model_path='multi_fidelity/models/low_fidelity/pinn_low.pth',
                 high_fidelity_data_path='data/high_fidelity/cement_paste_data.csv',
                 model_save_path='multi_fidelity/models/high_fidelity/pinn_high.pth',
                 device='cpu'):
        """
        Args:
            low_fidelity_model_path: 低保真度模型路径
            high_fidelity_data_path: 高保真度数据路径 (论文 Table 5)
            model_save_path: 微调后模型保存路径
            device: 训练设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 路径设置
        self.low_fidelity_model_path = project_root / low_fidelity_model_path
        self.high_fidelity_data_path = project_root / high_fidelity_data_path
        self.model_save_path = project_root / model_save_path

        # 加载低保真度模型
        self.load_low_fidelity_model()

        # 冻结神经网络层
        self.freeze_network_layers()

        # 加载高保真度数据
        self.load_data()

        # 优化器 (仅优化未冻结的参数,使用较小学习率)
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4  # 较小学习率,因为可训练参数更多
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

    def load_low_fidelity_model(self):
        """加载低保真度模型"""
        print("\n" + "="*60)
        print("加载低保真度模型")
        print("="*60)

        if not self.low_fidelity_model_path.exists():
            raise FileNotFoundError(
                f"低保真度模型不存在: {self.low_fidelity_model_path}\n"
                f"请先运行 train_low_fidelity.py 训练低保真度模型"
            )

        # 创建模型
        self.model = YodelPINN(input_dim=5, hidden_dim=128).to(self.device)

        # 加载权重
        checkpoint = torch.load(self.low_fidelity_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✅ 已加载低保真度模型: {self.low_fidelity_model_path}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def freeze_network_layers(self):
        """只冻结神经网络的第1层,保留后3层可训练"""
        print("\n" + "="*60)
        print("冻结神经网络层 (只冻结第1层)")
        print("="*60)

        # 只冻结神经网络的第1层 (保留后3层可训练)
        # self.model.net 结构: [0] Linear(5→128), [1] Tanh, [2] Linear(128→128), [3] Tanh, [4] Linear(128→128), [5] Tanh, [6] Linear(128→2)
        # 冻结 [0-1],保留 [2-6] 可训练
        for i, layer in enumerate(self.model.net):
            if i < 2:  # 只冻结第1层及其激活函数
                for param in layer.parameters():
                    param.requires_grad = False

        # 统计可训练参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"总参数量: {total_params:,}")
        print(f"冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

        # 打印可训练层信息
        print("\n可训练层:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")

        print("\n✅ 神经网络第1层已冻结,后3层保持可训练")

    def load_data(self):
        """加载高保真度数据 (论文 Table 5)"""
        print("\n" + "="*60)
        print("加载高保真度数据 (论文 Table 5)")
        print("="*60)

        # 加载数据
        df = pd.read_csv(self.high_fidelity_data_path)
        print(f"样本数: {len(df)}")
        print(f"  Phi: {df['Phi'].min():.3f} - {df['Phi'].max():.3f}")
        print(f"  d50: {df['d50_um'].min():.2f} - {df['d50_um'].max():.2f} μm")
        print(f"  Tau0: {df['Tau0_Pa'].min():.2f} - {df['Tau0_Pa'].max():.2f} Pa")

        # 5-Fold 交叉验证划分
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        X = df[['Phi', 'd50_um', 'sigma', 'Emix_J', 'Temp_C']].values
        y = df['Tau0_Pa'].values

        # 使用第一个 fold 作为训练/验证
        train_idx, val_idx = list(kf.split(X))[0]

        self.X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y[train_idx], dtype=torch.float32).to(self.device)
        self.X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(y[val_idx], dtype=torch.float32).to(self.device)

        print(f"\n训练集: {len(self.X_train)} 样本")
        print(f"验证集: {len(self.X_val)} 样本")
        print("✅ 数据加载完成")

    def log_mse_loss(self, pred, target):
        """Log-MSE Loss 函数"""
        return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()

        # 前向传播
        self.optimizer.zero_grad()
        tau0_pred, physics_params = self.model(self.X_train)

        # 计算 Loss
        loss = self.log_mse_loss(tau0_pred, self.y_train)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            tau0_pred, _ = self.model(self.X_val)
            loss = self.log_mse_loss(tau0_pred, self.y_val)

            # 计算 R²
            y_mean = self.y_val.mean()
            ss_tot = torch.sum((self.y_val - y_mean) ** 2)
            ss_res = torch.sum((self.y_val - tau0_pred) ** 2)
            r2 = 1 - ss_res / ss_tot

            # 计算 MAE
            mae = torch.mean(torch.abs(tau0_pred - self.y_val))

        return loss.item(), r2.item(), mae.item()

    def train(self, epochs=200, early_stop_patience=30):
        """微调模型"""
        print("\n" + "="*60)
        print("开始高保真度微调")
        print("="*60)
        print(f"Epochs: {epochs}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"Early Stop Patience: {early_stop_patience}")
        print("="*60)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch()

            # 评估
            val_loss, r2, mae = self.evaluate()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # 打印进度
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"R²: {r2:.4f} | "
                      f"MAE: {mae:.2f} Pa")

            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        print("\n" + "="*60)
        print("✅ 微调完成!")
        print(f"最佳验证 Loss: {best_loss:.4f}")
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
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Log-MSE Loss', fontsize=12)
        ax.set_title('High Fidelity Fine-tuning History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plot_path = project_root / 'multi_fidelity/results/plots/high_fidelity_finetuning.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ 微调曲线已保存至: {plot_path}")
        plt.close()


def main():
    """主函数"""
    print("="*60)
    print("高保真度 PINN 模型微调")
    print("="*60)

    # 创建训练器
    trainer = HighFidelityTrainer(
        low_fidelity_model_path='multi_fidelity/models/low_fidelity/pinn_low.pth',
        high_fidelity_data_path='data/high_fidelity/cement_paste_data.csv',
        model_save_path='multi_fidelity/models/high_fidelity/pinn_high.pth',
        device='cpu'
    )

    # 微调模型 (增加 epochs 和 patience)
    trainer.train(epochs=200, early_stop_patience=30)

    # 绘制训练历史
    trainer.plot_training_history()

    print("\n✅ 高保真度模型微调完成!")
    print("多保真度学习流程全部完成!")


if __name__ == '__main__':
    main()
