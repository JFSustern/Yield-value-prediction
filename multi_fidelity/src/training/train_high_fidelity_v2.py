"""
高保真度微调脚本 v2
加载低保真度 LianPINN，冻结前 N 层，用论文 Table 6 真实数据微调
输入: [Phi, d50_um, sigma, SP_percent] (4维)
高保真数据: data/high_fidelity/cement_paste_data.csv
"""

import os
import sys
import time
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multi_fidelity.src.model.pinn_lian2025 import LianPINN


class HighFidelityTrainerV2:

    def __init__(self,
                 low_fidelity_model_path='multi_fidelity/models/low_fidelity/lian_low.pth',
                 high_fidelity_data_path='data/high_fidelity/cement_paste_data.csv',
                 model_save_path='multi_fidelity/models/high_fidelity/lian_high.pth',
                 freeze_layers=1,
                 lr=1e-4,
                 device='cpu'):
        """
        Args:
            freeze_layers: 冻结前 N 个 Linear 层 (1=只冻第1层, 2=冻前2层, 3=冻前3层)
            lr: 微调学习率
        """
        self.device = torch.device(device)
        self.model_save_path = project_root / model_save_path
        self.freeze_layers = freeze_layers

        self._load_model(project_root / low_fidelity_model_path)
        self._freeze(freeze_layers)
        self._load_data(project_root / high_fidelity_data_path)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=lr)
        self.lr = lr
        self.history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_mae': []}

    def _load_model(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model = LianPINN(input_dim=4, hidden_dim=128).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        print(f"已加载低保真度模型: {path}")

    def _freeze(self, n):
        """冻结 self.model.net 中前 n 个 Linear 层"""
        # net 结构索引: 0=Linear, 1=Tanh, 2=Linear, 3=Tanh, 4=Linear, 5=Tanh, 6=Linear
        # Linear 层在偶数索引: 0,2,4,6
        linear_indices = [i for i, layer in enumerate(self.model.net)
                          if hasattr(layer, 'weight')]  # 只有 Linear 有 weight
        freeze_idx = set(linear_indices[:n])

        for i, layer in enumerate(self.model.net):
            if i in freeze_idx:
                for p in layer.parameters():
                    p.requires_grad = False

        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen    = total - trainable

        print(f"\n冻结策略: 前 {n} 个 Linear 层")
        print(f"  冻结索引: {sorted(freeze_idx)}")
        print(f"  总参数: {total:,}  |  冻结: {frozen:,} ({frozen/total*100:.1f}%)  "
              f"|  可训练: {trainable:,} ({trainable/total*100:.1f}%)")
        print("  可训练层:")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"    {name}: {list(p.shape)}")

    def _load_data(self, path):
        df = pd.read_csv(path)
        FEATURES = ['Phi', 'd50_um', 'sigma', 'SP_percent']
        TARGET   = 'Tau0_Pa'

        # 检查列是否存在，高保真数据可能列名不同
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"高保真数据缺少列: {missing}\n现有列: {df.columns.tolist()}")

        print(f"\n高保真度数据: {len(df)} 样本")
        print(f"  Phi:  {df.Phi.min():.3f} – {df.Phi.max():.3f}")
        print(f"  Tau0: {df[TARGET].min():.3f} – {df[TARGET].max():.3f} Pa  "
              f"(均值 {df[TARGET].mean():.3f})")

        # 8:2 划分（样本少，不用交叉验证）
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_train = max(1, int(len(df) * 0.8))

        X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
        y = torch.tensor(df[TARGET].values,   dtype=torch.float32)

        self.X_train = X[:n_train].to(self.device)
        self.y_train = y[:n_train].to(self.device)
        self.X_val   = X[n_train:].to(self.device)
        self.y_val   = y[n_train:].to(self.device)
        print(f"  训练: {len(self.X_train)}  验证: {len(self.X_val)}")

    def _log_mse(self, pred, target):
        return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)

    def _metrics(self, X, y):
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(X)
            loss = self._log_mse(pred, y)
            r2   = 1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)
            mae  = torch.mean(torch.abs(pred - y))
        return loss.item(), r2.item(), mae.item(), pred.cpu().numpy()

    def train(self, epochs=300, patience=40):
        print(f"\n{'='*60}")
        print(f"开始高保真度微调  lr={self.lr}  epochs={epochs}  patience={patience}")
        print(f"{'='*60}")

        best_loss = float('inf')
        wait = 0
        t0 = time.time()

        for ep in range(1, epochs + 1):
            # 训练
            self.model.train()
            self.optimizer.zero_grad()
            pred, _ = self.model(self.X_train)
            loss = self._log_mse(pred, self.y_train)
            loss.backward()
            self.optimizer.step()

            # 验证
            val_loss, r2, mae, _ = self._metrics(self.X_val, self.y_val)

            self.history['train_loss'].append(loss.item())
            self.history['val_loss'].append(val_loss)
            self.history['val_r2'].append(r2)
            self.history['val_mae'].append(mae)

            if ep % 20 == 0 or ep == 1:
                print(f"Epoch {ep:3d}/{epochs} | "
                      f"Train {loss.item():.5f} | Val {val_loss:.5f} | "
                      f"R²={r2:.4f} | MAE={mae:.4f} Pa")

            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                self._save()
            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stop at epoch {ep}")
                    break

        elapsed = time.time() - t0

        # 加载最佳模型做最终评估
        ckpt = torch.load(self.model_save_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])

        tr_loss, tr_r2, tr_mae, _ = self._metrics(self.X_train, self.y_train)
        va_loss, va_r2, va_mae, _ = self._metrics(self.X_val,   self.y_val)

        print(f"\n{'='*60}")
        print(f"微调完成  耗时 {elapsed:.1f}s")
        print(f"{'最佳模型评估':}")
        print(f"  训练集: Loss={tr_loss:.5f}  R²={tr_r2:.4f}  MAE={tr_mae:.4f} Pa")
        print(f"  验证集: Loss={va_loss:.5f}  R²={va_r2:.4f}  MAE={va_mae:.4f} Pa")
        print(f"  模型保存至: {self.model_save_path}")
        print(f"{'='*60}")

        return {
            'freeze_layers': self.freeze_layers,
            'lr': self.lr,
            'best_val_loss': best_loss,
            'train_r2': tr_r2,  'train_mae': tr_mae,
            'val_r2':   va_r2,  'val_mae':   va_mae,
            'epochs_run': len(self.history['train_loss']),
            'elapsed_s': round(elapsed, 1)
        }

    def _save(self):
        os.makedirs(self.model_save_path.parent, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'freeze_layers': self.freeze_layers,
            'lr': self.lr
        }, self.model_save_path)

    def plot(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'],   label='Val')
        axes[0].set_title('Log-MSE Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['val_r2'], color='green')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].set_title('Val R²'); axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.history['val_mae'], color='orange')
        axes[2].set_title('Val MAE (Pa)'); axes[2].grid(True, alpha=0.3)

        plt.suptitle(f'High Fidelity Fine-tuning  (freeze={self.freeze_layers} layers, lr={self.lr})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = project_root / f'multi_fidelity/results/plots/high_fidelity_v2_freeze{self.freeze_layers}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线保存至: {save_path}")
        plt.close()

    def scatter_plot(self, save_path=None):
        """预测值 vs 真实值散点图"""
        _, tr_r2, tr_mae, tr_pred = self._metrics(self.X_train, self.y_train)
        _, va_r2, va_mae, va_pred = self._metrics(self.X_val,   self.y_val)

        y_train_np = self.y_train.cpu().numpy()
        y_val_np   = self.y_val.cpu().numpy()
        all_y   = np.concatenate([y_train_np, y_val_np])
        all_lim = [all_y.min() * 0.9, all_y.max() * 1.1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, y_true, y_pred, r2, mae, title in [
            (axes[0], y_train_np, tr_pred, tr_r2, tr_mae, 'Train'),
            (axes[1], y_val_np,   va_pred, va_r2, va_mae, 'Validation')
        ]:
            ax.scatter(y_true, y_pred, alpha=0.7, s=60)
            ax.plot(all_lim, all_lim, 'r--', linewidth=1.5, label='Ideal')
            ax.set_xlim(all_lim); ax.set_ylim(all_lim)
            ax.set_xlabel('True τ₀ (Pa)'); ax.set_ylabel('Predicted τ₀ (Pa)')
            ax.set_title(f'{title}  R²={r2:.4f}  MAE={mae:.4f} Pa')
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('High Fidelity Model: Predicted vs True', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = project_root / f'multi_fidelity/results/plots/high_fidelity_v2_scatter_freeze{self.freeze_layers}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"散点图保存至: {save_path}")
        plt.close()


if __name__ == '__main__':
    trainer = HighFidelityTrainerV2(freeze_layers=1, lr=1e-4)
    metrics = trainer.train(epochs=300, patience=40)
    trainer.plot()
    trainer.scatter_plot()
    print("\n最终指标:", metrics)
