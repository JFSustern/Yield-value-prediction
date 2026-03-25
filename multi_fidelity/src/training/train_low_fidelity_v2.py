"""
低保真度模型训练脚本 v2
模型: LianPINN (基于论文 Lian et al. 2025 公式)
数据: data/synthetic_table6/ (与论文 Table 6 对齐的合成数据)
输入: [Phi, d50_um, sigma, SP_percent] (4维)
输出: Tau0_Pa
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


class LowFidelityTrainerV2:

    def __init__(self,
                 train_data_path='data/synthetic_table6/train_data.csv',
                 test_data_path='data/synthetic_table6/test_data.csv',
                 model_save_path='multi_fidelity/models/low_fidelity/lian_low.pth',
                 device='cpu'):

        self.device = torch.device(device)
        self.model_save_path = project_root / model_save_path

        self._load_data(project_root / train_data_path, project_root / test_data_path)

        self.model = LianPINN(input_dim=4, hidden_dim=128).to(self.device)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"模型: LianPINN  |  参数量: {total:,}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        self.history = {'train_loss': [], 'test_loss': [], 'test_r2': [], 'test_mae': [], 'lr': []}

    def _load_data(self, train_path, test_path):
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)

        FEATURES = ['Phi', 'd50_um', 'sigma', 'SP_percent']
        TARGET   = 'Tau0_Pa'

        self.X_train = torch.tensor(df_train[FEATURES].values, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(df_train[TARGET].values,   dtype=torch.float32).to(self.device)
        self.X_test  = torch.tensor(df_test[FEATURES].values,  dtype=torch.float32).to(self.device)
        self.y_test  = torch.tensor(df_test[TARGET].values,    dtype=torch.float32).to(self.device)

        print(f"\n训练集: {len(df_train)} 样本  |  测试集: {len(df_test)} 样本")
        print(f"  Phi:  {df_train.Phi.min():.3f} – {df_train.Phi.max():.3f}")
        print(f"  SP%:  {df_train.SP_percent.min():.2f} – {df_train.SP_percent.max():.2f}")
        print(f"  Tau0: {df_train.Tau0_Pa.min():.3f} – {df_train.Tau0_Pa.max():.3f} Pa  "
              f"(均值 {df_train.Tau0_Pa.mean():.3f})")

    def _log_mse(self, pred, target):
        return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)

    def _train_epoch(self, batch_size):
        self.model.train()
        idx = torch.randperm(len(self.X_train))
        total, n = 0.0, 0
        for i in range(0, len(self.X_train), batch_size):
            b = idx[i:i+batch_size]
            self.optimizer.zero_grad()
            pred, _ = self.model(self.X_train[b])
            loss = self._log_mse(pred, self.y_train[b])
            loss.backward()
            self.optimizer.step()
            total += loss.item(); n += 1
        return total / n

    def _evaluate(self):
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.model(self.X_test)
            loss = self._log_mse(pred, self.y_test)
            y_mean = self.y_test.mean()
            r2 = 1 - torch.sum((self.y_test - pred)**2) / torch.sum((self.y_test - y_mean)**2)
            mae = torch.mean(torch.abs(pred - self.y_test))
        return loss.item(), r2.item(), mae.item()

    def train(self, epochs=300, batch_size=64, patience=25):
        print(f"\n{'='*60}")
        print(f"开始低保真度训练  epochs={epochs}  batch={batch_size}  patience={patience}")
        print(f"{'='*60}")

        best_loss = float('inf')
        wait = 0
        t0 = time.time()

        for ep in range(1, epochs + 1):
            tr_loss = self._train_epoch(batch_size)
            te_loss, r2, mae = self._evaluate()
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(tr_loss)
            self.history['test_loss'].append(te_loss)
            self.history['test_r2'].append(r2)
            self.history['test_mae'].append(mae)
            self.history['lr'].append(lr)

            self.scheduler.step(te_loss)

            if ep % 20 == 0 or ep == 1:
                print(f"Epoch {ep:3d}/{epochs} | "
                      f"Train {tr_loss:.5f} | Test {te_loss:.5f} | "
                      f"R²={r2:.4f} | MAE={mae:.4f} Pa | lr={lr:.1e}")

            if te_loss < best_loss:
                best_loss = te_loss
                wait = 0
                self._save()
            else:
                wait += 1
                if wait >= patience:
                    print(f"\nEarly stop at epoch {ep}")
                    break

        elapsed = time.time() - t0
        # 最终评估
        te_loss, r2, mae = self._evaluate()
        print(f"\n{'='*60}")
        print(f"训练完成  耗时 {elapsed:.1f}s")
        print(f"最佳 Test Loss : {best_loss:.6f}")
        print(f"最终 R²        : {r2:.4f}")
        print(f"最终 MAE       : {mae:.4f} Pa")
        print(f"模型保存至     : {self.model_save_path}")
        print(f"{'='*60}")

        return {
            'best_test_loss': best_loss,
            'final_r2': r2,
            'final_mae': mae,
            'epochs_run': len(self.history['train_loss']),
            'elapsed_s': round(elapsed, 1)
        }

    def _save(self):
        os.makedirs(self.model_save_path.parent, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'model_class': 'LianPINN',
            'input_dim': 4,
            'hidden_dim': 128,
            'features': ['Phi', 'd50_um', 'sigma', 'SP_percent'],
            'target': 'Tau0_Pa'
        }, self.model_save_path)

    def plot(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['test_loss'],  label='Test')
        axes[0].set_title('Log-MSE Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history['test_r2'], color='green')
        axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[1].set_title('Test R²'); axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.history['test_mae'], color='orange')
        axes[2].set_title('Test MAE (Pa)'); axes[2].grid(True, alpha=0.3)

        plt.suptitle('Low Fidelity Training (LianPINN)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = project_root / 'multi_fidelity/results/plots/low_fidelity_v2.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"训练曲线保存至: {save_path}")
        plt.close()


if __name__ == '__main__':
    trainer = LowFidelityTrainerV2()
    metrics = trainer.train(epochs=300, batch_size=64, patience=25)
    trainer.plot()
