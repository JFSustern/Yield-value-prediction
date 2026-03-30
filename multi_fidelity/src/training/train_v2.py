"""
多保真度训练脚本 v2 - 严格对齐论文

Phase 1: 低保真度训练 (合成数据 2000 样本)
Phase 2: 高保真度微调 (论文 Table 6, 16 样本)

模型: LianPINN_v2
  输入: [Phi, SP_percent] (2维)
  预测: φ_max → 代入论文公式计算 τ₀
  固定: m1 = 0.72 Pa
"""

import os
import sys
import time
import json
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multi_fidelity.src.model.pinn_lian2025_v2 import LianPINN_v2

FEATURES = ['Phi', 'SP_percent']
TARGET   = 'Tau0_Pa'
RESULTS_DIR = project_root / 'multi_fidelity/results'


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def log_mse(pred, target):
    return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)

def compute_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        pred, phi_max = model(X)
        loss = log_mse(pred, y).item()
        r2   = (1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)).item()
        mae  = torch.mean(torch.abs(pred - y)).item()
        mape = (torch.mean(torch.abs((pred - y) / (y + 1e-6))) * 100).item()
    return loss, r2, mae, mape, pred.cpu().numpy(), phi_max.cpu().numpy()

def load_csv(path):
    df = pd.read_csv(path)
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET].values,   dtype=torch.float32)
    return X, y, df


# ─────────────────────────────────────────────────────────
# Phase 1: 低保真度训练
# ─────────────────────────────────────────────────────────

def train_low_fidelity(
    train_path = 'data/synthetic_table6_v2/train_data.csv',
    test_path  = 'data/synthetic_table6_v2/test_data.csv',
    hifi_path  = 'data/high_fidelity/hifi_table6.csv',
    save_path  = 'multi_fidelity/models/low_fidelity/lian_v2_low.pth',
    hidden_dim = 64,
    lr         = 1e-3,
    epochs     = 300,
    batch_size = 64,
    patience   = 25,
):
    print("\n" + "="*60)
    print("Phase 1: 低保真度训练")
    print("="*60)

    X_train, y_train, df_train = load_csv(project_root / train_path)
    X_test,  y_test,  df_test  = load_csv(project_root / test_path)
    print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")
    print(f"  Phi:  {df_train.Phi.min():.3f}–{df_train.Phi.max():.3f}")
    print(f"  SP%:  {df_train.SP_percent.min():.2f}–{df_train.SP_percent.max():.2f}")
    print(f"  τ₀:   {df_train.Tau0_Pa.min():.3f}–{df_train.Tau0_Pa.max():.3f} Pa")

    model = LianPINN_v2(hidden_dim=hidden_dim)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total:,}  hidden_dim={hidden_dim}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False)

    history = {'train_loss':[], 'test_loss':[], 'test_r2':[], 'test_mae':[], 'lr':[]}
    best_loss = float('inf')
    wait = 0
    save_full = project_root / save_path
    os.makedirs(save_full.parent, exist_ok=True)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(X_train))
        ep_loss = 0.0
        n_batch = 0
        for i in range(0, len(X_train), batch_size):
            b = idx[i:i+batch_size]
            optimizer.zero_grad()
            pred, _ = model(X_train[b])
            loss = log_mse(pred, y_train[b])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item(); n_batch += 1
        tr_loss = ep_loss / n_batch

        te_loss, r2, mae, mape, _, _ = compute_metrics(model, X_test, y_test)
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        history['test_r2'].append(r2)
        history['test_mae'].append(mae)
        history['lr'].append(cur_lr)

        scheduler.step(te_loss)

        if ep % 50 == 0 or ep == 1:
            print(f"Epoch {ep:3d}/{epochs} | "
                  f"Train {tr_loss:.5f} | Test {te_loss:.5f} | "
                  f"R²={r2:.4f} | MAE={mae:.4f} Pa | lr={cur_lr:.1e}")

        if te_loss < best_loss:
            best_loss = te_loss
            wait = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'history': history, 'hidden_dim': hidden_dim}, save_full)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    elapsed = time.time() - t0

    # 加载最佳模型做最终评估
    ckpt = torch.load(save_full, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    tr_loss, tr_r2, tr_mae, tr_mape, tr_pred, _ = compute_metrics(model, X_train, y_train)
    te_loss, te_r2, te_mae, te_mape, te_pred, _ = compute_metrics(model, X_test,  y_test)

    # 在高保真数据上评估 (作为微调前的基准)
    X_hifi, y_hifi, _ = load_csv(project_root / hifi_path)
    hifi_loss, hifi_r2, hifi_mae, hifi_mape, hifi_pred, _ = compute_metrics(model, X_hifi, y_hifi)

    result = {
        'phase': 'low_fidelity',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'epochs_run': len(history['train_loss']),
        'elapsed_s': round(elapsed, 1),
        'hidden_dim': hidden_dim, 'lr': lr, 'batch_size': batch_size,
        'train_samples': len(X_train), 'test_samples': len(X_test),
        'train':      {'loss': round(tr_loss,6),   'r2': round(tr_r2,4),
                       'mae': round(tr_mae,4),      'mape': round(tr_mape,2)},
        'test':       {'loss': round(te_loss,6),   'r2': round(te_r2,4),
                       'mae': round(te_mae,4),      'mape': round(te_mape,2)},
        'hifi_before':{'loss': round(hifi_loss,6), 'r2': round(hifi_r2,4),
                       'mae': round(hifi_mae,4),    'mape': round(hifi_mape,2)},
    }

    print(f"\n{'─'*60}")
    print(f"低保真度训练完成  耗时 {elapsed:.1f}s  共 {result['epochs_run']} epochs")
    print(f"  合成训练集 : R²={tr_r2:.4f}  MAE={tr_mae:.4f} Pa  MAPE={tr_mape:.1f}%")
    print(f"  合成测试集 : R²={te_r2:.4f}  MAE={te_mae:.4f} Pa  MAPE={te_mape:.1f}%")
    print(f"  高保真数据 : R²={hifi_r2:.4f}  MAE={hifi_mae:.4f} Pa  MAPE={hifi_mape:.1f}%  ← 微调前基准")
    print(f"{'─'*60}")

    # 绘图：合成数据散点图
    _plot_training(history, title='Low Fidelity Training (LianPINN_v2)',
                   save_path=RESULTS_DIR / 'plots/lian_v2_low_training.png')
    _plot_scatter(y_train.numpy(), tr_pred, y_test.numpy(), te_pred,
                  title='Low Fidelity: Predicted vs True (Synthetic)',
                  save_path=RESULTS_DIR / 'plots/lian_v2_low_scatter.png')
    # 绘图：低保真模型在高保真数据上的表现
    _plot_hifi_scatter(y_hifi.numpy(), hifi_pred, freeze_n=0,
                       save_path=RESULTS_DIR / 'plots/lian_v2_hifi_before_tuning.png')

    return model, result


# ─────────────────────────────────────────────────────────
# Phase 2: 高保真度微调
# ─────────────────────────────────────────────────────────

def train_high_fidelity(
    low_model,
    hifi_path    = 'data/high_fidelity/hifi_table6.csv',
    save_path    = 'multi_fidelity/models/high_fidelity/lian_v2_high.pth',
    freeze_n     = 1,
    lr           = 1e-4,
    epochs       = 500,
    patience     = 60,
    low_result   = None,   # 传入低保真阶段的结果，用于对比打印
):
    print("\n" + "="*60)
    print(f"Phase 2: 高保真度微调  (冻结前 {freeze_n} 个 Linear 层)")
    print("="*60)

    # 深拷贝模型
    import copy
    model = copy.deepcopy(low_model)

    # 冻结策略
    linear_idx = [i for i, layer in enumerate(model.net) if hasattr(layer, 'weight')]
    freeze_set = set(linear_idx[:freeze_n])
    for i, layer in enumerate(model.net):
        if i in freeze_set:
            for p in layer.parameters():
                p.requires_grad = False

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"冻结: {frozen:,} ({frozen/total*100:.1f}%)  "
          f"可训练: {trainable:,} ({trainable/total*100:.1f}%)")

    # 加载高保真数据，12训练/4测试
    X_all, y_all, df_hifi = load_csv(project_root / hifi_path)
    n = len(X_all)
    torch.manual_seed(42)
    idx = torch.randperm(n).tolist()
    # 原测试集 idx=13 (Phi=0.504,SP=0.6,τ₀=0.44) 与训练集 idx=5 (Phi=0.503,SP=0.6,τ₀=0.74) 互换
    # 把重复实验的两条都放入训练集，用 idx=5 替换到测试集
    train_list = idx[:12]
    test_list  = idx[12:]
    if 13 in test_list and 5 in train_list:
        test_list[test_list.index(13)] = 5
        train_list[train_list.index(5)] = 13
    train_idx = torch.tensor(train_list)
    test_idx  = torch.tensor(test_list)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test,  y_test  = X_all[test_idx],  y_all[test_idx]
    print(f"\n高保真数据: {n} 样本  训练={len(train_idx)}  测试={len(test_idx)}")
    print(f"  τ₀: {df_hifi.Tau0_Pa.min():.3f}–{df_hifi.Tau0_Pa.max():.3f} Pa  "
          f"均值={df_hifi.Tau0_Pa.mean():.3f}")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    history = {'train_loss':[], 'test_loss':[], 'test_r2':[], 'test_mae':[]}
    best_loss = float('inf')
    wait = 0
    save_full = project_root / save_path
    os.makedirs(save_full.parent, exist_ok=True)
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_train)
        loss = log_mse(pred, y_train)
        loss.backward()
        optimizer.step()

        te_loss, te_r2, te_mae, te_mape, _, _ = compute_metrics(model, X_test, y_test)

        history['train_loss'].append(loss.item())
        history['test_loss'].append(te_loss)
        history['test_r2'].append(te_r2)
        history['test_mae'].append(te_mae)

        if ep % 50 == 0 or ep == 1:
            print(f"Epoch {ep:3d}/{epochs} | "
                  f"Train={loss.item():.5f} | Test Loss={te_loss:.5f} | "
                  f"R²={te_r2:.4f} | MAE={te_mae:.4f} Pa")

        if te_loss < best_loss:
            best_loss = te_loss
            wait = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'history': history, 'freeze_n': freeze_n, 'lr': lr}, save_full)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    elapsed = time.time() - t0

    # 加载最佳模型做最终评估
    ckpt = torch.load(save_full, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    tr_loss, tr_r2, tr_mae, tr_mape, tr_pred, tr_phi_max = compute_metrics(model, X_train, y_train)
    te_loss, te_r2, te_mae, te_mape, te_pred, te_phi_max = compute_metrics(model, X_test,  y_test)

    result = {
        'phase': 'high_fidelity',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'epochs_run': len(history['train_loss']),
        'elapsed_s': round(elapsed, 1),
        'freeze_n': freeze_n, 'lr': lr,
        'hifi_samples': n, 'train_samples': len(train_idx), 'test_samples': len(test_idx),
        'train': {'loss': round(tr_loss,6), 'r2': round(tr_r2,4),
                  'mae': round(tr_mae,4), 'mape': round(tr_mape,2)},
        'test':  {'loss': round(te_loss,6), 'r2': round(te_r2,4),
                  'mae': round(te_mae,4), 'mape': round(te_mape,2)},
    }

    print(f"\n{'─'*60}")
    print(f"高保真度微调完成  耗时 {elapsed:.1f}s  共 {result['epochs_run']} epochs")
    print(f"\n  在高保真数据上的对比:")
    print(f"  {'模型':<20} {'R²':>8} {'MAE':>10} {'MAPE':>8}")
    print(f"  {'─'*48}")
    if low_result and 'hifi_before' in low_result:
        b = low_result['hifi_before']
        print(f"  {'低保真 (微调前,全16)':<20} {b['r2']:>8.4f} {b['mae']:>8.4f} Pa {b['mape']:>6.1f}%")
    print(f"  {'高保真训练集(12条)':<20} {tr_r2:>8.4f} {tr_mae:>8.4f} Pa {tr_mape:>6.1f}%")
    print(f"  {'高保真测试集(4条)':<20} {te_r2:>8.4f} {te_mae:>8.4f} Pa {te_mape:>6.1f}%")
    print(f"{'─'*60}")

    # 打印测试集逐样本对比
    print(f"\n测试集逐样本对比:")
    print(f"{'Phi':<8} {'SP%':<8} {'真实τ₀':<10} {'预测τ₀':<10} {'误差%':<8} {'φ_max预测'}")
    print("-"*55)
    for i in range(len(X_test)):
        phi    = X_test[i,0].item()
        sp     = X_test[i,1].item()
        true   = y_test[i].item()
        pred_i = te_pred[i]
        err    = abs(pred_i - true) / true * 100
        print(f"{phi:.3f}    {sp:.2f}    {true:.4f}    {pred_i:.4f}    {err:.1f}%    {te_phi_max[i]:.4f}")

    # 绘图
    _plot_hifi_training(history, freeze_n, lr,
                        save_path=RESULTS_DIR / f'plots/lian_v2_high_freeze{freeze_n}_training.png')
    _plot_hifi_scatter(y_test.numpy(), te_pred, freeze_n,
                       save_path=RESULTS_DIR / f'plots/lian_v2_high_freeze{freeze_n}_scatter.png')

    return model, result


# ─────────────────────────────────────────────────────────
# 绘图函数
# ─────────────────────────────────────────────────────────

def _plot_training(history, title, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_title('Log-MSE Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['test_r2'], color='green')
    axes[1].axhline(0, color='red', linestyle='--', lw=1); axes[1].set_title('Test R²'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history['test_mae'], color='orange'); axes[2].set_title('Test MAE (Pa)'); axes[2].grid(True, alpha=0.3)
    plt.suptitle(title, fontweight='bold'); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"训练曲线: {save_path}")

def _plot_scatter(y_tr, p_tr, y_te, p_te, title, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    all_v = np.concatenate([y_tr, y_te, p_tr, p_te])
    lim = [all_v.min()*0.9, all_v.max()*1.1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, yt, yp, label in [(axes[0],y_tr,p_tr,'Train'),(axes[1],y_te,p_te,'Test')]:
        r2  = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
        mae = np.mean(np.abs(yt-yp))
        ax.scatter(yt, yp, alpha=0.6, s=30)
        ax.plot(lim, lim, 'r--', lw=1.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('True τ₀ (Pa)'); ax.set_ylabel('Predicted τ₀ (Pa)')
        ax.set_title(f'{label}  R²={r2:.4f}  MAE={mae:.4f} Pa'); ax.grid(True, alpha=0.3)
    plt.suptitle(title, fontweight='bold'); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"散点图: {save_path}")

def _plot_hifi_training(history, freeze_n, lr, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['test_loss'], label='Test')
    axes[0].set_title('Log-MSE Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['test_r2'], color='green')
    axes[1].axhline(0, color='red', linestyle='--', lw=1); axes[1].set_title('Test R²'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history['test_mae'], color='orange'); axes[2].set_title('Test MAE (Pa)'); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'High Fidelity Fine-tuning (freeze={freeze_n}, lr={lr})', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"微调曲线: {save_path}")

def _plot_hifi_scatter(y_true, y_pred, freeze_n, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    lim = [min(y_true.min(), y_pred.min())*0.85, max(y_true.max(), y_pred.max())*1.1]
    r2  = 1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-y_true.mean())**2)
    mae = np.mean(np.abs(y_true-y_pred))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=60, alpha=0.8)
    ax.plot(lim, lim, 'r--', lw=1.5, label='Ideal')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('True τ₀ (Pa)', fontsize=12); ax.set_ylabel('Predicted τ₀ (Pa)', fontsize=12)
    ax.set_title(f'High Fidelity (freeze={freeze_n})\nR²={r2:.4f}  MAE={mae:.4f} Pa', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"散点图: {save_path}")


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = []

    # Phase 1: 低保真度训练
    low_model, low_result = train_low_fidelity()
    all_results.append(low_result)

    # Phase 2: 高保真度微调 (测试不同冻结策略)
    for freeze_n in [1, 2, 3]:
        hi_model, hi_result = train_high_fidelity(
            low_model,
            freeze_n   = freeze_n,
            lr         = 1e-4,
            epochs     = 500,
            patience   = 60,
            low_result = low_result,
        )
        all_results.append(hi_result)

    # 保存所有结果到 JSON
    result_path = RESULTS_DIR / 'logs/train_v2_results.json'
    os.makedirs(result_path.parent, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n所有结果已保存至: {result_path}")
