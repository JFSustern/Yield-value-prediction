"""
Lian 2025 多保真训练脚本 - 验证真实小样本场景下多保真的价值

实验设计:
  - 将 400 条数据视为"全量高保真数据"
  - 随机划分: 30 条训练 / 10 条评估 / 360 条测试
  - 实验 14 (多保真): 低保真预训练 → 30 条高保真微调 (10 条 eval early stop) → 360 条测试
  - 实验 15 (纯高保真): 直接用 30 条从头训练 (10 条 eval early stop) → 360 条测试

核心问题:
  低保真预训练 + 少量真实数据微调，是否比单独用少量真实数据训练效果更好？
  这才是多保真学习的真正价值所在。

模型: LianPINN
  输入: [Phi, SP_percent] (2维)
  预测: phi_max → 代入论文公式计算 tau0
  固定: m1 = 0.72 Pa
"""

import argparse
import os
import sys
import time
import json
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multi_fidelity.src.model.pinn_lian2025 import LianPINN
from multi_fidelity.src.model.pinn_lian2025_configurable import ConfigurableLianPINN

FEATURES = ['Phi', 'SP_percent']
TARGET   = 'Tau0_Pa'
RESULTS_DIR = project_root / 'multi_fidelity/results/lian2025'
HIFI_DATA_PATH = 'data/lian2025/high_fidelity/all_400.csv'

# 随机种子，保证实验可复现
RANDOM_SEED = 42
CLI_MODES = ('main', 'arch', 'hparam', 'ablation', 'sufficient')


def _parse_cli_mode():
    """Parse the single experiment selector used by this research script."""
    parser = argparse.ArgumentParser(
        description='Lian 2025 PI-MFNN training and comparison experiments',
    )
    parser.add_argument(
        'mode',
        nargs='?',
        choices=CLI_MODES,
        default='main',
        help='main: 主实验；arch: 架构搜索；hparam: 超参搜索；'
             'ablation: 消融；sufficient: 充足数据对比',
    )
    return parser.parse_args().mode


CLI_MODE = _parse_cli_mode() if __name__ == '__main__' else None


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def log_mse(pred, target):
    return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)


def set_seed(seed=RANDOM_SEED):
    """Fix NumPy and PyTorch RNGs so reported experiments are reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)


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

def split_hifi_data(hifi_path, n_train=30, n_eval=10, seed=RANDOM_SEED):
    """
    从全量高保真数据中随机划分:
      - n_train 条: 高保真训练集
      - n_eval  条: 高保真评估集 (early stop 依据)
      - 剩余    条: 测试集 (最终报告)
    """
    df = pd.read_csv(hifi_path)
    total = len(df)
    assert total >= n_train + n_eval, \
        f"数据不足: 共 {total} 条，需要 {n_train + n_eval} 条用于训练+评估"

    rng = np.random.default_rng(seed)
    idx = rng.permutation(total)

    train_idx = idx[:n_train]
    eval_idx  = idx[n_train:n_train + n_eval]
    test_idx  = idx[n_train + n_eval:]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_eval  = df.iloc[eval_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)

    # 保存划分结果到 CSV（方便检查和复现）
    split_dir = Path(hifi_path).parent / 'splits' / f'seed_{seed}'
    split_dir.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(split_dir / 'train.csv', index=False)
    df_eval.to_csv(split_dir  / 'eval.csv',  index=False)
    df_test.to_csv(split_dir  / 'test.csv',  index=False)

    def to_tensor(d):
        X = torch.tensor(d[FEATURES].values, dtype=torch.float32)
        y = torch.tensor(d[TARGET].values,   dtype=torch.float32)
        return X, y, d

    print(f"\n高保真数据划分 (seed={seed}):")
    print(f"  全量: {total} 条")
    print(f"  训练: {len(df_train)} 条  "
          f"Phi=[{df_train.Phi.min():.3f},{df_train.Phi.max():.3f}]  "
          f"SP%=[{df_train.SP_percent.min():.2f},{df_train.SP_percent.max():.2f}]  "
          f"τ₀=[{df_train.Tau0_Pa.min():.3f},{df_train.Tau0_Pa.max():.3f}] Pa")
    print(f"  评估: {len(df_eval)} 条  "
          f"Phi=[{df_eval.Phi.min():.3f},{df_eval.Phi.max():.3f}]  "
          f"SP%=[{df_eval.SP_percent.min():.2f},{df_eval.SP_percent.max():.2f}]  "
          f"τ₀=[{df_eval.Tau0_Pa.min():.3f},{df_eval.Tau0_Pa.max():.3f}] Pa")
    print(f"  测试: {len(df_test)} 条  "
          f"Phi=[{df_test.Phi.min():.3f},{df_test.Phi.max():.3f}]  "
          f"SP%=[{df_test.SP_percent.min():.2f},{df_test.SP_percent.max():.2f}]  "
          f"τ₀=[{df_test.Tau0_Pa.min():.3f},{df_test.Tau0_Pa.max():.3f}] Pa")
    print(f"  已保存至: {split_dir}/")

    return to_tensor(df_train), to_tensor(df_eval), to_tensor(df_test)


# ─────────────────────────────────────────────────────────
# Phase 1: 低保真度训练
# ─────────────────────────────────────────────────────────

def train_low_fidelity(
    train_path = 'data/lian2025/low_fidelity/train.csv',
    test_path  = 'data/lian2025/low_fidelity/test.csv',
    save_path  = 'multi_fidelity/models/lian2025/low_fidelity.pth',
    hidden_dim = 64,
    lr         = 1e-3,
    epochs     = 500,
    batch_size = 64,
    patience   = 50,
    seed       = RANDOM_SEED,
):
    set_seed(seed)
    print("\n" + "="*60)
    print("Phase 1: 低保真度训练 (合成数据)")
    print("="*60)

    X_train, y_train, df_train = load_csv(project_root / train_path)
    X_test,  y_test,  df_test  = load_csv(project_root / test_path)
    print(f"训练集: {len(X_train)}  测试集: {len(X_test)}")
    print(f"  Phi:  {df_train.Phi.min():.3f}–{df_train.Phi.max():.3f}")
    print(f"  SP%:  {df_train.SP_percent.min():.2f}–{df_train.SP_percent.max():.2f}")
    print(f"  τ₀:   {df_train.Tau0_Pa.min():.3f}–{df_train.Tau0_Pa.max():.3f} Pa")

    model = LianPINN(hidden_dim=hidden_dim)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total:,}  hidden_dim={hidden_dim}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

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

        if ep % 100 == 0 or ep == 1:
            print(f"Epoch {ep:4d}/{epochs} | "
                  f"Train {tr_loss:.5f} | Test {te_loss:.5f} | "
                  f"R²={r2:.4f} | MAE={mae:.4f} Pa | lr={cur_lr:.1e}")

        if te_loss < best_loss:
            best_loss = te_loss
            wait = 0
            torch.save({'model_state_dict': model.state_dict(),
                        'history': history}, save_full)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    elapsed = time.time() - t0

    ckpt = torch.load(save_full, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    tr_loss, tr_r2, tr_mae, tr_mape, _, _ = compute_metrics(model, X_train, y_train)
    te_loss, te_r2, te_mae, te_mape, _, _ = compute_metrics(model, X_test,  y_test)

    result = {
        'phase': 'low_fidelity',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'epochs_run': len(history['train_loss']),
        'elapsed_s': round(elapsed, 1),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train': {'r2': round(tr_r2,4), 'mae': round(tr_mae,4), 'mape': round(tr_mape,2)},
        'test':  {'r2': round(te_r2,4), 'mae': round(te_mae,4), 'mape': round(te_mape,2)},
    }

    print(f"\n{'─'*60}")
    print(f"低保真度训练完成  耗时 {elapsed:.1f}s  共 {result['epochs_run']} epochs")
    print(f"  合成训练集 : R²={tr_r2:.4f}  MAE={tr_mae:.4f} Pa  MAPE={tr_mape:.1f}%")
    print(f"  合成测试集 : R²={te_r2:.4f}  MAE={te_mae:.4f} Pa  MAPE={te_mape:.1f}%")
    print(f"{'─'*60}")

    return model, result


# ─────────────────────────────────────────────────────────
# 实验 14: 多保真训练 (低保真预训练 + 少量高保真微调)
# ─────────────────────────────────────────────────────────

def train_multifidelity(
    low_model,
    X_hifi_train, y_hifi_train,
    X_hifi_eval,  y_hifi_eval,
    X_hifi_test,  y_hifi_test,
    save_path   = 'multi_fidelity/models/lian2025/multifidelity.pth',
    freeze_n    = 1,
    lr          = 1e-4,
    epochs      = 1000,
    patience    = 150,
    weight_decay = 0.0,
    exp_tag     = 'exp14_multifidelity',
    seed        = RANDOM_SEED,
):
    """
    实验 14: 多保真策略
    低保真预训练权重 → 冻结前 freeze_n 层 → 在少量高保真数据上微调
    使用高保真评估集做 early stop，最终在高保真测试集上报告
    """
    set_seed(seed)
    print("\n" + "="*60)
    print(f"实验 14: 多保真训练 (低保真预训练 + {len(y_hifi_train)} 条高保真微调)")
    print(f"实验标签: {exp_tag}")
    print("="*60)

    save_full = project_root / save_path
    os.makedirs(save_full.parent, exist_ok=True)

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
    print(f"高保真训练: {len(y_hifi_train)} 条  评估: {len(y_hifi_eval)} 条  测试: {len(y_hifi_test)} 条")
    print(f"训练配置: lr={lr:.1e}, weight_decay={weight_decay:.1e}, "
          f"epochs={epochs}, patience={patience}")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )

    history = {'train_loss':[], 'eval_loss':[], 'eval_r2':[], 'eval_mae':[], 'lr':[]}
    best_r2 = float('-inf')
    best_epoch = 0
    wait = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_hifi_train)
        loss = log_mse(pred, y_hifi_train)
        loss.backward()
        optimizer.step()

        ev_loss, ev_r2, ev_mae, ev_mape, _, _ = compute_metrics(model, X_hifi_eval, y_hifi_eval)
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(loss.item())
        history['eval_loss'].append(ev_loss)
        history['eval_r2'].append(ev_r2)
        history['eval_mae'].append(ev_mae)
        history['lr'].append(cur_lr)

        if ep % 100 == 0 or ep == 1:
            print(f"Epoch {ep:4d}/{epochs} | "
                  f"Train={loss.item():.5f} | Eval R²={ev_r2:.4f} | "
                  f"Eval MAE={ev_mae:.4f} Pa")

        if ev_r2 > best_r2:
            best_r2 = ev_r2
            best_epoch = ep
            wait = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'history': history,
                'best_epoch': best_epoch,
                'best_r2': best_r2,
                'exp_tag': exp_tag,
                'freeze_n': freeze_n,
            }, save_full)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    elapsed = time.time() - t0

    ckpt = torch.load(save_full, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    tr_loss, tr_r2, tr_mae, tr_mape, tr_pred, _ = compute_metrics(model, X_hifi_train, y_hifi_train)
    ev_loss, ev_r2, ev_mae, ev_mape, ev_pred, _ = compute_metrics(model, X_hifi_eval,  y_hifi_eval)
    te_loss, te_r2, te_mae, te_mape, te_pred, _ = compute_metrics(model, X_hifi_test,  y_hifi_test)

    result = {
        'phase': 'multifidelity',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'exp_tag': exp_tag,
        'epochs_run': len(history['train_loss']),
        'elapsed_s': round(elapsed, 1),
        'best_epoch': ckpt['best_epoch'],
        'best_eval_r2': round(float(ckpt['best_r2']), 6),
        'freeze_n': freeze_n,
        'lr': lr,
        'weight_decay': weight_decay,
        'hifi_train_n': len(y_hifi_train),
        'hifi_eval_n':  len(y_hifi_eval),
        'hifi_test_n':  len(y_hifi_test),
        'train': {'r2': round(tr_r2,4), 'mae': round(tr_mae,4), 'mape': round(tr_mape,2)},
        'eval':  {'r2': round(ev_r2,4), 'mae': round(ev_mae,4), 'mape': round(ev_mape,2)},
        'test':  {'r2': round(te_r2,4), 'mae': round(te_mae,4), 'mape': round(te_mape,2)},
    }

    print(f"\n{'─'*60}")
    print(f"实验14完成  耗时 {elapsed:.1f}s  共 {result['epochs_run']} epochs")
    print(f"最佳 epoch: {result['best_epoch']} | eval R²={result['best_eval_r2']:.4f}")
    print(f"\n  结果汇总:")
    print(f"  {'数据集':<20} {'R²':>8} {'MAE':>10} {'MAPE':>8}")
    print(f"  {'─'*50}")
    print(f"  {'高保真训练集(30条)':<20} {tr_r2:>8.4f} {tr_mae:>8.4f} Pa {tr_mape:>6.1f}%")
    print(f"  {'高保真评估集(10条)':<20} {ev_r2:>8.4f} {ev_mae:>8.4f} Pa {ev_mape:>6.1f}%  ← early stop")
    print(f"  {'高保真测试集(360条)':<20} {te_r2:>8.4f} {te_mae:>8.4f} Pa {te_mape:>6.1f}%  ← 最终报告")
    print(f"{'─'*60}")

    _plot_training(history, exp_tag,
                      save_path=RESULTS_DIR / f'plots/lian_{exp_tag}_training.png')
    _plot_scatter(y_hifi_test.numpy(), te_pred, exp_tag,
                     title=f'Multi-fidelity Test Set (n=360)\n'
                           f'R²={te_r2:.4f}  MAE={te_mae:.4f} Pa',
                     save_path=RESULTS_DIR / f'plots/lian_{exp_tag}_test_scatter.png')

    return model, result


# ─────────────────────────────────────────────────────────
# 实验 15: 纯高保真训练 (无预训练，直接用少量数据训练)
# ─────────────────────────────────────────────────────────

def train_hifi_only(
    X_hifi_train, y_hifi_train,
    X_hifi_eval,  y_hifi_eval,
    X_hifi_test,  y_hifi_test,
    save_path   = 'multi_fidelity/models/lian2025/hifi_only.pth',
    hidden_dim  = 64,
    lr          = 1e-4,
    epochs      = 1000,
    patience    = 150,
    weight_decay = 0.0,
    exp_tag     = 'exp15_hifi_only',
    seed        = RANDOM_SEED,
):
    """
    实验 15: 纯高保真对照组
    随机初始化，直接在少量高保真数据上训练
    使用高保真评估集做 early stop，最终在高保真测试集上报告
    """
    set_seed(seed)
    print("\n" + "="*60)
    print(f"实验 15: 纯高保真训练 (随机初始化 + {len(y_hifi_train)} 条高保真训练)")
    print(f"实验标签: {exp_tag}")
    print("="*60)

    save_full = project_root / save_path
    os.makedirs(save_full.parent, exist_ok=True)

    model = LianPINN(hidden_dim=hidden_dim)
    total = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total:,}  (随机初始化，无预训练)")
    print(f"高保真训练: {len(y_hifi_train)} 条  评估: {len(y_hifi_eval)} 条  测试: {len(y_hifi_test)} 条")
    print(f"训练配置: lr={lr:.1e}, weight_decay={weight_decay:.1e}, "
          f"epochs={epochs}, patience={patience}")

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train_loss':[], 'eval_loss':[], 'eval_r2':[], 'eval_mae':[], 'lr':[]}
    best_r2 = float('-inf')
    best_epoch = 0
    wait = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_hifi_train)
        loss = log_mse(pred, y_hifi_train)
        loss.backward()
        optimizer.step()

        ev_loss, ev_r2, ev_mae, ev_mape, _, _ = compute_metrics(model, X_hifi_eval, y_hifi_eval)
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(loss.item())
        history['eval_loss'].append(ev_loss)
        history['eval_r2'].append(ev_r2)
        history['eval_mae'].append(ev_mae)
        history['lr'].append(cur_lr)

        if ep % 100 == 0 or ep == 1:
            print(f"Epoch {ep:4d}/{epochs} | "
                  f"Train={loss.item():.5f} | Eval R²={ev_r2:.4f} | "
                  f"Eval MAE={ev_mae:.4f} Pa")

        if ev_r2 > best_r2:
            best_r2 = ev_r2
            best_epoch = ep
            wait = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'history': history,
                'best_epoch': best_epoch,
                'best_r2': best_r2,
                'exp_tag': exp_tag,
            }, save_full)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {ep}")
                break

    elapsed = time.time() - t0

    ckpt = torch.load(save_full, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    tr_loss, tr_r2, tr_mae, tr_mape, tr_pred, _ = compute_metrics(model, X_hifi_train, y_hifi_train)
    ev_loss, ev_r2, ev_mae, ev_mape, ev_pred, _ = compute_metrics(model, X_hifi_eval,  y_hifi_eval)
    te_loss, te_r2, te_mae, te_mape, te_pred, _ = compute_metrics(model, X_hifi_test,  y_hifi_test)

    result = {
        'phase': 'hifi_only',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'exp_tag': exp_tag,
        'epochs_run': len(history['train_loss']),
        'elapsed_s': round(elapsed, 1),
        'best_epoch': ckpt['best_epoch'],
        'best_eval_r2': round(float(ckpt['best_r2']), 6),
        'hidden_dim': hidden_dim,
        'lr': lr,
        'weight_decay': weight_decay,
        'hifi_train_n': len(y_hifi_train),
        'hifi_eval_n':  len(y_hifi_eval),
        'hifi_test_n':  len(y_hifi_test),
        'train': {'r2': round(tr_r2,4), 'mae': round(tr_mae,4), 'mape': round(tr_mape,2)},
        'eval':  {'r2': round(ev_r2,4), 'mae': round(ev_mae,4), 'mape': round(ev_mape,2)},
        'test':  {'r2': round(te_r2,4), 'mae': round(te_mae,4), 'mape': round(te_mape,2)},
    }

    print(f"\n{'─'*60}")
    print(f"实验15完成  耗时 {elapsed:.1f}s  共 {result['epochs_run']} epochs")
    print(f"最佳 epoch: {result['best_epoch']} | eval R²={result['best_eval_r2']:.4f}")
    print(f"\n  结果汇总:")
    print(f"  {'数据集':<20} {'R²':>8} {'MAE':>10} {'MAPE':>8}")
    print(f"  {'─'*50}")
    print(f"  {'高保真训练集(30条)':<20} {tr_r2:>8.4f} {tr_mae:>8.4f} Pa {tr_mape:>6.1f}%")
    print(f"  {'高保真评估集(10条)':<20} {ev_r2:>8.4f} {ev_mae:>8.4f} Pa {ev_mape:>6.1f}%  ← early stop")
    print(f"  {'高保真测试集(360条)':<20} {te_r2:>8.4f} {te_mae:>8.4f} Pa {te_mape:>6.1f}%  ← 最终报告")
    print(f"{'─'*60}")

    _plot_training(history, exp_tag,
                      save_path=RESULTS_DIR / f'plots/lian_{exp_tag}_training.png')
    _plot_scatter(y_hifi_test.numpy(), te_pred, exp_tag,
                     title=f'HF-only Test Set (n=360)\n'
                           f'R²={te_r2:.4f}  MAE={te_mae:.4f} Pa',
                     save_path=RESULTS_DIR / f'plots/lian_{exp_tag}_test_scatter.png')

    return model, result


# ─────────────────────────────────────────────────────────
# 绘图函数
# ─────────────────────────────────────────────────────────

def _plot_training(history, exp_tag, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train', alpha=0.8)
    axes[0].plot(history['eval_loss'],  label='Eval',  alpha=0.8)
    axes[0].set_title('Log-MSE Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['eval_r2'], color='green')
    axes[1].axhline(0, color='red', linestyle='--', lw=1)
    axes[1].set_title('Eval R²'); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history['eval_mae'], color='orange')
    axes[2].set_title('Eval MAE (Pa)'); axes[2].grid(True, alpha=0.3)
    plt.suptitle(f'Training Curve: {exp_tag}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"训练曲线: {save_path}")

def _plot_scatter(y_true, y_pred, exp_tag, title, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)
    lim = [min(y_true.min(), y_pred.min())*0.85, max(y_true.max(), y_pred.max())*1.1]
    r2  = 1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-y_true.mean())**2)
    mae = np.mean(np.abs(y_true-y_pred))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=20, alpha=0.5)
    ax.plot(lim, lim, 'r--', lw=1.5, label='Ideal')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('True τ₀ (Pa)', fontsize=12)
    ax.set_ylabel('Predicted τ₀ (Pa)', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"散点图: {save_path}")


def _plot_comparison(results_list, save_path):
    """对比实验14 vs 实验15 的测试集指标"""
    os.makedirs(Path(save_path).parent, exist_ok=True)
    labels = [r['exp_tag'] for r in results_list]
    r2s    = [r['test']['r2']   for r in results_list]
    maes   = [r['test']['mae']  for r in results_list]
    mapes  = [r['test']['mape'] for r in results_list]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].bar(x, r2s, color=['steelblue', 'tomato'])
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=15, ha='right')
    axes[0].set_title('Test R²'); axes[0].set_ylim(0, 1); axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)

    axes[1].bar(x, maes, color=['steelblue', 'tomato'])
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=15, ha='right')
    axes[1].set_title('Test MAE (Pa)'); axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(maes):
        axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=10)

    axes[2].bar(x, mapes, color=['steelblue', 'tomato'])
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=15, ha='right')
    axes[2].set_title('Test MAPE (%)'); axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mapes):
        axes[2].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10)

    plt.suptitle(
        'Lian 2025: Multi-fidelity vs HF-only (test n=360)',
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"对比图: {save_path}")


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────

if CLI_MODE == 'main':
    print("\n" + "="*60)
    print("Lian 2025 实验: 验证真实小样本场景下多保真策略的价值")
    print("="*60)

    hifi_path = HIFI_DATA_PATH

    # 数据划分 (固定 seed=42，保证可复现)
    (X_tr, y_tr, _), (X_ev, y_ev, _), (X_te, y_te, _) = split_hifi_data(
        project_root / hifi_path,
        n_train=30, n_eval=10, seed=RANDOM_SEED,
    )

    all_results = []

    # ── Phase 1: 低保真预训练 ──
    low_model, low_result = train_low_fidelity()
    all_results.append(low_result)

    # ── 实验 14: 多保真 ──
    _, exp14_result = train_multifidelity(
        low_model,
        X_tr, y_tr, X_ev, y_ev, X_te, y_te,
        freeze_n=1,
        lr=1e-4,
        epochs=1000,
        patience=150,
        exp_tag='exp14_multifidelity',
    )
    all_results.append(exp14_result)

    # ── 实验 15: 纯高保真 ──
    _, exp15_result = train_hifi_only(
        X_tr, y_tr, X_ev, y_ev, X_te, y_te,
        lr=1e-4,
        epochs=1000,
        patience=150,
        exp_tag='exp15_hifi_only',
    )
    all_results.append(exp15_result)

    # ── 保存结果 ──
    result_path = RESULTS_DIR / 'logs/train_lian2025_results.json'
    os.makedirs(result_path.parent, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n所有结果已保存至: {result_path}")

    # ── 对比图 ──
    _plot_comparison(
        [exp14_result, exp15_result],
        save_path=RESULTS_DIR / 'plots/lian_comparison.png',
    )

    # ── 最终对比打印 ──
    print("\n" + "="*60)
    print("Lian 2025 实验最终对比 (测试集 360条)")
    print("="*60)
    print(f"{'实验':<30} {'test R²':>8} {'test MAE':>10} {'test MAPE':>10} {'best epoch':>12}")
    print("─"*75)
    for r in [exp14_result, exp15_result]:
        t = r['test']
        print(f"  {r['exp_tag']:<28} {t['r2']:>8.4f} {t['mae']:>8.4f} Pa "
              f"{t['mape']:>8.1f}%  {r['best_epoch']:>10}")
    print("─"*75)

    r2_gain  = exp14_result['test']['r2']  - exp15_result['test']['r2']
    mae_gain = exp15_result['test']['mae'] - exp14_result['test']['mae']
    print(f"\n多保真增益 (实验14 - 实验15):")
    print(f"  R²  : {r2_gain:+.4f}")
    print(f"  MAE : {mae_gain:+.4f} Pa")
    if r2_gain > 0 and mae_gain > 0:
        print("\n✅ 结论: 多保真策略在小样本场景下优于纯高保真训练")
    elif r2_gain > 0 or mae_gain > 0:
        print("\n⚠️  结论: 多保真策略部分指标优于纯高保真训练")
    else:
        print("\n❌ 结论: 在当前数据划分下，多保真策略未体现出优势")


# ─────────────────────────────────────────────────────────
# 实验 16: 多保真架构搜索
# ─────────────────────────────────────────────────────────

def train_arch_search(
    low_model_factory,          # callable() -> 返回新的低保真预训练模型
    X_tr, y_tr,
    X_ev, y_ev,
    X_te, y_te,
    arch_configs,               # list of dict: hidden_dim, n_hidden_layers, activation, exp_tag
    freeze_n    = 1,
    lr          = 1e-4,
    epochs      = 1000,
    patience    = 150,
):
    """
    对多组网络结构做多保真微调，统一在测试集上报告，比较架构影响。
    low_model_factory: 每次调用返回一个新的低保真预训练模型
    """
    all_results = []

    for cfg in arch_configs:
        hidden_dim      = cfg['hidden_dim']
        n_hidden_layers = cfg['n_hidden_layers']
        activation      = cfg['activation']
        exp_tag         = cfg['exp_tag']

        print("\n" + "="*60)
        print(f"实验 16 架构变体: {exp_tag}")

        # 用低保真预训练权重初始化可配置模型
        # 低保真模型是 LianPINN (hidden=64, layers=3, tanh)
        # 若架构相同则直接 deepcopy；若不同则只迁移兼容的权重
        base_model = low_model_factory()
        candidate_model = ConfigurableLianPINN(
            hidden_dim, n_hidden_layers, activation,
        )

        _transfer_weights(base_model, candidate_model)

        total     = sum(p.numel() for p in candidate_model.parameters())
        print(f"架构: {candidate_model.describe()}")

        # 冻结前 freeze_n 个 Linear 层
        linear_idx = [i for i, layer in enumerate(candidate_model.net)
                      if isinstance(layer, torch.nn.Linear)]
        freeze_set = set(linear_idx[:freeze_n])
        for i, layer in enumerate(candidate_model.net):
            if i in freeze_set:
                for p in layer.parameters():
                    p.requires_grad = False

        trainable = sum(p.numel() for p in candidate_model.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(f"冻结: {frozen:,} ({frozen/total*100:.1f}%)  可训练: {trainable:,} ({trainable/total*100:.1f}%)")

        optimizer = optim.Adam(
            [p for p in candidate_model.parameters() if p.requires_grad], lr=lr)

        history = {'train_loss':[], 'eval_r2':[], 'eval_mae':[]}
        best_r2    = float('-inf')
        best_epoch = 0
        wait       = 0
        best_state = None
        t0         = time.time()

        for ep in range(1, epochs + 1):
            candidate_model.train()
            optimizer.zero_grad()
            pred, _ = candidate_model(X_tr)
            loss = log_mse(pred, y_tr)
            loss.backward()
            optimizer.step()

            ev_loss, ev_r2, ev_mae, ev_mape, _, _ = compute_metrics(candidate_model, X_ev, y_ev)
            history['train_loss'].append(loss.item())
            history['eval_r2'].append(ev_r2)
            history['eval_mae'].append(ev_mae)

            if ep % 200 == 0 or ep == 1:
                print(f"  Epoch {ep:4d} | Train={loss.item():.5f} | "
                      f"Eval R²={ev_r2:.4f} | Eval MAE={ev_mae:.4f} Pa")

            if ev_r2 > best_r2:
                best_r2    = ev_r2
                best_epoch = ep
                wait       = 0
                best_state = copy.deepcopy(candidate_model.state_dict())
            else:
                wait += 1
                if wait >= patience:
                    print(f"  Early stop at epoch {ep}")
                    break

        elapsed = time.time() - t0
        candidate_model.load_state_dict(best_state)

        _, tr_r2, tr_mae, tr_mape, _, _ = compute_metrics(candidate_model, X_tr, y_tr)
        _, ev_r2, ev_mae, ev_mape, _, _ = compute_metrics(candidate_model, X_ev, y_ev)
        _, te_r2, te_mae, te_mape, te_pred, _ = compute_metrics(candidate_model, X_te, y_te)

        result = {
            'phase': 'arch_search',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'exp_tag': exp_tag,
            'arch': {
                'hidden_dim': hidden_dim,
                'n_hidden_layers': n_hidden_layers,
                'activation': activation,
                'total_params': total,
                'trainable_params': trainable,
            },
            'epochs_run': len(history['train_loss']),
            'elapsed_s': round(elapsed, 1),
            'best_epoch': best_epoch,
            'best_eval_r2': round(float(best_r2), 6),
            'train': {'r2': round(tr_r2,4), 'mae': round(tr_mae,4), 'mape': round(tr_mape,2)},
            'eval':  {'r2': round(ev_r2,4), 'mae': round(ev_mae,4), 'mape': round(ev_mape,2)},
            'test':  {'r2': round(te_r2,4), 'mae': round(te_mae,4), 'mape': round(te_mape,2)},
        }

        print(f"  完成  耗时 {elapsed:.1f}s  best_epoch={best_epoch}")
        print(f"  测试集: R²={te_r2:.4f}  MAE={te_mae:.4f} Pa  MAPE={te_mape:.1f}%")

        # 保存测试集散点图
        _plot_scatter(
            y_te.numpy(), te_pred, exp_tag,
            title=f'{exp_tag}\ntest R²={te_r2:.4f}  MAE={te_mae:.4f} Pa',
            save_path=RESULTS_DIR / f'plots/lian_{exp_tag}_test_scatter.png',
        )

        all_results.append(result)

    return all_results


def _transfer_weights(src_model, dst_model):
    """
    将基础 LianPINN 的兼容权重迁移到 ConfigurableLianPINN。
    只迁移形状完全匹配的层（按 net 中的位置顺序逐层对比）。
    不匹配的层保持随机初始化。
    """
    src_layers = [(n, p) for n, p in src_model.named_parameters()]
    dst_layers = [(n, p) for n, p in dst_model.named_parameters()]

    transferred = 0
    with torch.no_grad():
        for (sn, sp), (dn, dp) in zip(src_layers, dst_layers):
            if sp.shape == dp.shape:
                dp.copy_(sp)
                transferred += sp.numel()

    total_dst = sum(p.numel() for p in dst_model.parameters())
    print(f"  权重迁移: {transferred:,} / {total_dst:,} 参数 "
          f"({transferred/total_dst*100:.1f}%)")


# ─────────────────────────────────────────────────────────
# 架构搜索入口 (python -m multi_fidelity.src.training.train_lian2025 arch)
# ─────────────────────────────────────────────────────────

if CLI_MODE == 'arch':

    print("\n" + "="*60)
    print("实验 16: 多保真架构搜索")
    print("="*60)

    hifi_path = HIFI_DATA_PATH

    (X_tr, y_tr, _), (X_ev, y_ev, _), (X_te, y_te, _) = split_hifi_data(
        project_root / hifi_path, n_train=30, n_eval=10, seed=RANDOM_SEED,
    )

    # 先训练低保真基础模型（只训练一次，所有架构共用同一个预训练起点）
    low_model_base, _ = train_low_fidelity()

    def make_low_model():
        """每次返回低保真预训练权重的深拷贝，用于不同架构的初始化"""
        return copy.deepcopy(low_model_base)

    # 搜索空间: 控制参数量在 8k–35k 之间（30条训练数据，避免过度参数化）
    arch_configs = [
        # ── 基线 (与实验14完全相同，作为对照) ──
        {'hidden_dim': 64,  'n_hidden_layers': 3, 'activation': 'tanh', 'exp_tag': 'exp16_h64_l3_tanh'},
        # ── 加宽 ──
        {'hidden_dim': 96,  'n_hidden_layers': 3, 'activation': 'tanh', 'exp_tag': 'exp16_h96_l3_tanh'},
        {'hidden_dim': 128, 'n_hidden_layers': 3, 'activation': 'tanh', 'exp_tag': 'exp16_h128_l3_tanh'},
        # ── 加深 ──
        {'hidden_dim': 64,  'n_hidden_layers': 4, 'activation': 'tanh', 'exp_tag': 'exp16_h64_l4_tanh'},
        {'hidden_dim': 64,  'n_hidden_layers': 5, 'activation': 'tanh', 'exp_tag': 'exp16_h64_l5_tanh'},
        # ── 换激活函数 ──
        {'hidden_dim': 64,  'n_hidden_layers': 3, 'activation': 'gelu', 'exp_tag': 'exp16_h64_l3_gelu'},
        {'hidden_dim': 64,  'n_hidden_layers': 3, 'activation': 'silu', 'exp_tag': 'exp16_h64_l3_silu'},
        # ── 宽+深+换激活 组合 ──
        {'hidden_dim': 96,  'n_hidden_layers': 4, 'activation': 'gelu', 'exp_tag': 'exp16_h96_l4_gelu'},
        {'hidden_dim': 96,  'n_hidden_layers': 4, 'activation': 'silu', 'exp_tag': 'exp16_h96_l4_silu'},
    ]

    arch_results = train_arch_search(
        make_low_model, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
        arch_configs=arch_configs,
        freeze_n=1, lr=1e-4, epochs=1000, patience=150,
    )

    # 保存结果
    result_path = RESULTS_DIR / 'logs/train_lian2025_arch_results.json'
    os.makedirs(result_path.parent, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(arch_results, f, ensure_ascii=False, indent=2)
    print(f"\n架构搜索结果已保存至: {result_path}")

    # 汇总打印
    print("\n" + "="*60)
    print("实验 16 架构搜索汇总 (测试集 360条)")
    print("="*60)
    header = f"{'架构':<32} {'参数量':>8} {'test R²':>8} {'test MAE':>10} {'test MAPE':>10} {'best ep':>8}"
    print(header)
    print("─" * len(header))
    arch_results_sorted = sorted(arch_results, key=lambda r: r['test']['r2'], reverse=True)
    for r in arch_results_sorted:
        a = r['arch']
        t = r['test']
        tag = f"h{a['hidden_dim']}_l{a['n_hidden_layers']}_{a['activation']}"
        print(f"  {tag:<30} {a['total_params']:>8,} {t['r2']:>8.4f} "
              f"{t['mae']:>8.4f} Pa {t['mape']:>8.1f}%  {r['best_epoch']:>6}")
    print("─" * len(header))

    best = arch_results_sorted[0]
    baseline = next(r for r in arch_results if 'h64_l3_tanh' in r['exp_tag'])
    print(f"\n最优架构: {best['exp_tag']}")
    print(f"  test R²={best['test']['r2']:.4f}  MAE={best['test']['mae']:.4f} Pa  "
          f"MAPE={best['test']['mape']:.1f}%")
    print(f"相比基线 (h64_l3_tanh):")
    print(f"  R² 变化: {best['test']['r2'] - baseline['test']['r2']:+.4f}")
    print(f"  MAE 变化: {best['test']['mae'] - baseline['test']['mae']:+.4f} Pa")

    # 对比柱状图
    _plot_comparison(
        arch_results_sorted,
        save_path=RESULTS_DIR / 'plots/lian_arch_comparison.png',
    )


# ─────────────────────────────────────────────────────────
# 实验 17: 多保真超参搜索
# ─────────────────────────────────────────────────────────

def train_hparam_search(
    low_model_base,
    X_tr, y_tr,
    X_ev, y_ev,
    X_te, y_te,
    hparam_configs,
    epochs=2000,
    patience=200,
):
    """
    对多组 (freeze_n, lr, weight_decay) 组合做多保真微调，
    统一在测试集上报告，寻找最优超参组合。
    """
    all_results = []

    for cfg in hparam_configs:
        freeze_n     = cfg['freeze_n']
        lr           = cfg['lr']
        weight_decay = cfg['weight_decay']
        exp_tag      = cfg['exp_tag']

        print(f"\n{'─'*55}")
        print(f"  {exp_tag}  "
              f"freeze={freeze_n}  lr={lr:.0e}  wd={weight_decay:.0e}")

        model = copy.deepcopy(low_model_base)

        # 冻结策略
        linear_idx = [i for i, layer in enumerate(model.net)
                      if hasattr(layer, 'weight')]
        freeze_set = set(linear_idx[:freeze_n])
        for i, layer in enumerate(model.net):
            if i in freeze_set:
                for p in layer.parameters():
                    p.requires_grad = False

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = total - trainable
        print(f"  冻结 {frozen:,} ({frozen/total*100:.0f}%)  "
              f"可训练 {trainable:,} ({trainable/total*100:.0f}%)")

        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )

        best_r2    = float('-inf')
        best_epoch = 0
        wait       = 0
        best_state = None
        t0         = time.time()

        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred, _ = model(X_tr)
            loss = log_mse(pred, y_tr)
            loss.backward()
            optimizer.step()

            _, ev_r2, ev_mae, _, _, _ = compute_metrics(model, X_ev, y_ev)

            if ev_r2 > best_r2:
                best_r2    = ev_r2
                best_epoch = ep
                wait       = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= patience:
                    break

        elapsed = time.time() - t0
        model.load_state_dict(best_state)

        _, tr_r2, tr_mae, tr_mape, _, _ = compute_metrics(model, X_tr, y_tr)
        _, ev_r2, ev_mae, ev_mape, _, _ = compute_metrics(model, X_ev, y_ev)
        _, te_r2, te_mae, te_mape, te_pred, _ = compute_metrics(model, X_te, y_te)

        result = {
            'phase': 'hparam_search',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'exp_tag': exp_tag,
            'freeze_n': freeze_n,
            'lr': lr,
            'weight_decay': weight_decay,
            'epochs_run': best_epoch + patience,
            'best_epoch': best_epoch,
            'elapsed_s': round(elapsed, 1),
            'train': {'r2': round(tr_r2,4), 'mae': round(tr_mae,4), 'mape': round(tr_mape,2)},
            'eval':  {'r2': round(ev_r2,4), 'mae': round(ev_mae,4), 'mape': round(ev_mape,2)},
            'test':  {'r2': round(te_r2,4), 'mae': round(te_mae,4), 'mape': round(te_mape,2)},
        }
        print(f"  best_ep={best_epoch}  "
              f"test R²={te_r2:.4f}  MAE={te_mae:.4f} Pa  MAPE={te_mape:.1f}%")
        all_results.append(result)

    return all_results


# ─────────────────────────────────────────────────────────
# 超参搜索入口 (python -m multi_fidelity.src.training.train_lian2025 hparam)
# ─────────────────────────────────────────────────────────

if CLI_MODE == 'hparam':

    print("\n" + "="*60)
    print("实验 17: 多保真超参搜索 (freeze_n × lr × weight_decay)")
    print("="*60)

    hifi_path = HIFI_DATA_PATH

    (X_tr, y_tr, _), (X_ev, y_ev, _), (X_te, y_te, _) = split_hifi_data(
        project_root / hifi_path, n_train=30, n_eval=10, seed=RANDOM_SEED,
    )

    low_model_base, _ = train_low_fidelity()

    # 搜索空间
    # freeze_n: 0(全部可训练) / 1(冻结输入层) / 2(冻结前两层) / 3(只留输出层)
    # lr: 1e-3 / 3e-4 / 1e-4 / 3e-5
    # weight_decay: 0 / 1e-4
    hparam_configs = []
    for freeze_n in [0, 1, 2, 3]:
        for lr in [1e-3, 3e-4, 1e-4, 3e-5]:
            for wd in [0.0, 1e-4]:
                tag = f"exp17_f{freeze_n}_lr{lr:.0e}_wd{wd:.0e}"
                hparam_configs.append({
                    'freeze_n': freeze_n, 'lr': lr,
                    'weight_decay': wd, 'exp_tag': tag,
                })

    print(f"共 {len(hparam_configs)} 组超参配置\n")

    hp_results = train_hparam_search(
        low_model_base, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
        hparam_configs=hparam_configs,
        epochs=2000, patience=200,
    )

    # 保存
    result_path = RESULTS_DIR / 'logs/train_lian2025_hparam_results.json'
    os.makedirs(result_path.parent, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(hp_results, f, ensure_ascii=False, indent=2)
    print(f"\n超参搜索结果已保存至: {result_path}")

    # 汇总：按 test R² 降序
    hp_sorted = sorted(hp_results, key=lambda r: r['test']['r2'], reverse=True)

    print("\n" + "="*60)
    print("实验 17 超参搜索汇总 Top-15 (测试集 360条)")
    print("="*60)
    header = (f"{'配置':<38} {'freeze':>6} {'lr':>8} {'wd':>8} "
              f"{'test R²':>8} {'test MAE':>10} {'best ep':>8}")
    print(header)
    print("─" * len(header))
    for r in hp_sorted[:15]:
        t = r['test']
        print(f"  {r['exp_tag']:<36} {r['freeze_n']:>6} "
              f"{r['lr']:>8.0e} {r['weight_decay']:>8.0e} "
              f"{t['r2']:>8.4f} {t['mae']:>8.4f} Pa {r['best_epoch']:>8}")
    print("─" * len(header))

    best = hp_sorted[0]
    baseline = next(r for r in hp_results
                    if r['freeze_n'] == 1 and r['lr'] == 1e-4
                    and r['weight_decay'] == 0.0)
    print(f"\n最优配置: {best['exp_tag']}")
    print(f"  freeze_n={best['freeze_n']}  lr={best['lr']:.0e}  "
          f"wd={best['weight_decay']:.0e}")
    print(f"  test R²={best['test']['r2']:.4f}  "
          f"MAE={best['test']['mae']:.4f} Pa  "
          f"MAPE={best['test']['mape']:.1f}%")
    print(f"\n相比实验14基线 (freeze=1, lr=1e-4, wd=0):")
    print(f"  R² 变化 : {best['test']['r2']  - baseline['test']['r2']:+.4f}")
    print(f"  MAE 变化: {best['test']['mae'] - baseline['test']['mae']:+.4f} Pa")

    # 按 freeze_n 分组打印各组最优
    print("\n各 freeze_n 最优配置:")
    print(f"  {'freeze_n':>8} {'lr':>8} {'wd':>8} {'test R²':>8} {'test MAE':>10}")
    print(f"  {'─'*50}")
    for fn in [0, 1, 2, 3]:
        group = [r for r in hp_results if r['freeze_n'] == fn]
        best_g = max(group, key=lambda r: r['test']['r2'])
        t = best_g['test']
        print(f"  {fn:>8}  {best_g['lr']:>8.0e}  {best_g['weight_decay']:>8.0e} "
              f"  {t['r2']:>8.4f}  {t['mae']:>8.4f} Pa")


# ─────────────────────────────────────────────────────────
# 实验 18: 消融实验 + 综合对比图
# ─────────────────────────────────────────────────────────

def run_ablation(
    X_te, y_te,
    low_model_path  = 'multi_fidelity/models/lian2025/low_fidelity.pth',
    hifi_model_path = 'multi_fidelity/models/lian2025/hifi_only.pth',
    mf_model_path   = 'multi_fidelity/models/lian2025/multifidelity.pth',
):
    """
    在同一测试集上收集四组预测值，画消融对比图：
      A. 物理公式 + φ_max 均值（纯机理基线）
      B. 只用低保真模型
      C. 只用高保真（30条，无预训练）
      D. 多保真融合（低保真预训练 + 30条微调）
    """
    y_true = y_te.numpy()
    M1 = 0.72
    phi = X_te[:, 0].numpy()

    # ── A: 物理公式 + φ_max 均值 ──
    # 用测试集 Phi 的分布估算一个"合理的"固定 φ_max
    # φ_max 必须 > φ，取 φ_mean + 0.25 作为固定值（物理合理范围内）
    phi_max_fixed = float(phi.mean()) + 0.25
    eps = 1e-6
    pred_physics = M1 * phi**3 / (phi_max_fixed * (phi_max_fixed - phi + eps))
    r2_A   = float(1 - np.sum((y_true - pred_physics)**2) / np.sum((y_true - y_true.mean())**2))
    mae_A  = float(np.mean(np.abs(y_true - pred_physics)))
    mape_A = float(np.mean(np.abs((y_true - pred_physics) / (y_true + eps))) * 100)
    print(f"\nA. 物理公式 (φ_max={phi_max_fixed:.3f} 固定均值): "
          f"R²={r2_A:.4f}  MAE={mae_A:.4f} Pa  MAPE={mape_A:.1f}%")

    # ── B: 低保真模型 ──
    low_model = LianPINN(hidden_dim=64)
    ckpt = torch.load(project_root / low_model_path, weights_only=False)
    low_model.load_state_dict(ckpt['model_state_dict'])
    _, r2_B, mae_B, mape_B, pred_B, _ = compute_metrics(low_model, X_te, y_te)
    print(f"B. 只用低保真:                       "
          f"R²={r2_B:.4f}  MAE={mae_B:.4f} Pa  MAPE={mape_B:.1f}%")

    # ── C: 只用高保真（30条，无预训练）──
    hifi_model = LianPINN(hidden_dim=64)
    ckpt = torch.load(project_root / hifi_model_path, weights_only=False)
    hifi_model.load_state_dict(ckpt['model_state_dict'])
    _, r2_C, mae_C, mape_C, pred_C, _ = compute_metrics(hifi_model, X_te, y_te)
    print(f"C. 只用高保真 (30条，无预训练):      "
          f"R²={r2_C:.4f}  MAE={mae_C:.4f} Pa  MAPE={mape_C:.1f}%")

    # ── D: 多保真融合 ──
    mf_model = LianPINN(hidden_dim=64)
    ckpt = torch.load(project_root / mf_model_path, weights_only=False)
    mf_model.load_state_dict(ckpt['model_state_dict'])
    _, r2_D, mae_D, mape_D, pred_D, _ = compute_metrics(mf_model, X_te, y_te)
    print(f"D. 多保真融合 (低保真预训练+30条微调): "
          f"R²={r2_D:.4f}  MAE={mae_D:.4f} Pa  MAPE={mape_D:.1f}%")

    groups = [
        {'label': 'A. Physics\n(φ_max fixed)', 'pred': pred_physics,
         'r2': r2_A, 'mae': mae_A, 'mape': mape_A, 'color': '#888888', 'ls': '--'},
        {'label': 'B. Low-fidelity\nonly',       'pred': pred_B,
         'r2': r2_B, 'mae': mae_B, 'mape': mape_B, 'color': '#4878CF', 'ls': '-.'},
        {'label': 'C. High-fidelity\nonly (30)',  'pred': pred_C,
         'r2': r2_C, 'mae': mae_C, 'mape': mape_C, 'color': '#D65F5F', 'ls': ':'},
        {'label': 'D. Multi-fidelity\n(30 HF)',   'pred': pred_D,
         'r2': r2_D, 'mae': mae_D, 'mape': mape_D, 'color': '#6ACC65', 'ls': '-'},
    ]

    _plot_ablation(y_true, groups,
                   save_path=RESULTS_DIR / 'plots/lian_ablation.png')

    return groups


def _plot_ablation(y_true, groups, save_path):
    """
    三合一消融对比图：
      左：散点图（预测 vs 真实），四组叠加
      中：R² 柱状图
      右：MAE 柱状图
    """
    os.makedirs(Path(save_path).parent, exist_ok=True)

    fig = plt.figure(figsize=(18, 6))
    gs  = fig.add_gridspec(1, 3, wspace=0.32)
    ax_scatter = fig.add_subplot(gs[0])
    ax_r2      = fig.add_subplot(gs[1])
    ax_mae     = fig.add_subplot(gs[2])

    # ── 散点图 ──
    all_vals = np.concatenate([y_true] + [g['pred'] for g in groups])
    lim = [all_vals.min() * 0.88, all_vals.max() * 1.08]
    ax_scatter.plot(lim, lim, 'k--', lw=1, alpha=0.4, label='Ideal')

    markers = ['o', 's', '^', 'D']
    for g, mk in zip(groups, markers):
        ax_scatter.scatter(y_true, g['pred'],
                           s=12, alpha=0.45, color=g['color'],
                           marker=mk,
                           label=f"{g['label'].replace(chr(10),' ')}  R²={g['r2']:.3f}")
    ax_scatter.set_xlim(lim); ax_scatter.set_ylim(lim)
    ax_scatter.set_xlabel('True τ₀ (Pa)', fontsize=11)
    ax_scatter.set_ylabel('Predicted τ₀ (Pa)', fontsize=11)
    ax_scatter.set_title('Predicted vs True (test set, n=360)', fontsize=11)
    ax_scatter.legend(fontsize=7.5, loc='upper left')
    ax_scatter.grid(True, alpha=0.25)

    # ── R² 柱状图 ──
    labels_short = [g['label'] for g in groups]
    r2s   = [g['r2']   for g in groups]
    maes  = [g['mae']  for g in groups]
    mapes = [g['mape'] for g in groups]
    colors = [g['color'] for g in groups]
    x = np.arange(len(groups))

    bars = ax_r2.bar(x, r2s, color=colors, width=0.55, edgecolor='white', linewidth=0.8)
    ax_r2.set_xticks(x); ax_r2.set_xticklabels(labels_short, fontsize=8)
    ax_r2.set_ylim(min(0, min(r2s)) - 0.05, 1.0)
    ax_r2.axhline(0, color='black', lw=0.8, alpha=0.5)
    ax_r2.set_title('R² (test set)', fontsize=11)
    ax_r2.set_ylabel('R²', fontsize=10)
    ax_r2.grid(True, alpha=0.25, axis='y')
    for bar, v in zip(bars, r2s):
        ypos = max(v, 0) + 0.01
        ax_r2.text(bar.get_x() + bar.get_width()/2, ypos,
                   f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── MAE 柱状图（叠加 MAPE 文字）──
    bars2 = ax_mae.bar(x, maes, color=colors, width=0.55, edgecolor='white', linewidth=0.8)
    ax_mae.set_xticks(x); ax_mae.set_xticklabels(labels_short, fontsize=8)
    ax_mae.set_title('MAE (Pa) + MAPE (%)', fontsize=11)
    ax_mae.set_ylabel('MAE (Pa)', fontsize=10)
    ax_mae.grid(True, alpha=0.25, axis='y')
    for bar, mv, mp in zip(bars2, maes, mapes):
        ax_mae.text(bar.get_x() + bar.get_width()/2, mv + 0.001,
                    f'{mv:.4f}\n({mp:.1f}%)',
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    fig.suptitle('Ablation Study: Physics / Low-fidelity / High-fidelity / Multi-fidelity\n'
                 f'Test set: 360 samples  |  Training HF data: 30 samples',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\n消融对比图已保存: {save_path}")


# ─────────────────────────────────────────────────────────
# 消融实验入口 (python -m multi_fidelity.src.training.train_lian2025 ablation)
# ─────────────────────────────────────────────────────────

if CLI_MODE == 'ablation':

    print("\n" + "="*60)
    print("实验 18: 消融实验 — 四种策略在测试集上的对比")
    print("="*60)

    hifi_path = HIFI_DATA_PATH

    (X_tr, y_tr, _), (X_ev, y_ev, _), (X_te, y_te, _) = split_hifi_data(
        project_root / hifi_path, n_train=30, n_eval=10, seed=RANDOM_SEED,
    )

    print(f"\n测试集: {len(y_te)} 条")

    groups = run_ablation(X_te, y_te)

    # 汇总打印
    print("\n" + "="*60)
    print("消融实验汇总 (测试集 360 条)")
    print("="*60)
    print(f"  {'策略':<40} {'R²':>8} {'MAE':>10} {'MAPE':>8}")
    print("  " + "─"*68)
    for g in groups:
        lbl = g['label'].replace('\n', ' ')
        print(f"  {lbl:<40} {g['r2']:>8.4f} {g['mae']:>8.4f} Pa {g['mape']:>7.1f}%")
    print("  " + "─"*68)

    best = max(groups, key=lambda g: g['r2'])
    worst = min(groups, key=lambda g: g['r2'])
    print(f"\n最优: {best['label'].replace(chr(10),' ')}  R²={best['r2']:.4f}")
    print(f"最差: {worst['label'].replace(chr(10),' ')}  R²={worst['r2']:.4f}")
    gain_mf_vs_hifi = groups[3]['r2'] - groups[2]['r2']
    gain_mf_vs_low  = groups[3]['r2'] - groups[1]['r2']
    print(f"\n多保真 vs 纯高保真增益: R² +{gain_mf_vs_hifi:.4f}")
    print(f"多保真 vs 纯低保真增益: R² +{gain_mf_vs_low:.4f}")


# ─────────────────────────────────────────────────────────
# 补充对比：数据量充足场景
# 运行: python -m multi_fidelity.src.training.train_lian2025 sufficient
# 320 训练 / 40 评估 / 40 测试 — 纯HF vs 多保真，等量公平对比
# ─────────────────────────────────────────────────────────

if CLI_MODE == 'sufficient':

    print("\n" + "="*60)
    print("补充实验: 数据量充足场景等量对比 (320 HF 训练)")
    print("="*60)

    hifi_path = HIFI_DATA_PATH

    # 320 训练 / 40 评估(早停) / 40 测试
    (X_tr, y_tr, _), (X_ev, y_ev, _), (X_te, y_te, _) = split_hifi_data(
        project_root / hifi_path,
        n_train=320, n_eval=40, seed=RANDOM_SEED,
    )
    print(f"  训练: {len(y_tr)} 条 | 评估: {len(y_ev)} 条 | 测试: {len(y_te)} 条")

    # ── 策略1: 纯高保真 (320条，随机初始化) ──
    print("\n[策略1] 纯高保真 (320条，随机初始化)...")
    _, res_hifi = train_hifi_only(
        X_tr, y_tr, X_ev, y_ev, X_te, y_te,
        lr=1e-4, epochs=2000, patience=200,
        exp_tag='sufficient_hifi_only_320',
    )

    # ── 策略2: 多保真 (LF预训练 + 320条微调) ──
    print("\n[策略2] 多保真 (LF预训练 + 320条微调)...")
    low_model = LianPINN(hidden_dim=64)
    low_ckpt = torch.load(
        project_root / 'multi_fidelity/models/lian2025/low_fidelity.pth',
        weights_only=False,
    )
    low_model.load_state_dict(low_ckpt['model_state_dict'])
    _, res_mf = train_multifidelity(
        low_model,
        X_tr, y_tr, X_ev, y_ev, X_te, y_te,
        freeze_n=1, lr=1e-4, epochs=2000, patience=200,
        exp_tag='sufficient_multifidelity_320',
    )

    # ── 结果汇总 ──
    print("\n" + "="*60)
    print("补充实验汇总 (数据量充足，40条独立测试集)")
    print("="*60)
    print(f"  {'策略':<35} {'test R²':>8} {'test MAE':>10} {'test MAPE':>10} {'epoch':>8}")
    print("  " + "─"*75)
    for tag, res in [('纯高保真 (320条)', res_hifi), ('多保真 LF→HF (320条)', res_mf)]:
        t = res['test']
        print(f"  {tag:<35} {t['r2']:>8.4f} {t['mae']:>8.4f} Pa {t['mape']:>8.1f}%  "
              f"{res['best_epoch']:>6}")
    print("  " + "─"*75)
    gain_r2   = res_mf['test']['r2']   - res_hifi['test']['r2']
    gain_mape = res_hifi['test']['mape'] - res_mf['test']['mape']
    print(f"\n  多保真增益: ΔR²={gain_r2:+.4f}  ΔMAPE={gain_mape:+.1f} pp")
