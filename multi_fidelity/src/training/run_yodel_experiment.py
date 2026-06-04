"""
YODEL 体系跨模型泛化验证实验

用途: PI-MFNN 框架在第二本构方程（YODEL）上的验证，
      论文 §4.x 跨体系泛化实验的配套训练脚本

物理方程 (YODEL PCL):
    τ₀ = m_y * φ² / [φ_max(c_d) * (φ_max(c_d) − φ)²]
    m_y = 0.12 Pa, 输入 = [phi, c_d]

实验设计（镜像 Lian 2025 四策略消融 + 多方法对比）:
  策略 A: 纯物理公式      φ_max 取固定均值常数
  策略 B: 纯低保真        仅用 LF 合成数据训练
  策略 C: 纯高保真 (HF)   30 条 HF 随机初始化训练
  策略 D: 多保真融合 (本文) LF 预训练 → 30 条 HF 冻结层微调

运行方式:
  cd /path/to/Project
  python -m multi_fidelity.src.training.run_yodel_experiment \\
      --lf-data  data/yodel_lf    \\
      --hf-data  data/yodel_hf    \\
      --out-dir  multi_fidelity/results/yodel_exp
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multi_fidelity.src.model.pinn_yodel_v1 import YodelPINN_v1   # noqa: E402


# ── 常量 ──────────────────────────────────────────────────────────────────

FEATURES = ['phi', 'c_d']
TARGET   = 'tau0_Pa'
SEED     = 42

LF_TRAIN_N  = 1600   # 低保真训练集大小
HF_TRAIN_N  = 30     # 稀缺场景高保真训练数
HF_EVAL_N   = 10


# ── 工具函数 ──────────────────────────────────────────────────────────────

def set_seed(s: int = SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def log_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)


def compute_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        pred, phi_max = model(X)
        r2   = (1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)).item()
        mae  = torch.mean(torch.abs(pred - y)).item()
        mape = (torch.mean(torch.abs((pred - y) / (y + 1e-6))) * 100).item()
    return r2, mae, mape, pred.cpu().numpy()


def load_tensors(path: Path):
    df = pd.read_csv(path)
    X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET].values,   dtype=torch.float32)
    return X, y, df


# ── 训练循环 ──────────────────────────────────────────────────────────────

def train_lf(model, X_train, y_train, X_test, y_test,
             lr=1e-3, max_epochs=500, patience=50,
             batch_size=64):
    """低保真预训练 (全量 LF 数据, ReduceLROnPlateau)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    best_r2, best_state, wait = -1.0, None, 0
    history = []

    for ep in range(1, max_epochs + 1):
        model.train()
        idx = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            b = idx[i:i+batch_size]
            optimizer.zero_grad()
            pred, _ = model(X_train[b])
            loss = log_mse(pred, y_train[b])
            loss.backward()
            optimizer.step()

        r2, mae, mape, _ = compute_metrics(model, X_test, y_test)
        scheduler.step(1 - r2)   # ReduceLROnPlateau on R²
        history.append({'ep': ep, 'r2': r2, 'mae': mae})

        if r2 > best_r2:
            best_r2 = r2
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    r2, mae, mape, _ = compute_metrics(model, X_test, y_test)
    print(f"  LF 预训练完成 | epoch={ep} | 合成测试 R²={r2:.4f} MAPE={mape:.1f}%")
    return best_r2, history


def train_hf(model, X_train, y_train, X_eval, y_eval,
             lr=1e-4, max_epochs=1000, patience=150,
             freeze_n=1):
    """高保真微调 (冻结前 freeze_n 个 Linear 层, 按 eval R² 早停)."""
    # 冻结浅层
    linear_layers = [m for m in model.net if hasattr(m, 'weight')]
    for layer in linear_layers[:freeze_n]:
        for p in layer.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=1e-5)

    best_r2, best_state, wait = -1.0, None, 0

    for ep in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_train)
        loss = log_mse(pred, y_train)
        loss.backward()
        optimizer.step()

        r2_eval, _, _, _ = compute_metrics(model, X_eval, y_eval)
        if r2_eval > best_r2:
            best_r2 = r2_eval
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    # 解冻（复原 requires_grad 以便后续操作不受影响）
    for p in model.parameters():
        p.requires_grad = True

    return best_r2, ep


# ── 实验策略 ──────────────────────────────────────────────────────────────

def run_strategy_A(X_test, y_test) -> dict:
    """策略 A: 纯物理公式 (φ_max 固定均值)."""
    # φ_max 固定均值: 从 HF 训练集无法确定，用 LF 数据 φ_max 均值
    phi_max_fixed = 0.620   # YODEL 体系 φ_max 典型均值 (由数据分析确定后可更新)
    m_y = YodelPINN_v1.M_Y

    phi = X_test[:, 0]
    eps = 1e-6
    diff = torch.relu(torch.tensor(phi_max_fixed) - phi) + eps
    pred = m_y * phi**2 / (torch.tensor(phi_max_fixed) * diff**2)

    y = y_test
    r2   = (1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)).item()
    mae  = torch.mean(torch.abs(pred - y)).item()
    mape = (torch.mean(torch.abs((pred - y) / (y + 1e-6))) * 100).item()
    print(f"  策略 A (纯物理) | phi_max={phi_max_fixed} | "
          f"R²={r2:.4f} MAE={mae:.4f} MAPE={mape:.1f}%")
    return {'label': 'A_physics', 'R2': r2, 'MAE': mae, 'MAPE': mape,
            'phi_max_fixed': phi_max_fixed}


def run_strategy_B(X_lf_train, y_lf_train, X_lf_test, y_lf_test,
                   X_test, y_test) -> dict:
    """策略 B: 纯低保真 (仅用 LF 合成数据)."""
    set_seed()
    model = YodelPINN_v1()
    best_lf_r2, _ = train_lf(model, X_lf_train, y_lf_train,
                              X_lf_test, y_lf_test)

    r2, mae, mape, _ = compute_metrics(model, X_test, y_test)
    print(f"  策略 B (纯低保真) | test R²={r2:.4f} MAPE={mape:.1f}%")
    return {'label': 'B_lf_only', 'R2': r2, 'MAE': mae, 'MAPE': mape,
            'lf_pretrain_R2': best_lf_r2, 'model': model}


def run_strategy_C(X_hf_train, y_hf_train, X_hf_eval, y_hf_eval,
                   X_test, y_test) -> dict:
    """策略 C: 纯高保真 (随机初始化, 30 条 HF)."""
    set_seed()
    model = YodelPINN_v1()
    _, best_epoch = train_hf(model, X_hf_train, y_hf_train,
                             X_hf_eval, y_hf_eval,
                             lr=1e-4, freeze_n=0)   # freeze_n=0 = 不冻结

    r2, mae, mape, _ = compute_metrics(model, X_test, y_test)
    print(f"  策略 C (纯高保真) | best_epoch={best_epoch} | "
          f"test R²={r2:.4f} MAPE={mape:.1f}%")
    return {'label': 'C_hf_only', 'R2': r2, 'MAE': mae, 'MAPE': mape,
            'best_epoch': best_epoch, 'model': model}


def run_strategy_D(X_lf_train, y_lf_train, X_lf_test, y_lf_test,
                   X_hf_train, y_hf_train, X_hf_eval, y_hf_eval,
                   X_test, y_test,
                   freeze_n: int = 1) -> dict:
    """策略 D: 多保真融合 (LF 预训练 → HF 冻结层微调)."""
    set_seed()
    model = YodelPINN_v1()

    # 阶段 1: LF 预训练
    best_lf_r2, _ = train_lf(model, X_lf_train, y_lf_train,
                              X_lf_test, y_lf_test)

    # 阶段 2: HF 冻结层微调
    best_eval_r2, best_epoch = train_hf(model,
                                        X_hf_train, y_hf_train,
                                        X_hf_eval, y_hf_eval,
                                        lr=1e-4, freeze_n=freeze_n)

    r2, mae, mape, _ = compute_metrics(model, X_test, y_test)
    print(f"  策略 D (多保真)   | freeze={freeze_n} best_epoch={best_epoch} | "
          f"test R²={r2:.4f} MAPE={mape:.1f}%")
    return {'label': 'D_mfnn', 'R2': r2, 'MAE': mae, 'MAPE': mape,
            'best_epoch': best_epoch, 'lf_pretrain_R2': best_lf_r2,
            'freeze_n': freeze_n, 'model': model}


# ── 散点图 ────────────────────────────────────────────────────────────────

def scatter_four(results, X_test, y_test, out_path):
    """四策略预测值 vs 真实值散点图."""
    y_np = y_test.cpu().numpy()
    lim  = [y_np.min() * 0.85, y_np.max() * 1.1]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    labels = ['(a) 纯物理公式', '(b) 纯低保真', '(c) 纯高保真', '(d) 多保真融合 (本文)']

    # 为策略 A 单独计算预测值
    phi_max_fixed = results[0].get('phi_max_fixed', 0.620)
    m_y = YodelPINN_v1.M_Y
    phi = X_test[:, 0]
    eps = 1e-6
    diff_a = torch.relu(torch.tensor(phi_max_fixed) - phi) + eps
    pred_A = (m_y * phi**2 / (torch.tensor(phi_max_fixed) * diff_a**2)).numpy()

    preds = [pred_A]
    for res in results[1:]:
        if 'model' in res:
            _, _, _, pred_np = compute_metrics(res['model'], X_test, y_test)
            preds.append(pred_np)
        else:
            preds.append(np.full_like(y_np, np.nan))

    for ax, pred, label, res in zip(axes, preds, labels, results):
        r2   = res['R2']
        mape = res['MAPE']
        ax.scatter(y_np, pred, alpha=0.6, s=20, color='steelblue')
        ax.plot(lim, lim, 'r--', lw=1.2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('True τ₀ (Pa)', fontsize=8)
        ax.set_ylabel('Pred τ₀ (Pa)', fontsize=8)
        ax.set_title(f"{label}\n$R^2$={r2:.3f}  MAPE={mape:.1f}%", fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.suptitle('YODEL 陶瓷悬浮液体系：四策略性能对比', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"散点图保存至 {out_path}")


# ── 主程序 ────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载数据 ──
    lf_dir = Path(args.lf_data)
    hf_dir = Path(args.hf_data)

    X_lf_train, y_lf_train, _ = load_tensors(lf_dir / 'train.csv')
    X_lf_test,  y_lf_test,  _ = load_tensors(lf_dir / 'test.csv')

    X_hf_train, y_hf_train, _ = load_tensors(hf_dir / 'train_scarce.csv')
    X_hf_eval,  y_hf_eval,  _ = load_tensors(hf_dir / 'eval.csv')
    X_test,     y_test,     _  = load_tensors(hf_dir / 'test.csv')

    print(f"\n数据摘要:")
    print(f"  LF 训练: {len(X_lf_train)}  LF 测试: {len(X_lf_test)}")
    print(f"  HF 训练: {len(X_hf_train)}  HF 评估: {len(X_hf_eval)}  测试: {len(X_test)}")
    print(f"  tau0 范围: {y_test.min():.3f} – {y_test.max():.3f} Pa")
    print()

    # ── 2. 运行四策略 ──
    print("=" * 60)
    print("四策略消融实验 (YODEL 陶瓷悬浮液体系)")
    print("=" * 60)

    t0 = time.time()
    res_A = run_strategy_A(X_test, y_test)
    res_B = run_strategy_B(X_lf_train, y_lf_train, X_lf_test, y_lf_test,
                           X_test, y_test)
    res_C = run_strategy_C(X_hf_train, y_hf_train, X_hf_eval, y_hf_eval,
                           X_test, y_test)
    res_D = run_strategy_D(X_lf_train, y_lf_train, X_lf_test, y_lf_test,
                           X_hf_train, y_hf_train, X_hf_eval, y_hf_eval,
                           X_test, y_test,
                           freeze_n=args.freeze_n)
    elapsed = time.time() - t0

    results = [res_A, res_B, res_C, res_D]

    # ── 3. 打印汇总表 ──
    print(f"\n{'='*65}")
    print(f"{'策略':<22} {'R²':>8} {'MAE/Pa':>9} {'MAPE/%':>9}")
    print(f"{'─'*65}")
    labels_short = [
        'A. 纯物理公式 (φ_max 均值)',
        'B. 纯低保真',
        f'C. 纯高保真 ({HF_TRAIN_N} 条)',
        f'D. 多保真融合 (本文, {HF_TRAIN_N} 条)',
    ]
    for lbl, res in zip(labels_short, results):
        print(f"{lbl:<32} {res['R2']:>8.4f} {res['MAE']:>9.4f} {res['MAPE']:>9.1f}")
    print(f"{'─'*65}")
    print(f"多保真增益 (D vs C): "
          f"ΔR²={res_D['R2']-res_C['R2']:+.4f}  "
          f"ΔMAPE={res_D['MAPE']-res_C['MAPE']:+.1f} pp")
    print(f"总耗时: {elapsed:.0f} s")
    print(f"{'='*65}\n")

    # ── 4. 保存结果 ──
    summary = {
        'system':     'YODEL_ceramic',
        'equation':   'tau0 = M_Y * phi^2 / [phi_max * (phi_max - phi)^2]',
        'M_Y_Pa':     YodelPINN_v1.M_Y,
        'hf_train_n': HF_TRAIN_N,
        'test_n':     len(X_test),
        'elapsed_s':  round(elapsed, 1),
        'strategies': {r['label']: {k: v for k, v in r.items()
                                    if k not in ('model',)}
                       for r in results},
        'mf_gain_R2':   round(res_D['R2'] - res_C['R2'], 4),
        'mf_gain_MAPE': round(res_D['MAPE'] - res_C['MAPE'], 1),
    }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"汇总结果保存至 {out_dir / 'summary.json'}")

    # ── 5. 散点图 ──
    scatter_four(results, X_test, y_test, out_dir / 'scatter_four_strategies.png')

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YODEL 体系跨模型泛化验证实验')
    parser.add_argument('--lf-data',  default='data/yodel_lf',
                        help='低保真数据目录')
    parser.add_argument('--hf-data',  default='data/yodel_hf',
                        help='高保真数据目录')
    parser.add_argument('--out-dir',  default='multi_fidelity/results/yodel_exp',
                        help='结果输出目录')
    parser.add_argument('--freeze-n', type=int, default=1,
                        help='冻结前 N 个 Linear 层 (默认 1)')
    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(project_root)
    main(args)
