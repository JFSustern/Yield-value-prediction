"""
Zhou 1999 体系跨模型泛化验证实验

用途: PI-MFNN 框架在 Zhou 1999 Al₂O₃ 悬浮液真实数据上的验证，
      论文跨体系泛化实验的配套训练脚本

物理方程 (完整 YODEL Eq. 42):
    τ = m₁_eff × φ(φ − φ₀)² / [φ_max_ref × (φ_max_ref − φ)]
    固定: φ_max_ref=0.570, φ₀=0.026
    反演: m₁_eff (颗粒间力参数, Pa)

与 Lian 2025 体系的架构对称性:
    Lian:  输入=[φ,SP%]  反演=φ_max  PCL=Lian方程  HF=自有真实数据
    Zhou:  输入=[φ,d_s]  反演=m₁_eff PCL=YODEL方程 HF=Zhou 1999实验数据

实验设计（镜像四策略消融）:
    策略 A: 纯物理公式    m₁_eff 取各粉末固定均值，φ_max=0.570 固定
    策略 B: 纯低保真      仅用 LF 合成数据训练
    策略 C: 纯高保真      30 条 HF 随机初始化训练
    策略 D: 多保真融合    LF 预训练 → 30 条 HF 冻结层微调

运行方式:
  cd /path/to/Project
  python -m multi_fidelity.src.training.run_zhou1999_experiment \\
      --lf-data  data/zhou1999_lf \\
      --hf-data  data/zhou1999_hf \\
      --out-dir  multi_fidelity/results/zhou1999_exp
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

from multi_fidelity.src.model.pinn_zhou1999_v1 import ZhouPINN_v1   # noqa

# ── 常量 ──────────────────────────────────────────────────────────────────

FEATURES   = ['phi', 'd_s_um']
TARGET     = 'tau_Pa'
SEED       = 42
HF_TRAIN_N = 30
HF_EVAL_N  = 10

# 策略 A 用固定 m₁ 均值 (各粉末 m₁ 的简单平均)
M1_FIXED_MEAN = 747.0    # (310+470+830+1380)/4 ≈ 747 Pa

# ── 工具 ──────────────────────────────────────────────────────────────────

def set_seed(s: int = SEED):
    np.random.seed(s); torch.manual_seed(s)


def log_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)


def compute_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        pred, m1 = model(X)
        r2   = (1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)).item()
        mae  = torch.mean(torch.abs(pred - y)).item()
        mape = (torch.mean(torch.abs((pred - y) / (y + 1e-6))) * 100).item()
    return r2, mae, mape, pred.cpu().numpy()


def load_tensors(path: Path):
    df = pd.read_csv(path)
    X  = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y  = torch.tensor(df[TARGET].values,   dtype=torch.float32)
    return X, y, df

# ── 训练循环 ──────────────────────────────────────────────────────────────

def train_lf(model, X_tr, y_tr, X_te, y_te,
             lr=1e-3, max_epochs=500, patience=50, batch_size=64):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False)
    best_r2, best_state, wait = -1.0, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        idx = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            b = idx[i:i+batch_size]
            optimizer.zero_grad()
            pred, _ = model(X_tr[b])
            loss = log_mse(pred, y_tr[b])
            loss.backward(); optimizer.step()
        r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
        scheduler.step(1 - r2)
        if r2 > best_r2:
            best_r2, best_state, wait = r2, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    r2, _, mape, _ = compute_metrics(model, X_te, y_te)
    print(f'  LF 预训练完成 | epoch={ep} | 合成测试 R²={r2:.4f} MAPE={mape:.1f}%')
    return best_r2


def train_hf(model, X_tr, y_tr, X_ev, y_ev,
             lr=1e-4, max_epochs=1000, patience=150, freeze_n=1):
    linear_layers = [m for m in model.net if hasattr(m, 'weight')]
    for layer in linear_layers[:freeze_n]:
        for p in layer.parameters(): p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=lr, weight_decay=1e-5)
    best_r2, best_state, wait = -1.0, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _ = model(X_tr)
        loss = log_mse(pred, y_tr)
        loss.backward(); optimizer.step()
        r2_ev, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if r2_ev > best_r2:
            best_r2, best_state, wait = r2_ev, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    for p in model.parameters(): p.requires_grad = True
    return best_r2, ep

# ── 四策略 ────────────────────────────────────────────────────────────────

def run_strategy_A(X_test, y_test) -> dict:
    """策略 A: 纯物理公式 (m₁_eff = 固定均值)."""
    phi  = X_test[:, 0]
    phi0 = ZhouPINN_v1.PHI_0
    pm   = ZhouPINN_v1.PHI_MAX_REF
    eps  = 1e-6
    num  = M1_FIXED_MEAN * phi * torch.clamp(phi - phi0, min=eps) ** 2
    den  = pm * torch.clamp(pm - phi, min=eps)
    pred = num / den
    y    = y_test
    r2   = (1 - torch.sum((y-pred)**2) / torch.sum((y-y.mean())**2)).item()
    mae  = torch.mean(torch.abs(pred-y)).item()
    mape = (torch.mean(torch.abs((pred-y)/(y+1e-6)))*100).item()
    print(f'  策略 A (纯物理) | m₁_fixed={M1_FIXED_MEAN} Pa | '
          f'R²={r2:.4f} MAPE={mape:.1f}%')
    return dict(label='A_physics', R2=r2, MAE=mae, MAPE=mape, m1_fixed=M1_FIXED_MEAN)


def run_strategy_B(X_lf_tr, y_lf_tr, X_lf_te, y_lf_te, X_te, y_te) -> dict:
    set_seed()
    model = ZhouPINN_v1()
    best_lf_r2 = train_lf(model, X_lf_tr, y_lf_tr, X_lf_te, y_lf_te)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f'  策略 B (纯低保真) | test R²={r2:.4f} MAPE={mape:.1f}%')
    return dict(label='B_lf_only', R2=r2, MAE=mae, MAPE=mape,
                lf_pretrain_R2=best_lf_r2, model=model)


def run_strategy_C(X_hf_tr, y_hf_tr, X_hf_ev, y_hf_ev, X_te, y_te) -> dict:
    set_seed()
    model = ZhouPINN_v1()
    _, best_epoch = train_hf(model, X_hf_tr, y_hf_tr, X_hf_ev, y_hf_ev,
                             lr=1e-4, freeze_n=0)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f'  策略 C (纯高保真) | best_epoch={best_epoch} | '
          f'test R²={r2:.4f} MAPE={mape:.1f}%')
    return dict(label='C_hf_only', R2=r2, MAE=mae, MAPE=mape,
                best_epoch=best_epoch, model=model)


def run_strategy_D(X_lf_tr, y_lf_tr, X_lf_te, y_lf_te,
                   X_hf_tr, y_hf_tr, X_hf_ev, y_hf_ev,
                   X_te, y_te, freeze_n=1) -> dict:
    set_seed()
    model = ZhouPINN_v1()
    best_lf_r2 = train_lf(model, X_lf_tr, y_lf_tr, X_lf_te, y_lf_te)
    best_ev_r2, best_epoch = train_hf(model, X_hf_tr, y_hf_tr, X_hf_ev, y_hf_ev,
                                      lr=1e-4, freeze_n=freeze_n)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f'  策略 D (多保真)   | freeze={freeze_n} epoch={best_epoch} | '
          f'test R²={r2:.4f} MAPE={mape:.1f}%')
    return dict(label='D_mfnn', R2=r2, MAE=mae, MAPE=mape,
                best_epoch=best_epoch, lf_pretrain_R2=best_lf_r2,
                freeze_n=freeze_n, model=model)

# ── 散点图 ────────────────────────────────────────────────────────────────

def scatter_four(results, X_test, y_test, out_path: Path):
    y_np  = y_test.cpu().numpy()
    # log scale for 0–7000 Pa range
    lim   = [max(y_np.min() * 0.6, 0.5), y_np.max() * 1.6]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    panel_labels = ['(a) Pure Physics', '(b) LF Only', '(c) HF Only', '(d) PI-MFNN']

    # strategy A prediction
    phi  = X_test[:, 0]
    phi0 = ZhouPINN_v1.PHI_0; pm = ZhouPINN_v1.PHI_MAX_REF; eps = 1e-6
    pred_A = (M1_FIXED_MEAN * phi * torch.clamp(phi-phi0,min=eps)**2 /
              (pm * torch.clamp(pm-phi, min=eps))).numpy()

    preds = [pred_A]
    for res in results[1:]:
        if 'model' in res:
            _, _, _, p = compute_metrics(res['model'], X_test, y_test)
            preds.append(p)
        else:
            preds.append(np.full_like(y_np, np.nan))

    for ax, pred, lbl, res in zip(axes, preds, panel_labels, results):
        mask = ~np.isnan(pred)
        ax.scatter(y_np[mask], pred[mask], alpha=0.7, s=25, color='steelblue')
        ax.plot(lim, lim, 'r--', lw=1.2)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lim);  ax.set_ylim(lim)
        ax.set_xlabel('True τ (Pa)', fontsize=8)
        ax.set_ylabel('Pred τ (Pa)', fontsize=8)
        ax.set_title(f"{lbl}\nR²={res['R2']:.3f}  MAPE={res['MAPE']:.1f}%", fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.suptitle('Zhou 1999 Al₂O₃: Four-Strategy Comparison', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'散点图保存至 {out_path}')

# ── 主程序 ────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    X_lf_tr, y_lf_tr, _ = load_tensors(Path(args.lf_data) / 'train.csv')
    X_lf_te, y_lf_te, _ = load_tensors(Path(args.lf_data) / 'test.csv')
    X_hf_tr, y_hf_tr, _ = load_tensors(Path(args.hf_data) / 'train_scarce.csv')
    X_hf_ev, y_hf_ev, _ = load_tensors(Path(args.hf_data) / 'eval.csv')
    X_te,    y_te,    _  = load_tensors(Path(args.hf_data) / 'test.csv')

    print(f'\n数据摘要:')
    print(f'  LF 训练: {len(X_lf_tr)}  LF 测试: {len(X_lf_te)}')
    print(f'  HF 训练: {len(X_hf_tr)}  HF 评估: {len(X_hf_ev)}  测试: {len(X_te)}')
    print(f'  tau 范围 (test): {y_te.min():.1f} – {y_te.max():.1f} Pa')
    if len(X_te) < 15:
        print(f'  ⚠️  测试集仅 {len(X_te)} 条（受 Zhou 1999 公开数据量限制），指标供参考')
    print()

    print('=' * 60)
    print('四策略消融实验 (Zhou 1999 Al₂O₃ 体系 — 真实 HF 数据)')
    print('=' * 60)

    t0 = time.time()
    res_A = run_strategy_A(X_te, y_te)
    res_B = run_strategy_B(X_lf_tr, y_lf_tr, X_lf_te, y_lf_te, X_te, y_te)
    res_C = run_strategy_C(X_hf_tr, y_hf_tr, X_hf_ev, y_hf_ev, X_te, y_te)
    res_D = run_strategy_D(X_lf_tr, y_lf_tr, X_lf_te, y_lf_te,
                           X_hf_tr, y_hf_tr, X_hf_ev, y_hf_ev,
                           X_te, y_te, freeze_n=args.freeze_n)
    elapsed = time.time() - t0

    results = [res_A, res_B, res_C, res_D]
    lbls    = ['A. 纯物理 (m₁ 均值)', 'B. 纯低保真',
               f'C. 纯高保真 ({HF_TRAIN_N}条)',
               f'D. 多保真融合 (本文, {HF_TRAIN_N}条)']

    print(f'\n{"="*65}')
    print(f'{"策略":<32} {"R²":>8} {"MAE/Pa":>9} {"MAPE/%":>9}')
    print(f'{"─"*65}')
    for lbl, res in zip(lbls, results):
        print(f'{lbl:<32} {res["R2"]:>8.4f} {res["MAE"]:>9.1f} {res["MAPE"]:>9.1f}')
    print(f'{"─"*65}')
    print(f'多保真增益 (D vs C): '
          f'ΔR²={res_D["R2"]-res_C["R2"]:+.4f}  '
          f'ΔMAPE={res_D["MAPE"]-res_C["MAPE"]:+.1f} pp')
    print(f'总耗时: {elapsed:.0f} s')
    print(f'{"="*65}\n')

    summary = dict(
        system    = 'Zhou1999_Al2O3',
        equation  = 'tau = m1_eff * phi*(phi-phi0)^2 / [phi_max*(phi_max-phi)]',
        phi_max_fixed = ZhouPINN_v1.PHI_MAX_REF,
        phi0_fixed    = ZhouPINN_v1.PHI_0,
        hf_train_n    = HF_TRAIN_N,
        test_n        = len(X_te),
        elapsed_s     = round(elapsed, 1),
        strategies    = {r['label']: {k: v for k, v in r.items() if k != 'model'}
                         for r in results},
        mf_gain_R2    = round(res_D['R2'] - res_C['R2'], 4),
        mf_gain_MAPE  = round(res_D['MAPE'] - res_C['MAPE'], 1),
        note = ('HF data: estimated digitization of Zhou 1999 Fig.1. '
                'Replace with WebPlotDigitizer results before publication.'),
    )

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'结果保存至 {out_dir / "summary.json"}')

    scatter_four(results, X_te, y_te, out_dir / 'scatter_four_zhou1999.png')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zhou 1999 Al₂O₃ 体系跨模型验证')
    parser.add_argument('--lf-data',  default='data/zhou1999_lf')
    parser.add_argument('--hf-data',  default='data/zhou1999_hf')
    parser.add_argument('--out-dir',  default='multi_fidelity/results/zhou1999_exp')
    parser.add_argument('--freeze-n', type=int, default=1)
    args = parser.parse_args()
    os.chdir(project_root)
    main(args)
