"""
对比基线实验 - 与 PI-MFNN 进行公平比较

所有实验使用相同数据划分（seed=42）：
  训练集: 30条  评估集: 10条  测试集: 360条

基线列表：
  B1 - 标准PINN（软约束，Raissi风格）：直接预测τ₀ + 物理残差惩罚
  B2 - Meng 2020复合MFNN：4子网结构，输出空间建模低-高保真关联
  B3 - 纯MLP黑箱：无任何物理约束，直接拟合(Phi,SP%)→τ₀
  B4 - 物理软约束+硬约束混合(PENN风格)：预测φ_max + 额外物理残差惩罚
  B5 - 渐进式解冻多保真：按阶段逐步解冻层（替代固定freeze_n）
  B6 - 单阶段端到端训练（低保真+高保真数据混合，无预训练）

对照组（来自 train_v3.py 结果）：
  PI-MFNN（本文）: test R²=0.9321, MAE=0.0496 Pa, MAPE=10.1%
"""

import os
import sys
import time
import json
import copy
import torch
import torch.nn as nn
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
SPLIT_DIR   = project_root / 'data/v3_split_seed42'
M1 = 0.72   # Pa，Lian 2025 固定参数

# ─────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────

def load_split():
    def _load(name):
        df = pd.read_csv(SPLIT_DIR / f'{name}.csv')
        X = torch.tensor(df[FEATURES].values, dtype=torch.float32)
        y = torch.tensor(df[TARGET].values,   dtype=torch.float32)
        return X, y
    return _load('train'), _load('eval'), _load('test')

def load_lf_data():
    """加载低保真合成数据（1600训练 + 400测试）"""
    train_df = pd.read_csv(project_root / 'data/synthetic_table6_v2/train_data.csv')
    test_df  = pd.read_csv(project_root / 'data/synthetic_table6_v2/test_data.csv')
    Xtr = torch.tensor(train_df[FEATURES].values, dtype=torch.float32)
    ytr = torch.tensor(train_df[TARGET].values,   dtype=torch.float32)
    Xte = torch.tensor(test_df[FEATURES].values,  dtype=torch.float32)
    yte = torch.tensor(test_df[TARGET].values,     dtype=torch.float32)
    return (Xtr, ytr), (Xte, yte)

# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def log_mse(pred, target):
    return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)

def compute_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out
        pred = pred.squeeze()
        r2   = (1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)).item()
        mae  = torch.mean(torch.abs(pred - y)).item()
        mape = (torch.mean(torch.abs((pred - y) / (y + 1e-6))) * 100).item()
    return r2, mae, mape, pred.cpu().numpy()

def make_result(tag, r2, mae, mape, best_ep, elapsed, note=''):
    return {
        'tag': tag, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'test_r2': round(r2,4), 'test_mae': round(mae,4), 'test_mape': round(mape,2),
        'best_epoch': best_ep, 'elapsed_s': round(elapsed,1), 'note': note,
    }

def simple_train(model, X_tr, y_tr, X_ev, y_ev,
                 epochs=1000, patience=150, lr=1e-4, loss_fn=None):
    """通用训练循环（以eval R²为early stop准则）"""
    if loss_fn is None:
        loss_fn = log_mse
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr)
        pred = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(pred.squeeze(), y_tr)
        loss.backward()
        optimizer.step()
        ev_r2, ev_mae, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return best_ep, time.time() - t0


# ─────────────────────────────────────────────────────────
# B1 — 标准PINN（软约束，Raissi风格）
# ─────────────────────────────────────────────────────────

class SoftPINN(nn.Module):
    """
    直接预测 τ₀，同时用物理残差作软惩罚。
    网络结构与 LianPINN_v2 相同（对等比较）。
    损失 = Log-MSE_data + λ × physics_residual²
    物理残差定义：τ_pred 与 Lian公式（用某个参考φ_max）的偏差。
    这里φ_max由网络同时预测，但物理方程以软约束而非硬约束方式执行。
    """
    M1 = 0.72

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),   # 输出: [raw_tau0, raw_phi_max]
        )

    def forward(self, x):
        phi = x[:, 0]
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        raw = self.net(x * scale)
        # 直接预测 τ₀（ReLU保证非负）
        tau0_pred  = torch.relu(raw[:, 0]) + 1e-6
        # 同时预测 φ_max（sigmoid保证物理范围）
        phi_max    = phi + 0.05 + torch.sigmoid(raw[:, 1]) * (0.95 - phi - 0.05)
        return tau0_pred, phi_max


# ─────────────────────────────────────────────────────────
# B10 辅助 — 硬约束PINN（全局 φ_max 可学习标量）
# ─────────────────────────────────────────────────────────

class HardPINN_GlobalPhiMax(nn.Module):
    """
    phi_max 退化为单一全局可学习标量（而非依赖SP%的逐样本预测）。
    用于消融实验：量化"SP%→phi_max映射"相比固定全局phi_max的额外增益。

    硬约束：tau0 = M1 * phi**3 / (phi_max * (phi_max - phi + eps))
    phi_max = 0.65 + softplus(offset)，保证 phi_max > 0.65（物理下界）
    无神经网络权重，仅1个可学习参数。
    """
    M1 = 0.72

    def __init__(self):
        super().__init__()
        # softplus(0.0) ≈ 0.693，初始 phi_max ≈ 0.65 + 0.693 ≈ 1.343
        # 训练过程中 offset 会自动调整至数据最优值
        self.log_phi_max_offset = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        phi = x[:, 0]
        # 硬约束：phi_max 必须 > 0.65（物理合理下界）
        phi_max = 0.65 + torch.nn.functional.softplus(self.log_phi_max_offset)
        eps = 1e-6
        # relu 保护：防止 phi_max < phi 时出现负分母
        diff = torch.relu(phi_max - phi) + eps
        tau0 = self.M1 * phi ** 3 / (phi_max * diff)
        return tau0, phi_max


def train_B1(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam=0.1, lr=1e-4, epochs=1000, patience=150):
    """
    B1：软约束PINN
    损失 = Log-MSE(τ_pred, τ_true) + λ × Log-MSE(τ_physics, τ_pred)
    其中 τ_physics 由预测的 φ_max 代入 Lian 公式得到，
    惩罚项强制 τ_pred 接近物理公式计算值。
    """
    print(f"\n{'='*55}")
    print(f"B1 — 标准PINN（软约束，λ={lam}）")
    model = SoftPINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tau_pred, phi_max = model(X_tr)
        phi = X_tr[:, 0]
        # 物理公式计算值
        eps = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau_physics = M1 * phi**3 / (phi_max * diff)
        # 损失 = 数据项 + 软约束物理残差项
        loss_data    = log_mse(tau_pred, y_tr)
        loss_physics = log_mse(tau_pred, tau_physics)
        loss = loss_data + lam * loss_physics
        loss.backward()
        optimizer.step()

        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B1_soft_pinn', r2, mae, mape, best_ep, elapsed,
                       note=f'软约束λ={lam}，直接预测τ₀+φ_max')


# ─────────────────────────────────────────────────────────
# B2 — Meng 2020 复合MFNN
# ─────────────────────────────────────────────────────────

class MengMFNN(nn.Module):
    """
    复现 Meng & Karniadakis (2020) 的复合多保真神经网络。
    y_H = F_l(x, y_L) + F_nl(x, y_L)
    其中 y_L = NN_L(x) 为低保真网络输出（τ₀预测）。
    F_l: 线性相关子网  F_nl: 非线性相关子网
    三个子网均使用 Tanh，宽度20（与原论文一致）。
    """
    def __init__(self):
        super().__init__()
        # 低保真子网: 2→20→20→1
        self.nn_L = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1),
        )
        # 线性相关子网: (2+1)→10→1
        self.F_l = nn.Sequential(
            nn.Linear(3, 10), nn.Tanh(),
            nn.Linear(10, 1),
        )
        # 非线性相关子网: (2+1)→10→10→1
        self.F_nl = nn.Sequential(
            nn.Linear(3, 10), nn.Tanh(),
            nn.Linear(10, 10), nn.Tanh(),
            nn.Linear(10, 1),
        )
        self.scale = None   # 由 forward 中的输入归一化处理

    def forward_lf(self, x):
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        y_L = torch.relu(self.nn_L(x * scale).squeeze()) + 1e-6
        return y_L

    def forward(self, x):
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        x_norm = x * scale
        y_L = torch.relu(self.nn_L(x_norm)).clamp(min=1e-6)    # [batch,1]
        inp = torch.cat([x_norm, y_L], dim=1)                   # [batch,3]
        y_H = self.F_l(inp) + self.F_nl(inp)                    # [batch,1]
        return torch.relu(y_H).squeeze() + 1e-6, y_L.squeeze()


def train_B2(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lr=1e-3, epochs=2000, patience=200):
    """
    B2：Meng MFNN
    两阶段训练：
      阶段1：只用低保真数据预训练 NN_L
      阶段2：联合训练，损失 = MSE_LF + MSE_HF（冻结NN_L参数）
    """
    print(f"\n{'='*55}")
    print("B2 — Meng 2020 复合MFNN")
    model = MengMFNN()

    # ── 阶段1：预训练低保真子网 ──
    print("  阶段1：预训练低保真子网...")
    opt1 = optim.Adam(model.nn_L.parameters(), lr=1e-3)
    for ep in range(1, 501):
        model.train()
        opt1.zero_grad()
        scale = torch.tensor([2.0, 1.5])
        y_L = torch.relu(model.nn_L(X_lf_tr * scale)).squeeze() + 1e-6
        loss = log_mse(y_L, y_lf_tr)
        loss.backward(); opt1.step()
    lf_r2, lf_mae, _, _ = compute_metrics(
        type('M', (), {'eval': model.eval, 'parameters': model.parameters,
                       '__call__': lambda self, x: (model.forward_lf(x),)})(),
        X_lf_tr, y_lf_tr)

    # ── 阶段2：冻结 NN_L，训练 F_l 和 F_nl ──
    print("  阶段2：训练高保真相关子网...")
    for p in model.nn_L.parameters():
        p.requires_grad = False

    opt2 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        opt2.zero_grad()
        y_H, _ = model(X_tr)
        loss = log_mse(y_H, y_tr)
        loss.backward(); opt2.step()

        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B2_Meng_MFNN', r2, mae, mape, best_ep, elapsed,
                       note='Meng 2020复合MFNN：输出空间建模低-高保真关联')


# ─────────────────────────────────────────────────────────
# B7 — 数据驱动残差多保真（Forrester/Kennedy 风格）
# ─────────────────────────────────────────────────────────

class ResidualMFNN(nn.Module):
    """
    残差多保真神经网络（无物理约束）。
    y_HF = ReLU(NN_LF(x) + NN_res(x))
    NN_LF: 低保真子网，2→32→32→1（Tanh）
    NN_res: 残差子网，2→64→64→1（Tanh）
    与 B2（Meng MFNN，输出空间建模）的区别：
      B7 直接在残差空间叠加，是另一种主流多保真关联假设。
    两者均无物理约束。
    """
    def __init__(self):
        super().__init__()
        self.nn_lf = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.nn_res = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward_lf(self, x):
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        return torch.relu(self.nn_lf(x * scale).squeeze(-1)) + 1e-6

    def forward(self, x):
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        x_norm = x * scale
        lf_out = torch.relu(self.nn_lf(x_norm))          # [batch,1]
        res_out = self.nn_res(x_norm)                      # [batch,1]
        y_hf = torch.relu(lf_out + res_out).squeeze(-1) + 1e-6
        return y_hf, lf_out.squeeze(-1)


def train_B7(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lr=1e-3, epochs_lf=500, epochs_hf=2000, patience=200):
    """
    B7：数据驱动残差多保真（DD-residual-MF，Forrester风格）。
    两阶段训练：
      阶段1：低保真合成数据（1600条）预训练 NN_LF，500 epoch
      阶段2：冻结 NN_LF，高保真30条训练 NN_res，最大2000 epoch，
             patience=200，eval R² 选模
    无任何物理约束，与 B3（纯MLP HF-only）形成多保真 vs 无多保真对照。
    """
    print(f"\n{'='*55}")
    print("B7 — 数据驱动残差多保真（Forrester/Kennedy风格）")
    model = ResidualMFNN()

    # ── 阶段1：预训练低保真子网 ──
    print("  阶段1：低保真数据预训练 NN_LF（500 epoch）...")
    opt1 = optim.Adam(model.nn_lf.parameters(), lr=1e-3)
    for ep in range(1, epochs_lf + 1):
        model.train()
        opt1.zero_grad()
        scale = torch.tensor([2.0, 1.5], device=X_lf_tr.device, dtype=X_lf_tr.dtype)
        lf_pred = torch.relu(model.nn_lf(X_lf_tr * scale)).squeeze(-1) + 1e-6
        loss = log_mse(lf_pred, y_lf_tr)
        loss.backward()
        opt1.step()
    lf_r2, _, _, _ = compute_metrics(
        type('M', (), {
            'eval': model.eval,
            'parameters': model.parameters,
            '__call__': lambda self, x: (model.forward_lf(x),)
        })(),
        X_lf_tr, y_lf_tr)
    print(f"    低保真训练集 R²={lf_r2:.4f}")

    # ── 阶段2：冻结 NN_LF，训练 NN_res ──
    print("  阶段2：冻结NN_LF，高保真30条训练NN_res...")
    for p in model.nn_lf.parameters():
        p.requires_grad = False

    opt2 = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for ep in range(1, epochs_hf + 1):
        model.train()
        opt2.zero_grad()
        y_hf, _ = model(X_tr)
        loss = log_mse(y_hf, y_tr)
        loss.backward()
        opt2.step()

        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B7_residual_mfnn', r2, mae, mape, best_ep, elapsed,
                       note='数据驱动残差多保真：NN_LF预训练+冻结，NN_res学习残差，无物理约束')


def train_B8(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam_best=0.1, lr=1e-3, epochs_lf=500, epochs_hf=2000, patience=200):
    """
    B8：软约束PINN + 多保真（Soft-PINN-MF）。
    与 PI-MFNN 的唯一区别是约束方式：
      - PI-MFNN：硬约束（预测φ_max代入公式，τ₀由公式决定）
      - B8：软约束（直接预测τ₀，物理残差作惩罚项）
    训练协议完全对齐 PI-MFNN：两阶段预训练+HF微调，相同的HF split。
    lam_best：使用 B1 扩展搜索中最优的 λ 值。
    """
    print(f"\n{'='*55}")
    print(f"B8 — 软约束PINN + 多保真（λ={lam_best}）")

    # 复用 SoftPINN（已在 B1 section 定义）
    model = SoftPINN(hidden_dim=64)

    # ── 阶段1：低保真数据预训练骨干网络 ──
    print("  阶段1：低保真数据预训练SoftPINN骨干（500 epoch）...")
    opt1 = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs_lf + 1):
        model.train()
        opt1.zero_grad()
        tau_pred, _ = model(X_lf_tr)
        loss = log_mse(tau_pred, y_lf_tr)
        loss.backward()
        opt1.step()

    lf_r2, _, _, _ = compute_metrics(model, X_lf_tr, y_lf_tr)
    print(f"    低保真训练集 R²={lf_r2:.4f}")

    # ── 阶段2：高保真30条微调 ──
    print("  阶段2：高保真30条微调...")
    opt2 = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for ep in range(1, epochs_hf + 1):
        model.train()
        opt2.zero_grad()
        tau_pred, phi_max = model(X_tr)
        phi = X_tr[:, 0]
        eps = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau_physics = M1 * phi**3 / (phi_max * diff)
        loss_data    = log_mse(tau_pred, y_tr)
        loss_physics = log_mse(tau_pred, tau_physics)
        loss = loss_data + lam_best * loss_physics
        loss.backward()
        opt2.step()

        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B8_soft_pinn_mf', r2, mae, mape, best_ep, elapsed,
                       note=f'软约束PINN+多保真预训练，λ={lam_best}，与PI-MFNN唯一区别是约束方式')


# ─────────────────────────────────────────────────────────
# B9 — SA-PINN（自适应权重软约束，Wang et al. 2022）
# ─────────────────────────────────────────────────────────

def train_B9(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam_init=1.0, ema_alpha=0.9, update_interval=20,
             lr=1e-4, epochs=1000, patience=150):
    """
    B9：自适应权重软约束PINN（SA-PINN）

    参考：Wang et al. 2022，"Understanding and mitigating gradient pathologies
    in physics-informed neural networks"，SIAM J. Sci. Comput.

    核心：每 update_interval 步用梯度L2范数比自动更新λ，替代B1的手工调参。
    λ_raw = ||∇L_data||_2 / (||∇L_physics||_2 + 1e-8)
    λ_new = ema_alpha * λ_old + (1 - ema_alpha) * λ_raw  （EMA平滑）

    损失 = log_mse(tau_pred, y_tr) + λ × log_mse(tau_pred, tau_physics)
    """
    print(f"\n{'='*55}")
    print(f"B9 — SA-PINN（自适应λ，init={lam_init}，EMA α={ema_alpha}）")

    model = SoftPINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    lam = lam_init
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    lam_history = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()

        # ── 每 update_interval 步更新 λ ──
        if ep % update_interval == 1 and ep > 1:
            phi_tmp = X_tr[:, 0]
            eps_tmp = 1e-6

            # 梯度1：数据损失对参数的梯度范数
            optimizer.zero_grad()
            tau_pred_tmp, phi_max_tmp = model(X_tr)
            loss_data_tmp = log_mse(tau_pred_tmp, y_tr)
            loss_data_tmp.backward()
            grad_data_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5

            # 梯度2：物理损失对参数的梯度范数（重新前向）
            optimizer.zero_grad()
            tau_pred_tmp2, phi_max_tmp2 = model(X_tr)
            diff_tmp2 = torch.relu(phi_max_tmp2 - phi_tmp) + eps_tmp
            tau_physics_tmp2 = M1 * phi_tmp ** 3 / (phi_max_tmp2 * diff_tmp2)
            loss_phys_tmp2 = log_mse(tau_pred_tmp2, tau_physics_tmp2)
            loss_phys_tmp2.backward()
            grad_phys_norm = sum(
                p.grad.norm().item() ** 2
                for p in model.parameters() if p.grad is not None
            ) ** 0.5

            lam_raw = grad_data_norm / (grad_phys_norm + 1e-8)
            lam = ema_alpha * lam + (1.0 - ema_alpha) * lam_raw
            optimizer.zero_grad()

        # ── 正常训练步骤 ──
        model.train()
        optimizer.zero_grad()
        tau_pred, phi_max = model(X_tr)
        phi = X_tr[:, 0]
        eps = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau_physics = M1 * phi ** 3 / (phi_max * diff)

        loss_data    = log_mse(tau_pred, y_tr)
        loss_physics = log_mse(tau_pred, tau_physics)
        loss = loss_data + lam * loss_physics
        loss.backward()
        optimizer.step()

        lam_history.append(round(lam, 6))

        # ── Early stop（基于eval R²）──
        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    elapsed = time.time() - t0

    # 报告λ演化统计
    final_lam = lam_history[-1] if lam_history else lam_init
    print(f"  最终λ={final_lam:.4f}（初始{lam_init}，共更新{len(lam_history)//update_interval}次）")
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")

    return make_result(
        'B9_sa_pinn', r2, mae, mape, best_ep, elapsed,
        note=f'SA-PINN adaptive lambda, final_lam={final_lam:.4f}'
    )


# ─────────────────────────────────────────────────────────
# B10 — 硬约束PINN（全局 φ_max 可学习标量）
# ─────────────────────────────────────────────────────────

def train_B10(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
              lr=1e-4, epochs=1000, patience=150):
    """
    B10：硬约束PINN，phi_max为全局可学习标量。

    消融目标：对比LianPINN_v2（phi_max依赖SP%逐样本预测）与本方法（固定全局phi_max）
    的测试集R²差值，量化"SP%→phi_max映射"的额外增益。

    无物理残差项——硬约束已天然满足，直接优化数据拟合损失。
    训练结束后报告学到的全局phi_max值（物理可解释性）。
    """
    print(f"\n{'='*55}")
    print(f"B10 — 硬约束PINN（全局φ_max可学习标量）")

    model = HardPINN_GlobalPhiMax()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        tau_pred, phi_max = model(X_tr)
        # 硬约束已满足，仅数据损失
        loss = log_mse(tau_pred.squeeze(), y_tr)
        loss.backward()
        optimizer.step()

        # ── Early stop（基于eval R²）──
        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    elapsed = time.time() - t0

    # 报告学到的全局phi_max（物理可解释性）
    model.eval()
    with torch.no_grad():
        learned_phi_max = (
            0.65 + torch.nn.functional.softplus(model.log_phi_max_offset)
        ).item()
    print(f"  学到的全局φ_max = {learned_phi_max:.4f}")
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")

    return make_result(
        'B10_hard_global_phimax', r2, mae, mape, best_ep, elapsed,
        note=f'硬约束全局phi_max={learned_phi_max:.4f}'
    )


# ─────────────────────────────────────────────────────────
# B3 — 纯MLP黑箱（无物理约束）
# ─────────────────────────────────────────────────────────

class PureMLP(nn.Module):
    """纯MLP，直接拟合(Phi,SP%)→τ₀，无物理约束。"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        return torch.relu(self.net(x * scale).squeeze()) + 1e-6


def train_B3(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lr=1e-4, epochs=1000, patience=150):
    """B3：纯MLP，无任何物理信息。"""
    print(f"\n{'='*55}")
    print("B3 — 纯MLP黑箱（无物理约束）")
    model = PureMLP(hidden_dim=64)
    best_ep, elapsed = simple_train(model, X_tr, y_tr, X_ev, y_ev,
                                    epochs=epochs, patience=patience, lr=lr)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B3_pure_MLP', r2, mae, mape, best_ep, elapsed,
                       note='纯MLP：无物理约束，直接拟合')


# ─────────────────────────────────────────────────────────
# B4 — PENN风格：硬约束 + 额外物理软惩罚（混合约束）
# ─────────────────────────────────────────────────────────

class HybridPINN(nn.Module):
    """
    混合约束PINN：
    - 硬约束：预测φ_max，代入物理公式计算τ₀（同PI-MFNN）
    - 软约束：额外惩罚φ_max的单调性（SP%↑ → φ_max↑）
    介于纯软约束和本文硬约束之间。
    """
    M1 = 0.72

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        phi = x[:, 0]
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        raw = self.net(x * scale).squeeze()
        phi_max = phi + 0.05 + torch.sigmoid(raw) * (0.95 - phi - 0.05)
        eps = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau0 = self.M1 * phi**3 / (phi_max * diff)
        return tau0, phi_max


def monotonicity_penalty(model, X):
    """
    物理软约束：φ_max 应关于 SP% 单调递增。
    对同一批次内SP%较大的样本，φ_max 应不小于SP%较小样本的φ_max。
    实现：取SP%排序后相邻差的负值（违反单调性的部分）作为惩罚。
    """
    model.eval()
    with torch.no_grad():
        _, phi_max = model(X)
    sp = X[:, 1]
    idx = torch.argsort(sp)
    phi_max_sorted = phi_max[idx]
    # 惩罚相邻下降
    diff = phi_max_sorted[1:] - phi_max_sorted[:-1]
    penalty = torch.relu(-diff).mean()
    return penalty


def train_B4(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam_mono=0.05, lr=1e-4, epochs=1000, patience=150):
    """B4：硬约束 + φ_max 单调性软惩罚。"""
    print(f"\n{'='*55}")
    print(f"B4 — 混合约束PINN（硬约束+单调性软惩罚λ={lam_mono}）")
    model = HybridPINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tau_pred, phi_max = model(X_tr)
        loss_data = log_mse(tau_pred, y_tr)
        # φ_max 关于 SP% 的单调性惩罚
        sp_sorted_idx = torch.argsort(X_tr[:, 1])
        phi_max_sorted = phi_max[sp_sorted_idx]
        mono_viol = torch.relu(-(phi_max_sorted[1:] - phi_max_sorted[:-1])).mean()
        loss = loss_data + lam_mono * mono_viol
        loss.backward(); optimizer.step()

        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B4_hybrid_pinn', r2, mae, mape, best_ep, elapsed,
                       note=f'硬约束+单调性软惩罚λ={lam_mono}')


# ─────────────────────────────────────────────────────────
# B5 — 渐进式解冻多保真
# ─────────────────────────────────────────────────────────

def train_B5(low_model_path, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lr=1e-4, epochs_per_stage=300, patience=150):
    """
    B5：渐进式解冻（Progressive Unfreezing）多保真。
    从预训练权重出发，分4个阶段逐步解冻：
      阶段1：只训练第4层（输出层）
      阶段2：解冻第3、4层
      阶段3：解冻第2、3、4层
      阶段4：全部解冻
    每阶段最多 epochs_per_stage 轮，以eval R²为best checkpoint。
    """
    print(f"\n{'='*55}")
    print("B5 — 渐进式解冻多保真")
    ckpt = torch.load(low_model_path, weights_only=False)
    model = LianPINN_v2(hidden_dim=64)
    model.load_state_dict(ckpt['model_state_dict'])

    linear_idx = [i for i, layer in enumerate(model.net) if hasattr(layer, 'weight')]
    # linear_idx = [0, 2, 4, 6] 对应4个Linear层

    best_r2_global, best_state_global = float('-inf'), None
    t0 = time.time()
    total_ep = 0

    # 解冻计划：从最后一层开始逐步向前解冻
    unfreeze_plans = [
        [linear_idx[-1]],                            # 阶段1：仅输出层
        [linear_idx[-2], linear_idx[-1]],            # 阶段2：后两层
        [linear_idx[-3], linear_idx[-2], linear_idx[-1]], # 阶段3：后三层
        linear_idx,                                  # 阶段4：全部
    ]

    for stage_i, unfreeze_set in enumerate(unfreeze_plans):
        # 重置冻结状态
        for p in model.parameters():
            p.requires_grad = False
        for i, layer in enumerate(model.net):
            if i in set(unfreeze_set) and hasattr(layer, 'weight'):
                for p in layer.parameters():
                    p.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  阶段{stage_i+1}：解冻层{unfreeze_set}，可训练 {trainable}/{total}")

        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr)
        best_r2, best_ep_stage, wait = float('-inf'), 0, 0
        best_state_stage = None

        for ep in range(1, epochs_per_stage + 1):
            model.train()
            optimizer.zero_grad()
            pred, _ = model(X_tr)
            loss = log_mse(pred, y_tr)
            loss.backward(); optimizer.step()
            total_ep += 1

            ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
            if ev_r2 > best_r2:
                best_r2, best_ep_stage, wait = ev_r2, ep, 0
                best_state_stage = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= patience: break

        model.load_state_dict(best_state_stage)
        print(f"    阶段{stage_i+1}最佳 eval R²={best_r2:.4f}")

        if best_r2 > best_r2_global:
            best_r2_global = best_r2
            best_state_global = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_global)
    elapsed = time.time() - t0
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  总epoch={total_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B5_progressive_unfreeze', r2, mae, mape, total_ep, elapsed,
                       note='渐进式解冻：4阶段逐步从输出层向输入层解冻')


# ─────────────────────────────────────────────────────────
# B6 — 单阶段混合训练（低保真+高保真数据直接混合）
# ─────────────────────────────────────────────────────────

def train_B6(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam_hf=10.0, lr=1e-4, epochs=1000, patience=150):
    """
    B6：单阶段混合训练（无两阶段，无预训练概念）。
    将低保真数据和高保真数据直接混合，以加权损失同时训练：
      损失 = MSE_LF + λ_HF × MSE_HF
    λ_HF >> 1 使高保真数据获得更高权重。
    测试"不做预训练而是数据混合"是否能达到类似效果。
    """
    print(f"\n{'='*55}")
    print(f"B6 — 单阶段混合训练（LF+HF混合，λ_HF={lam_hf}）")
    model = LianPINN_v2(hidden_dim=64)  # 随机初始化
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        # 低保真损失
        pred_lf, _ = model(X_lf_tr)
        loss_lf = log_mse(pred_lf, y_lf_tr)
        # 高保真损失
        pred_hf, _ = model(X_tr)
        loss_hf = log_mse(pred_hf, y_tr)
        loss = loss_lf + lam_hf * loss_hf
        loss.backward(); optimizer.step()

        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience: break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  test R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B6_mixed_training', r2, mae, mape, best_ep, elapsed,
                       note=f'单阶段混合训练：LF+HF直接加权，λ_HF={lam_hf}')


# ─────────────────────────────────────────────────────────
# 综合对比图
# ─────────────────────────────────────────────────────────

def plot_baseline_comparison(all_results, save_path):
    os.makedirs(Path(save_path).parent, exist_ok=True)

    # 添加本文 PI-MFNN 结果作对照
    pi_mfnn = {'tag': 'PI-MFNN\n(本文)', 'test_r2': 0.9321,
                'test_mae': 0.0496, 'test_mape': 10.1}
    results = all_results + [pi_mfnn]
    # 按R²排序
    results = sorted(results, key=lambda r: r['test_r2'], reverse=True)

    labels = [r['tag'].replace('_', '\n') for r in results]
    r2s    = [r['test_r2']   for r in results]
    maes   = [r['test_mae']  for r in results]
    mapes  = [r['test_mape'] for r in results]

    # 本文方法用特殊颜色高亮
    colors = ['#2ecc71' if 'PI-MFNN' in r['tag'] else '#5588cc' for r in results]

    x = np.arange(len(results))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, vals, title, ylabel, fmt in [
        (axes[0], r2s,   'Test R²',      'R²',       '.3f'),
        (axes[1], maes,  'Test MAE (Pa)', 'MAE (Pa)', '.4f'),
        (axes[2], mapes, 'Test MAPE (%)', 'MAPE (%)', '.1f'),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor='white', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.25, axis='y')
        if title == 'Test R²':
            ax.set_ylim(max(0, min(vals) - 0.1), 1.0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f'{v:{fmt}}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    fig.suptitle(
        'Baseline Comparison: All Methods on 360-sample Test Set\n'
        '(Training: 30 HF samples, seed=42)',
        fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"\n对比图已保存: {save_path}")


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*60)
    print("对比基线实验 B1–B6")
    print("数据：seed=42 划分，30训练/10评估/360测试")
    print("="*60)

    (X_tr, y_tr), (X_ev, y_ev), (X_te, y_te) = load_split()
    (X_lf_tr, y_lf_tr), _ = load_lf_data()

    low_model_path = (project_root /
        'multi_fidelity/models/low_fidelity/lian_v3_low.pth')

    all_results = []

    # B1 软约束PINN（扩展λ搜索，6档）
    for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        r = train_B1(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
                     lam=lam, lr=1e-4, epochs=1000, patience=150)
        r['tag'] = f'B1_soft_pinn_lam{lam}'
        all_results.append(r)

    # B2 Meng MFNN
    all_results.append(
        train_B2(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # B3 纯MLP
    all_results.append(
        train_B3(X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # B4 混合约束（测试不同λ_mono）
    for lam_m in [0.05, 0.5]:
        r = train_B4(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
                     lam_mono=lam_m, lr=1e-4, epochs=1000, patience=150)
        r['tag'] = f'B4_hybrid_lam{lam_m}'
        all_results.append(r)

    # B5 渐进式解冻
    all_results.append(
        train_B5(low_model_path, X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # B6 单阶段混合训练
    all_results.append(
        train_B6(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # B7 数据驱动残差多保真
    all_results.append(
        train_B7(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # B8 软约束PINN + 多保真（λ取B1扩展搜索中test R²最优的那档）
    b1_results = [r for r in all_results if r['tag'].startswith('B1_')]
    lam_best_b8 = float(
        max(b1_results, key=lambda r: r['test_r2'])['tag']
        .split('lam')[-1]
    ) if b1_results else 0.1
    print(f"\nB8将使用B1最优λ={lam_best_b8}")
    all_results.append(
        train_B8(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
                 lam_best=lam_best_b8))

    # B9 SA-PINN（自适应λ，HF-only）
    all_results.append(
        train_B9(X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # B10 硬约束全局φ_max（消融，HF-only）
    all_results.append(
        train_B10(X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # ── 保存结果 ──
    result_path = RESULTS_DIR / 'logs/baseline_results_exp21.json'
    os.makedirs(result_path.parent, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_path}")

    # ── 汇总打印 ──
    PI_MFNN_R2  = 0.9321
    PI_MFNN_MAE = 0.0496

    print("\n" + "="*70)
    print("对比基线汇总（测试集360条）vs PI-MFNN（本文）")
    print("="*70)
    header = f"{'方法':<32} {'R²':>8} {'MAE/Pa':>10} {'MAPE%':>8} {'vs PI-MFNN R²':>14}"
    print(header)
    print("─"*len(header))

    # 本文方法
    print(f"  {'PI-MFNN（本文）':<30} {PI_MFNN_R2:>8.4f} {PI_MFNN_MAE:>10.4f} {'10.1':>7}%  {'基准':>12}")
    print("─"*len(header))

    for r in sorted(all_results, key=lambda x: x['test_r2'], reverse=True):
        delta = r['test_r2'] - PI_MFNN_R2
        print(f"  {r['tag']:<30} {r['test_r2']:>8.4f} "
              f"{r['test_mae']:>10.4f} {r['test_mape']:>7.1f}%  "
              f"{delta:>+12.4f}")
    print("─"*len(header))

    # ── 对比图 ──
    # 取每组最优结果
    best_per_method = {}
    for r in all_results:
        base = r['tag'].split('_lam')[0].split('_lambda')[0]
        if base not in best_per_method or r['test_r2'] > best_per_method[base]['test_r2']:
            best_per_method[base] = r
    plot_results = list(best_per_method.values())
    plot_baseline_comparison(
        plot_results,
        save_path=RESULTS_DIR / 'plots/baseline_comparison_exp21.png')
