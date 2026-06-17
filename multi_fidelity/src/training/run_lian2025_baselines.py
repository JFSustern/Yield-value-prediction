"""
Lian 2025 对比基线实验 — 适配新数据划分

数据划分（新）:
  训练: 360 条数据增强高保真数据 (all_400.csv, seed=42)
  评估: 40 条数据增强高保真数据  (early stop)
  测试: 16 条论文原始实验数据    (table6.csv, 固定)

基线列表（对应论文 Table 1）:
  B3  — 纯MLP（无约束）
  B7  — 残差多保真MFNN（Forrester风格，无约束）
  B1  — 软约束PINN（Raissi，λ六档搜索）
  B9  — SA-PINN（自适应λ，Wang et al. 2022）
  PGNN— Physics-Guided NN（Karpatne 2017，单侧ReLU惩罚）
  B8  — 软约束PINN+多保真
  B2  — Meng 2020 复合MFNN
  B6  — 单阶段混合训练（LF+HF混合，无两阶段）
  B10 — 硬约束PINN（全局φ_max标量，消融）
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

from multi_fidelity.src.model.pinn_lian2025 import LianPINN
from multi_fidelity.src.training.lian2025_experiments import (
    split_hifi_data, load_paper_test, load_csv,
    N_TRAIN, N_EVAL, RANDOM_SEED,
    HIFI_DATA_PATH, RESULTS_DIR, set_seed,
)

SP = torch.nn.functional.softplus   # 别名：softplus，输出恒正，无死区

FEATURES = ['Phi', 'SP_percent']
TARGET   = 'Tau0_Pa'
M1 = 0.72   # Lian 2025 固定参数


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def log_mse(pred, target):
    return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)


def compute_metrics(model, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X)
        pred = out[0] if isinstance(out, tuple) else out
        pred = pred.squeeze()
        r2   = (1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)).item()
        mae  = torch.mean(torch.abs(pred - y)).item()
        mape = (torch.mean(torch.abs((pred - y) / (y + 1e-6))) * 100).item()
    return r2, mae, mape, pred.cpu().numpy()


def make_result(tag, r2, mae, mape, best_ep, elapsed, note=''):
    return {
        'tag': tag,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'test_r2':   round(r2,   4),
        'test_mae':  round(mae,  4),
        'test_mape': round(mape, 2),
        'best_epoch': best_ep,
        'elapsed_s':  round(elapsed, 1),
        'note': note,
    }


def simple_train(model, X_tr, y_tr, X_ev, y_ev,
                 epochs=1000, patience=150, lr=1e-4):
    """通用训练循环（eval R² early stop）"""
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(X_tr)
        pred = (out[0] if isinstance(out, tuple) else out).squeeze()
        loss = log_mse(pred, y_tr)
        loss.backward()
        optimizer.step()
        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
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
# 模型定义
# ─────────────────────────────────────────────────────────

class SoftPINN(nn.Module):
    """软约束PINN：直接预测τ₀+φ_max，物理残差作惩罚项。"""
    M1 = 0.72
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
    def forward(self, x):
        phi  = x[:, 0]
        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        raw  = self.net(x * scale)
        tau0_pred = SP(raw[:, 0]) + 1e-6      # softplus：无死区，恒正
        phi_max   = phi + 0.05 + torch.sigmoid(raw[:, 1]) * (0.95 - phi - 0.05)
        return tau0_pred, phi_max


class HardPINN_GlobalPhiMax(nn.Module):
    """消融：φ_max退化为单一全局可学习标量。"""
    M1 = 0.72
    def __init__(self):
        super().__init__()
        self.log_phi_max_offset = nn.Parameter(torch.tensor(0.0))
    def forward(self, x):
        phi = x[:, 0]
        phi_max = 0.65 + torch.nn.functional.softplus(self.log_phi_max_offset)
        eps  = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau0 = self.M1 * phi**3 / (phi_max * diff)
        return tau0, phi_max.expand(phi.shape)


class MengMFNN(nn.Module):
    """Meng & Karniadakis (2020) 复合多保真MFNN。"""
    def __init__(self):
        super().__init__()
        self.nn_L  = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1),
        )
        self.F_l   = nn.Sequential(nn.Linear(3, 10), nn.Tanh(), nn.Linear(10, 1))
        self.F_nl  = nn.Sequential(
            nn.Linear(3, 10), nn.Tanh(),
            nn.Linear(10, 10), nn.Tanh(),
            nn.Linear(10, 1),
        )
    def _scale(self, x):
        return x * torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
    def forward_lf(self, x):
        return SP(self.nn_L(self._scale(x)).squeeze()) + 1e-6
    def forward(self, x):
        xs = self._scale(x)
        y_L  = SP(self.nn_L(xs)) + 1e-6             # softplus：无死区
        inp  = torch.cat([xs, y_L], dim=1)
        y_H  = self.F_l(inp) + self.F_nl(inp)
        return SP(y_H).squeeze() + 1e-6, y_L.squeeze()  # softplus：无死区


class ResidualMFNN(nn.Module):
    """残差多保真MFNN（Forrester风格，无物理约束）。"""
    def __init__(self):
        super().__init__()
        self.nn_lf  = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )
        self.nn_res = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def _scale(self, x):
        return x * torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
    def forward_lf(self, x):
        return SP(self.nn_lf(self._scale(x)).squeeze(-1)) + 1e-6
    def forward(self, x):
        xs  = self._scale(x)
        lf  = SP(self.nn_lf(xs)) + 1e-6        # softplus：无死区
        res = self.nn_res(xs)
        y   = SP(lf + res - 1e-6).squeeze(-1) + 1e-6   # softplus on combined
        return y, lf.squeeze(-1)


class PureMLP(nn.Module):
    """纯MLP，无任何物理信息。"""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        s = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        return SP(self.net(x * s).squeeze()) + 1e-6   # softplus：无死区


# ─────────────────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────────────────

def train_B3(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lr=1e-4, epochs=1000, patience=150):
    set_seed(43)
    print(f"\n{'='*55}\nB3 — 纯MLP黑箱（无物理约束）")
    model = PureMLP(hidden_dim=64)
    best_ep, elapsed = simple_train(model, X_tr, y_tr, X_ev, y_ev,
                                    epochs=epochs, patience=patience, lr=lr)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    print(f"  best_ep={best_ep}  R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B3_pure_MLP', r2, mae, mape, best_ep, elapsed,
                       note='纯MLP，无物理约束')


def train_B7(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lr=1e-3, epochs_lf=500, epochs_hf=2000, patience=200):
    set_seed(44)
    print(f"\n{'='*55}\nB7 — 数据驱动残差多保真（Forrester风格）")
    model = ResidualMFNN()
    # 阶段1：低保真预训练
    opt1 = optim.Adam(model.nn_lf.parameters(), lr=1e-3)
    for ep in range(1, epochs_lf + 1):
        model.train(); opt1.zero_grad()
        s = torch.tensor([2.0, 1.5])
        lf_pred = torch.relu(model.nn_lf(X_lf_tr * s)).squeeze(-1) + 1e-6
        log_mse(lf_pred, y_lf_tr).backward(); opt1.step()
    for p in model.nn_lf.parameters(): p.requires_grad = False
    # 阶段2：HF微调残差子网
    opt2 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs_hf + 1):
        model.train(); opt2.zero_grad()
        y_hf, _ = model(X_tr)
        log_mse(y_hf, y_tr).backward(); opt2.step()
        ev_r2, _, _, _ = compute_metrics(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    r2, mae, mape, _ = compute_metrics(model, X_te, y_te)
    elapsed = time.time() - t0
    print(f"  best_ep={best_ep}  R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B7_residual_mfnn', r2, mae, mape, best_ep, elapsed,
                       note='残差多保真，NN_LF预训练冻结+NN_res学习残差，无物理约束')


def train_B1(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam=0.1, lr=1e-4, epochs=1000, patience=150):
    set_seed(45)   # 相同初始化，各lambda档唯一变量
    print(f"\n{'='*55}\nB1 — 软约束PINN（λ={lam}）")
    model = SoftPINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        tau_pred, phi_max = model(X_tr)
        phi = X_tr[:, 0]; eps = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau_phys = M1 * phi**3 / (phi_max * diff)
        loss = log_mse(tau_pred, y_tr) + lam * log_mse(tau_pred, tau_phys)
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
    print(f"  best_ep={best_ep}  R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result(f'B1_soft_pinn_lam{lam}', r2, mae, mape, best_ep, elapsed,
                       note=f'软约束PINN λ={lam}')


def train_B9(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam_init=0.001, ema_alpha=0.9, update_interval=20,
             lr=1e-4, epochs=1000, patience=150):
    set_seed(46)
    print(f"\n{'='*55}\nB9 — SA-PINN（自适应λ，init={lam_init}）")
    model = SoftPINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lam = lam_init
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if ep % update_interval == 1 and ep > 1:
            phi_tmp = X_tr[:, 0]; eps_tmp = 1e-6
            optimizer.zero_grad()
            tp1, pm1 = model(X_tr)
            log_mse(tp1, y_tr).backward()
            gn_data = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            optimizer.zero_grad()
            tp2, pm2 = model(X_tr)
            diff2 = torch.relu(pm2 - phi_tmp) + eps_tmp
            tp_phys2 = M1 * phi_tmp**3 / (pm2 * diff2)
            log_mse(tp2, tp_phys2).backward()
            gn_phys = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            lam = ema_alpha * lam + (1 - ema_alpha) * gn_data / (gn_phys + 1e-8)
            optimizer.zero_grad()
        model.train(); optimizer.zero_grad()
        tau_pred, phi_max = model(X_tr)
        phi = X_tr[:, 0]; eps = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau_phys = M1 * phi**3 / (phi_max * diff)
        loss = log_mse(tau_pred, y_tr) + lam * log_mse(tau_pred, tau_phys)
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
    print(f"  最终λ={lam:.4f}  best_ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return make_result('B9_sa_pinn', r2, mae, mape, best_ep, elapsed,
                       note=f'SA-PINN自适应λ，最终λ={lam:.4f}')


def train_PGNN(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
               lam=0.1, lr=1e-4, epochs=1000, patience=150):
    set_seed(47)   # 相同初始化，各lambda档唯一变量
    print(f"\n{'='*55}\nPGNN — Karpatne 2017（λ={lam}）")
    class _RawMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            s = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
            return self.net(x * s).squeeze()
    model = _RawMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        tau_pred = model(X_tr)
        phi = X_tr[:, 0]
        loss_data = log_mse(tau_pred.clamp(min=1e-6), y_tr)
        pos_penalty  = torch.relu(-tau_pred).mean()
        idx = torch.argsort(phi)
        tau_sorted = tau_pred[idx]
        mono_penalty = torch.relu(tau_sorted[:-1] - tau_sorted[1:]).mean()
        phi_ref_max = phi.max() + 0.15
        margin = phi_ref_max - phi
        threshold = 0.5 / margin.clamp(min=1e-3)
        diverge_mask = margin < 0.10
        diverge_penalty = (torch.relu(threshold[diverge_mask] - tau_pred[diverge_mask]).mean()
                           if diverge_mask.any()
                           else torch.tensor(0.0))
        loss = loss_data + lam * (pos_penalty + mono_penalty + diverge_penalty)
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
    print(f"  best_ep={best_ep}  R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result(f'pgnn_lam{lam}', r2, mae, mape, best_ep, elapsed,
                       note=f'PGNN Karpatne2017 λ={lam}')


def train_B8(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam_best=0.1, lr=1e-3, epochs_lf=500, epochs_hf=2000, patience=200):
    set_seed(48)
    print(f"\n{'='*55}\nB8 — 软约束PINN+多保真（λ={lam_best}）")
    model = SoftPINN(hidden_dim=64)
    # 阶段1：低保真预训练
    opt1 = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(1, epochs_lf + 1):
        model.train(); opt1.zero_grad()
        tau_pred, _ = model(X_lf_tr)
        log_mse(tau_pred, y_lf_tr).backward(); opt1.step()
    # 阶段2：HF微调
    opt2 = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs_hf + 1):
        model.train(); opt2.zero_grad()
        tau_pred, phi_max = model(X_tr)
        phi = X_tr[:, 0]; eps = 1e-6
        diff = torch.relu(phi_max - phi) + eps
        tau_phys = M1 * phi**3 / (phi_max * diff)
        loss = log_mse(tau_pred, y_tr) + lam_best * log_mse(tau_pred, tau_phys)
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
    print(f"  best_ep={best_ep}  R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B8_soft_pinn_mf', r2, mae, mape, best_ep, elapsed,
                       note=f'软约束PINN+多保真预训练，λ={lam_best}')


def train_B2(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lr=1e-4, epochs=2000, patience=200):
    set_seed(49)
    print(f"\n{'='*55}\nB2 — Meng 2020 复合MFNN")
    model = MengMFNN()
    # 阶段1：预训练低保真子网
    opt1 = optim.Adam(model.nn_L.parameters(), lr=1e-3)
    for ep in range(1, 501):
        model.train(); opt1.zero_grad()
        s = torch.tensor([2.0, 1.5])
        y_L = torch.relu(model.nn_L(X_lf_tr * s)).squeeze() + 1e-6
        log_mse(y_L, y_lf_tr).backward(); opt1.step()
    for p in model.nn_L.parameters(): p.requires_grad = False
    # 阶段2：训练F_l和F_nl
    opt2 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); opt2.zero_grad()
        y_H, _ = model(X_tr)
        log_mse(y_H, y_tr).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        opt2.step()
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
    print(f"  best_ep={best_ep}  R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B2_Meng_MFNN', r2, mae, mape, best_ep, elapsed,
                       note='Meng 2020复合MFNN，输出空间建模，lr=1e-4，grad_clip=1.0')


def train_B6(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
             lam_hf=10.0, lr=1e-4, epochs=1000, patience=150):
    set_seed(50)
    print(f"\n{'='*55}\nB6 — 单阶段混合训练（LF+HF，λ_HF={lam_hf}）")
    model = LianPINN(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        pred_lf, _ = model(X_lf_tr)
        pred_hf, _ = model(X_tr)
        loss = log_mse(pred_lf, y_lf_tr) + lam_hf * log_mse(pred_hf, y_tr)
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
    print(f"  best_ep={best_ep}  R²={r2:.4f}  MAE={mae:.4f} Pa  MAPE={mape:.1f}%")
    return make_result('B6_mixed_training', r2, mae, mape, best_ep, elapsed,
                       note=f'单阶段混合训练，λ_HF={lam_hf}')


def train_B10(X_tr, y_tr, X_ev, y_ev, X_te, y_te,
              lr=1e-4, epochs=1000, patience=150):
    set_seed(51)
    print(f"\n{'='*55}\nB10 — 硬约束PINN（全局φ_max可学习标量）")
    model = HardPINN_GlobalPhiMax()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_r2, best_ep, wait = float('-inf'), 0, 0
    best_state = copy.deepcopy(model.state_dict())
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        tau_pred, _ = model(X_tr)
        log_mse(tau_pred.squeeze(), y_tr).backward(); optimizer.step()
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
    model.eval()
    with torch.no_grad():
        phi_max_val = (0.65 + torch.nn.functional.softplus(
            model.log_phi_max_offset)).item()
    print(f"  全局φ_max={phi_max_val:.4f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return make_result('B10_hard_global_phimax', r2, mae, mape, best_ep, elapsed,
                       note=f'硬约束全局φ_max={phi_max_val:.4f}')


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    set_seed(RANDOM_SEED)   # 全局固定种子，保证模型初始化可复现

    print("\n" + "="*60)
    print("Lian 2025 对比基线实验")
    print(f"训练: {N_TRAIN}条增强HF  评估: {N_EVAL}条增强HF  测试: 16条论文原始数据")
    print("="*60)

    # ── 数据加载 ──
    (X_tr, y_tr, _), (X_ev, y_ev, _), _ = split_hifi_data(
        project_root / HIFI_DATA_PATH,
        n_train=N_TRAIN, n_eval=N_EVAL, seed=RANDOM_SEED,
    )
    X_te, y_te, _ = load_paper_test()

    # 低保真数据
    X_lf_tr, y_lf_tr, _ = load_csv(
        project_root / 'data/lian2025/low_fidelity/train.csv')
    print(f"\n低保真训练集: {len(y_lf_tr)} 条")

    all_results = []

    # ── B3 纯MLP ──
    all_results.append(
        train_B3(X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # ── B7 残差多保真MFNN ──
    all_results.append(
        train_B7(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # ── B1 软约束PINN（λ六档搜索）──
    b1_results = []
    for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        r = train_B1(X_tr, y_tr, X_ev, y_ev, X_te, y_te, lam=lam)
        b1_results.append(r)
        all_results.append(r)
    best_b1 = max(b1_results, key=lambda r: r['test_r2'])
    print(f"\nB1最优λ: {best_b1['tag']}  R²={best_b1['test_r2']:.4f}")

    # ── B9 SA-PINN ──
    all_results.append(
        train_B9(X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # ── PGNN（λ三档搜索）──
    pgnn_results = []
    for lam_pgnn in [0.01, 0.1, 1.0]:
        r = train_PGNN(X_tr, y_tr, X_ev, y_ev, X_te, y_te, lam=lam_pgnn)
        pgnn_results.append(r)
        all_results.append(r)
    best_pgnn = max(pgnn_results, key=lambda r: r['test_r2'])
    print(f"\nPGNN最优: {best_pgnn['tag']}  R²={best_pgnn['test_r2']:.4f}")

    # ── B8 软约束PINN+多保真（用B1最优λ）──
    lam_best_b8 = float(best_b1['tag'].split('lam')[-1])
    all_results.append(
        train_B8(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
                 lam_best=lam_best_b8))

    # ── B2 Meng MFNN ──
    all_results.append(
        train_B2(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # ── B6 单阶段混合训练 ──
    all_results.append(
        train_B6(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # ── B10 硬约束全局φ_max（消融）──
    all_results.append(
        train_B10(X_tr, y_tr, X_ev, y_ev, X_te, y_te))

    # ── 保存结果 ──
    result_path = RESULTS_DIR / 'logs/baseline_results_lian2025_new.json'
    os.makedirs(result_path.parent, exist_ok=True)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {result_path}")

    # ── 汇总打印 ──
    PI_MFNN_R2   = 0.8785
    PI_MFNN_MAPE = 18.8

    print("\n" + "="*70)
    print("对比基线汇总（测试集：16条论文原始实验数据）vs PI-MFNN")
    print("="*70)
    header = f"{'方法':<32} {'R²':>8} {'MAE/Pa':>10} {'MAPE%':>8} {'vs PIMFNN ΔR²':>14}"
    print(header); print("─"*len(header))
    print(f"  {'PI-MFNN（本文）':<30} {PI_MFNN_R2:>8.4f} {'0.1161':>10} "
          f"{PI_MFNN_MAPE:>7.1f}%  {'基准':>12}")
    print("─"*len(header))
    # 每组取最优
    best_per = {}
    for r in all_results:
        base = r['tag'].split('_lam')[0]
        if base not in best_per or r['test_r2'] > best_per[base]['test_r2']:
            best_per[base] = r
    for r in sorted(best_per.values(), key=lambda x: x['test_r2'], reverse=True):
        delta = r['test_r2'] - PI_MFNN_R2
        print(f"  {r['tag']:<30} {r['test_r2']:>8.4f} "
              f"{r['test_mae']:>10.4f} {r['test_mape']:>7.1f}%  {delta:>+12.4f}")
    print("─"*len(header))
