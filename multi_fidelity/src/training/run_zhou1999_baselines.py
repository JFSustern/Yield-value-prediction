"""
Zhou 1999 体系多方法对比实验

对比维度与 Lian 2025 体系保持对称，验证架构优势的跨体系一致性。
共比较 9 种方法，组织为：
  - 无约束：纯MLP、残差多保真MFNN
  - 软约束：软约束PINN（λ六档搜索）、SA-PINN、软约束PINN+多保真
  - 其他：Meng复合MFNN、单阶段混合训练
  - 硬约束：PI-MFNN HF-only（策略C）、PI-MFNN多保真（策略D）

注：测试集 10 条（受 Zhou 1999 公开数据量限制），结果以定性参考为主。

运行：
  cd /path/to/Project
  python -m multi_fidelity.src.training.run_zhou1999_baselines \\
      --lf-data data/zhou1999/low_fidelity \\
      --hf-data data/zhou1999/high_fidelity \\
      --out-dir multi_fidelity/results/zhou1999
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
import torch.nn as nn
import torch.optim as optim

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from multi_fidelity.src.model.pinn_zhou1999 import ZhouPINN   # noqa

# ── 常量 ──────────────────────────────────────────────────────────────────

FEATURES = ['phi', 'd_s_um']
TARGET   = 'tau_Pa'
SEED     = 42

PHI_MAX_REF = ZhouPINN.PHI_MAX_REF   # 0.570 固定
PHI_0       = ZhouPINN.PHI_0          # 0.026 固定
M1_LO       = ZhouPINN.M1_LO         # 50 Pa
M1_HI       = ZhouPINN.M1_HI         # 3000 Pa
K_H_REF     = 65.0    # Pa·μm²，LF 标定常数

# ── 工具 ──────────────────────────────────────────────────────────────────

def set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)


def log_mse(pred, target):
    return torch.mean((torch.log1p(pred) - torch.log1p(target)) ** 2)


def metrics(pred, y):
    """pred, y: 1-D tensors"""
    with torch.no_grad():
        r2   = (1 - torch.sum((y - pred)**2) / torch.sum((y - y.mean())**2)).item()
        mae  = torch.mean(torch.abs(pred - y)).item()
        mape = (torch.mean(torch.abs((pred - y) / (y + 1e-6))) * 100).item()
    return r2, mae, mape


def eval_model(model, X, y):
    model.eval()
    with torch.no_grad():
        out  = model(X)
        pred = out[0].squeeze() if isinstance(out, tuple) else out.squeeze()
    r2, mae, mape = metrics(pred, y)
    return r2, mae, mape, pred.cpu().numpy()


def load_tensors(path):
    df = pd.read_csv(path)
    X  = torch.tensor(df[FEATURES].values, dtype=torch.float32)
    y  = torch.tensor(df[TARGET].values,   dtype=torch.float32)
    return X, y


def simple_train(model, X_tr, y_tr, X_ev, y_ev,
                 lr=1e-4, epochs=1000, patience=150, loss_fn=None):
    """通用训练循环（按 eval R² 早停）。"""
    if loss_fn is None:
        loss_fn = log_mse
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(X_tr)
        pred = out[0].squeeze() if isinstance(out, tuple) else out.squeeze()
        loss_fn(pred, y_tr).backward()
        optimizer.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return best_ep, time.time() - t0


def yodel_pcl(phi, m1_eff, phi_max=PHI_MAX_REF, phi0=PHI_0, eps=1e-6):
    """YODEL PCL (可微，用于软约束残差)。"""
    num   = m1_eff * phi * torch.clamp(phi - phi0, min=eps) ** 2
    denom = phi_max * torch.clamp(phi_max - phi, min=eps)
    return num / denom

# ── 方法定义 ──────────────────────────────────────────────────────────────

class MLP_Zhou(nn.Module):
    """纯 MLP（无物理约束），[φ, d_s] → τ 直接预测。"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.scale = torch.tensor([2.5, 3.0])

    def forward(self, x):
        raw  = self.net(x * self.scale.to(x.device)).squeeze()
        pred = torch.relu(raw) + 1e-6
        return pred,


class SoftPINN_Zhou(nn.Module):
    """
    软约束 PINN：网络同时输出 [τ_pred, raw_m1]；
    损失 = data_loss + λ × YODEL_residual²
    其中 residual = τ_pred - YODEL(φ, m1_decoded)。
    这是本文硬约束架构的软约束对照版本。
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),   # [raw_tau, raw_m1]
        )
        self.scale = torch.tensor([2.5, 3.0])

    def forward(self, x):
        phi  = x[:, 0]
        raw  = self.net(x * self.scale.to(x.device))
        tau_pred = torch.relu(raw[:, 0]) + 1e-6
        m1_eff   = M1_LO + torch.sigmoid(raw[:, 1]) * (M1_HI - M1_LO)
        tau_phys = yodel_pcl(phi, m1_eff)
        return tau_pred, m1_eff, tau_phys


class ResidualMFNN_Zhou(nn.Module):
    """
    残差多保真MFNN (Forrester/Kennedy 风格)：
    NN_LF(φ, d_s) 预训练低保真 τ；
    NN_HF(φ, d_s) 学习 (τ_HF - τ_LF) 残差。
    最终：τ_HF_pred = τ_LF_frozen + δ
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.scale = torch.tensor([2.5, 3.0])
        self.net_lf = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.net_delta = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward_lf(self, x):
        s = self.scale.to(x.device)
        return torch.relu(self.net_lf(x * s)).squeeze() + 1e-6

    def forward(self, x):
        s     = self.scale.to(x.device)
        tau_L = torch.relu(self.net_lf(x * s)).squeeze() + 1e-6
        delta = self.net_delta(x * s).squeeze()
        return torch.relu(tau_L + delta) + 1e-6,


class MengMFNN_Zhou(nn.Module):
    """Meng & Karniadakis (2020) 复合多保真神经网络。"""
    def __init__(self):
        super().__init__()
        self.scale = torch.tensor([2.5, 3.0])
        self.nn_L  = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 1))
        self.F_l   = nn.Sequential(
            nn.Linear(3, 10), nn.Tanh(), nn.Linear(10, 1))
        self.F_nl  = nn.Sequential(
            nn.Linear(3, 10), nn.Tanh(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1))

    def forward(self, x):
        s   = self.scale.to(x.device)
        y_L = torch.relu(self.nn_L(x * s)).clamp(min=1e-6)
        inp = torch.cat([x * s, y_L], dim=1)
        y_H = self.F_l(inp) + self.F_nl(inp)
        return torch.relu(y_H).squeeze() + 1e-6,

    def forward_lf(self, x):
        s = self.scale.to(x.device)
        return torch.relu(self.nn_L(x * s)).squeeze() + 1e-6


# ── 各方法训练函数 ────────────────────────────────────────────────────────

def run_mlp(X_tr, y_tr, X_ev, y_ev, X_te, y_te):
    """纯 MLP（无物理约束）。"""
    set_seed()
    model = MLP_Zhou()
    best_ep, _ = simple_train(model, X_tr, y_tr, X_ev, y_ev, lr=1e-3)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  纯MLP          | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='纯MLP', constraint='无约束', mf=False,
                R2=r2, MAE=mae, MAPE=mape)


def run_residual_mfnn(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te):
    """残差多保真MFNN（Forrester 风格）。"""
    set_seed()
    model = ResidualMFNN_Zhou()
    # 阶段1：预训练 LF 子网
    opt1 = optim.Adam(model.net_lf.parameters(), lr=1e-3)
    for ep in range(400):
        model.train(); opt1.zero_grad()
        pred_lf = model.forward_lf(X_lf_tr)
        log_mse(pred_lf, y_lf_tr).backward(); opt1.step()
    for p in model.net_lf.parameters(): p.requires_grad = False
    # 阶段2：训练残差子网
    opt2 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    for ep in range(1, 1001):
        model.train(); opt2.zero_grad()
        pred, = model(X_tr)
        log_mse(pred, y_tr).backward(); opt2.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(best_state)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  残差MFNN       | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='残差多保真MFNN†', constraint='无约束', mf=True,
                R2=r2, MAE=mae, MAPE=mape)


def run_soft_pinn(X_tr, y_tr, X_ev, y_ev, X_te, y_te, lam=0.1):
    """软约束PINN（固定 λ）。"""
    set_seed()
    model = SoftPINN_Zhou()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    for ep in range(1, 1001):
        model.train(); optimizer.zero_grad()
        tau_pred, _, tau_phys = model(X_tr)
        loss = log_mse(tau_pred, y_tr) + lam * log_mse(tau_pred, tau_phys)
        loss.backward(); optimizer.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(best_state)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  软约束PINN λ={lam} | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label=f'软约束PINN(λ={lam})', constraint='软约束', mf=False,
                R2=r2, MAE=mae, MAPE=mape, lam=lam)


def run_soft_pinn_search(X_tr, y_tr, X_ev, y_ev, X_te, y_te):
    """软约束PINN λ 六档搜索，返回最优。"""
    best, best_res = -1.0, None
    for lam in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        res = run_soft_pinn(X_tr, y_tr, X_ev, y_ev, X_te, y_te, lam=lam)
        if res['R2'] > best:
            best, best_res = res['R2'], res
    best_res['label'] = f"软约束PINN(最优λ={best_res['lam']})"
    print(f"  → 最优λ={best_res['lam']}  R²={best_res['R2']:.4f}")
    return best_res


def run_sa_pinn(X_tr, y_tr, X_ev, y_ev, X_te, y_te):
    """SA-PINN：自适应 λ（Wang 2021 梯度比值权重）。"""
    set_seed()
    model    = SoftPINN_Zhou()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    for ep in range(1, 1001):
        model.train(); optimizer.zero_grad()
        tau_pred, _, tau_phys = model(X_tr)
        L_d = log_mse(tau_pred, y_tr)
        L_p = log_mse(tau_pred, tau_phys)
        # 自适应权重：λ = mean(|∇L_d|) / mean(|∇L_p|)
        g_d = torch.autograd.grad(L_d, model.parameters(),
                                  create_graph=False, retain_graph=True,
                                  allow_unused=True)
        g_p = torch.autograd.grad(L_p, model.parameters(),
                                  create_graph=False, retain_graph=True,
                                  allow_unused=True)
        norm_d = sum(g.abs().mean() for g in g_d if g is not None).item()
        norm_p = sum(g.abs().mean() for g in g_p if g is not None).item() + 1e-10
        lam_val = norm_d / norm_p
        optimizer.zero_grad()
        (L_d + lam_val * L_p).backward()
        optimizer.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(best_state)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  SA-PINN        | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='SA-PINN(自适应λ)', constraint='软约束', mf=False,
                R2=r2, MAE=mae, MAPE=mape)


def run_soft_pinn_mf(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te,
                     lam=0.1):
    """软约束PINN + 多保真（先 LF 预训练，再软约束 HF 微调）。"""
    set_seed()
    model    = SoftPINN_Zhou()
    # LF 预训练（只用数据损失）
    opt1 = optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(400):
        model.train(); opt1.zero_grad()
        tau_lf, *_ = model(X_lf_tr)
        log_mse(tau_lf, y_lf_tr).backward(); opt1.step()
    # HF 微调（软约束损失）
    opt2 = optim.Adam(model.parameters(), lr=1e-4)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    for ep in range(1, 1001):
        model.train(); opt2.zero_grad()
        tau_pred, _, tau_phys = model(X_tr)
        (log_mse(tau_pred, y_tr) + lam * log_mse(tau_pred, tau_phys)).backward()
        opt2.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(best_state)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  软约束PINN+MF† | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='软约束PINN+多保真†', constraint='软约束', mf=True,
                R2=r2, MAE=mae, MAPE=mape)


def run_meng_mfnn(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te):
    """Meng 复合MFNN。"""
    set_seed()
    model = MengMFNN_Zhou()
    # 阶段1：预训练 NN_L
    opt1 = optim.Adam(model.nn_L.parameters(), lr=1e-3)
    for ep in range(400):
        model.train(); opt1.zero_grad()
        y_L = model.forward_lf(X_lf_tr)
        log_mse(y_L, y_lf_tr).backward(); opt1.step()
    for p in model.nn_L.parameters(): p.requires_grad = False
    # 阶段2：训练 F_l + F_nl
    opt2 = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    for ep in range(1, 1501):
        model.train(); opt2.zero_grad()
        pred, = model(X_tr)
        log_mse(pred, y_tr).backward(); opt2.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(best_state)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  Meng MFNN†     | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='Meng复合MFNN†', constraint='其他', mf=True,
                R2=r2, MAE=mae, MAPE=mape)


def run_single_stage(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te):
    """单阶段混合训练（LF+HF 同时训练，非两阶段）。"""
    set_seed()
    model = ZhouPINN()
    X_mix = torch.cat([X_lf_tr[:len(X_tr)], X_tr], dim=0)
    y_mix = torch.cat([y_lf_tr[:len(X_tr)], y_tr], dim=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    for ep in range(1, 1001):
        model.train(); optimizer.zero_grad()
        pred, _ = model(X_mix)
        log_mse(pred, y_mix).backward(); optimizer.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(best_state)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  单阶段混合†    | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='单阶段混合训练†', constraint='其他', mf=True,
                R2=r2, MAE=mae, MAPE=mape)


def run_hard_hf_only(X_tr, y_tr, X_ev, y_ev, X_te, y_te):
    """硬约束，纯高保真（ZhouPINN，随机初始化）。"""
    set_seed()
    model = ZhouPINN()
    best_ep, _ = simple_train(model, X_tr, y_tr, X_ev, y_ev, lr=1e-4)
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  硬约束 HF-only | ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='PI-MFNN HF-only（本文硬约束，无MF）', constraint='硬约束', mf=False,
                R2=r2, MAE=mae, MAPE=mape)


def run_hard_mf(X_lf_tr, y_lf_tr, X_lf_te, y_lf_te,
                X_tr, y_tr, X_ev, y_ev, X_te, y_te, freeze_n=1):
    """硬约束，多保真融合（PI-MFNN，本文方法）。"""
    set_seed()
    model = ZhouPINN()
    # LF 预训练
    optimizer_lf = optim.Adam(model.parameters(), lr=1e-3)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_lf, mode='min', factor=0.5, patience=10)
    best_r2_lf, best_state_lf, wait = float('-inf'), None, 0
    for ep in range(1, 501):
        model.train()
        idx = torch.randperm(len(X_lf_tr))
        for i in range(0, len(X_lf_tr), 64):
            b = idx[i:i+64]
            optimizer_lf.zero_grad()
            pred, _ = model(X_lf_tr[b])
            log_mse(pred, y_lf_tr[b]).backward()
            optimizer_lf.step()
        ev_r2_lf, *_ = eval_model(model, X_lf_te, y_lf_te)
        scheduler.step(1 - ev_r2_lf)
        if ev_r2_lf > best_r2_lf:
            best_r2_lf, best_state_lf, wait = ev_r2_lf, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= 50: break
    model.load_state_dict(best_state_lf)
    print(f"    LF预训练 R²={best_r2_lf:.4f}")
    # HF 微调
    linear_layers = [m for m in model.net if hasattr(m, 'weight')]
    for layer in linear_layers[:freeze_n]:
        for p in layer.parameters(): p.requires_grad = False
    optimizer_hf = optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=1e-4, weight_decay=1e-5)
    best_r2, best_ep, wait, best_state = float('-inf'), 0, 0, None
    for ep in range(1, 1001):
        model.train(); optimizer_hf.zero_grad()
        pred, _ = model(X_tr)
        log_mse(pred, y_tr).backward(); optimizer_hf.step()
        ev_r2, *_ = eval_model(model, X_ev, y_ev)
        if ev_r2 > best_r2:
            best_r2, best_ep, wait = ev_r2, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= 150: break
    model.load_state_dict(best_state)
    for p in model.parameters(): p.requires_grad = True
    r2, mae, mape, _ = eval_model(model, X_te, y_te)
    print(f"  PI-MFNN (本文)†| ep={best_ep}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return dict(label='PI-MFNN（本文）†', constraint='硬约束', mf=True,
                R2=r2, MAE=mae, MAPE=mape)


# ── 排名图 ────────────────────────────────────────────────────────────────

def _ranking_label(result):
    """Use ASCII/English labels so plots render without platform-specific CJK fonts."""
    label = result['label']
    if label.startswith('软约束PINN(最优'):
        return f"Soft PINN (best lambda={result.get('lam')})"

    labels = {
        '纯MLP': 'Pure MLP',
        '残差多保真MFNN†': 'Residual MFNN',
        'SA-PINN(自适应λ)': 'SA-PINN',
        '软约束PINN+多保真†': 'Soft PINN + MF',
        'Meng复合MFNN†': 'Meng MFNN',
        '单阶段混合训练†': 'Single-stage mixed',
        'PI-MFNN HF-only（本文硬约束，无MF）': 'PI-MFNN HF-only',
        'PI-MFNN（本文）†': 'PI-MFNN',
    }
    return labels.get(label, label)


def plot_ranking(results, out_path):
    methods = [_ranking_label(r) for r in results]
    r2s     = [r['R2'] for r in results]
    colors  = []
    for r in results:
        if r['constraint'] == '硬约束':
            colors.append('#245F85' if r['mf'] else '#89B8D8')
        elif r['constraint'] == '软约束':
            colors.append('#D56B5D' if r['mf'] else '#E8B66B')
        else:
            colors.append('#BFC9D9')

    order = sorted(range(len(r2s)), key=lambda i: r2s[i])
    methods = [methods[i] for i in order]
    r2s     = [r2s[i]     for i in order]
    colors  = [colors[i]  for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.barh(range(len(methods)), r2s, color=colors,
                   edgecolor='white', linewidth=0.4, height=0.65)
    for bar, val in zip(bars, r2s):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=6)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=6.5)
    ax.set_xlabel('Test R²', fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.set_title('Zhou 1999 Al₂O₃: Multi-Method R² Ranking\n'
                 '(test n=10, 30 real HF training points)', fontsize=7.5)
    ax.axvline(0.9, color='gray', lw=0.6, ls='--', alpha=0.6)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'排名图保存至 {out_path}')


# ── 主程序 ────────────────────────────────────────────────────────────────

def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    X_lf_tr, y_lf_tr = load_tensors(Path(args.lf_data) / 'train.csv')
    X_lf_te, y_lf_te = load_tensors(Path(args.lf_data) / 'test.csv')
    X_tr,    y_tr    = load_tensors(Path(args.hf_data) / 'train.csv')
    X_ev,    y_ev    = load_tensors(Path(args.hf_data) / 'eval.csv')
    X_te,    y_te    = load_tensors(Path(args.hf_data) / 'test.csv')

    print(f'数据摘要: LF 训练={len(X_lf_tr)} HF 训练={len(X_tr)} 评估={len(X_ev)} 测试={len(X_te)}')
    if len(X_te) < 20:
        print(f'⚠️  测试集仅 {len(X_te)} 条，数值结果以定性参考为主')
    print()

    print('='*60)
    print('Zhou 1999 Al₂O₃ 多方法对比实验')
    print('='*60)

    t0 = time.time()
    results = [
        run_mlp(X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_residual_mfnn(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_soft_pinn_search(X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_sa_pinn(X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_soft_pinn_mf(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_meng_mfnn(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_single_stage(X_lf_tr, y_lf_tr, X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_hard_hf_only(X_tr, y_tr, X_ev, y_ev, X_te, y_te),
        run_hard_mf(X_lf_tr, y_lf_tr, X_lf_te, y_lf_te,
                    X_tr, y_tr, X_ev, y_ev, X_te, y_te,
                    freeze_n=args.freeze_n),
    ]
    elapsed = time.time() - t0

    print(f'\n{"="*65}')
    print(f'{"方法":<32} {"R²":>7} {"MAE/Pa":>8} {"MAPE/%":>8}')
    print(f'{"─"*65}')
    for r in results:
        print(f'{r["label"][:35]:<35} {r["R2"]:>7.4f} {r["MAE"]:>8.1f} {r["MAPE"]:>8.1f}')
    print(f'{"─"*65}')
    print(f'总耗时: {elapsed:.0f} s')
    print(f'† 额外使用低保真合成数据')

    summary = dict(system='Zhou1999_Al2O3_baselines', test_n=len(X_te),
                   elapsed_s=round(elapsed, 1),
                   results=[{k: v for k, v in r.items() if k != 'model'}
                             for r in results])
    with open(out_dir / 'baselines_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n结果保存至 {out_dir / "baselines_summary.json"}')

    plot_ranking(results, out_dir / 'baselines_ranking.png')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lf-data',  default='data/zhou1999/low_fidelity')
    parser.add_argument('--hf-data',  default='data/zhou1999/high_fidelity')
    parser.add_argument('--out-dir',  default='multi_fidelity/results/zhou1999')
    parser.add_argument('--freeze-n', type=int, default=1)
    args = parser.parse_args()
    os.chdir(project_root)
    main(args)
