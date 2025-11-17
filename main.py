import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- 1  机理参数 ----------
R     = 8.314
Ea    = 60e3          # J/mol
A     = 1e5           # arb. units (调整为更合理的值)
Phi_m = 0.68          # 最大堆积
eta_liquid_0 = 1.0    # Pa·s @ 25°C
T_ref = 25 + 273.15   # K

# ---------- 2  工况网格 ----------
N = 5000
np.random.seed(42)
T  = np.random.uniform(25, 65, N) + 273.15        # K
Phi= np.random.uniform(0.50, 0.66, N)             # 体积分数
E  = np.random.uniform(100, 800, N)               # kJ 混合功
t  = np.random.uniform(0, 3*3600, N)              # s  静置时间

# ---------- 3  机理生成函数 ----------
def tau0_mechanism(T, Phi, E, t):
    # 3.1 KD 有效堆积 → 网络密度
    Phi_eff = Phi_m * (1 - np.exp(-E/400))          # 混合越好 → 有效 Φ_m 越高
    X = Phi / Phi_eff
    # 修复：避免分母为 0，使用更稳定的公式 原来是network = (X/(1-X))**2
    network = np.clip(X, 0, 0.99)**2 / (1 - np.clip(X, 0, 0.99))**2  # 颗粒网络强度

    # 3.2 Arrhenius 固化增强
    k = A * np.exp(-Ea/(R*T))
    alpha = 1 - np.exp(-k*t)                        # 固化度 α∈[0,1)
    chem_boost = 1 + 5*alpha                        # 交联使 τ₀ 增强

    # 3.3 基准尺度
    tau0_base = 50 * network                          # Pa
    tau0 = tau0_base * chem_boost
    return tau0, alpha

tau0, alpha = tau0_mechanism(T, Phi, E, t)

# ---------- 4  构造 DataFrame ----------
df = pd.DataFrame({'T':T, 'Phi':Phi, 'E':E, 't':t, 'tau0':tau0,
                   '1/T':1/T, 'X':Phi/Phi_m, 'logt':np.log1p(t),
                   'alpha':alpha})
print(df.head())

# ---------- 5  神经网络 ----------
X = df[['1/T', 'X', 'E', 'logt']].values.astype(np.float32)
y = df['tau0'].values.astype(np.float32)[:,None]

# 数据标准化（关键！）
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / (X_std + 1e-8)

y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / (y_std + 1e-8)

device = torch.device("mps")
X_tensor = torch.from_numpy(X_normalized).to(device)
y_tensor = torch.from_numpy(y_normalized).to(device)

# 分割训练集和验证集（8:2）- 按 y 值分层采样
np.random.seed(42)
# 按 y 值排序后分割，确保训练集和验证集的分布相似
sorted_idx = np.argsort(y_normalized.flatten())
train_idx = sorted_idx[::5]  # 每5个中取1个作为验证集
val_idx = sorted_idx[1::5]   # 错开采样
# 如果数量不够，补充
if len(train_idx) < int(0.8*len(X_normalized)):
    remaining = sorted_idx[len(train_idx)+len(val_idx):]
    train_idx = np.concatenate([train_idx, remaining[:int(0.8*len(X_normalized))-len(train_idx)]])
if len(val_idx) < int(0.2*len(X_normalized)):
    remaining = sorted_idx[len(train_idx)+len(val_idx):]
    val_idx = np.concatenate([val_idx, remaining[:int(0.2*len(X_normalized))-len(val_idx)]])

X_train_tensor = X_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
X_val_tensor = X_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

net = nn.Sequential(
        nn.Linear(4, 32), nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(16, 1)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                    batch_size=256, shuffle=True)

# ---------- 6  训练（带 Early Stopping） ----------
best_val_loss = float('inf')
patience = 50
patience_counter = 0

for epoch in range(4000):
    # 训练阶段
    net.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(net(xb), yb)
        loss.backward()
        optimizer.step()

    # 验证阶段
    net.eval()
    with torch.no_grad():
        val_loss = criterion(net(X_val_tensor), y_val_tensor).item()

    if epoch % 50 == 0:
        print(f'Epoch {epoch:3d}  val_loss={val_loss:.4f}')

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        best_model_state = {k: v.cpu() for k, v in net.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            # 恢复最佳模型
            net.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            break

# ---------- 7  预测 ----------
net.eval()
with torch.no_grad():
    y_pred_normalized = net(X_tensor).cpu().numpy()

# 反标准化预测值
y_pred = y_pred_normalized * y_std + y_mean

# 计算 R²
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - y_mean)**2)
r2 = 1 - ss_res / ss_tot

print(f'全样本 R² = {r2:.6f}')

# 分别计算训练集和验证集的 R²
y_train_pred_normalized = net(X_train_tensor).cpu().detach().numpy()
y_train_pred = y_train_pred_normalized * y_std + y_mean
y_train = y[train_idx]
ss_res_train = np.sum((y_train - y_train_pred)**2)
ss_tot_train = np.sum((y_train - y_mean)**2)
r2_train = 1 - ss_res_train / ss_tot_train

y_val_pred_normalized = net(X_val_tensor).cpu().detach().numpy()
y_val_pred = y_val_pred_normalized * y_std + y_mean
y_val = y[val_idx]
ss_res_val = np.sum((y_val - y_val_pred)**2)
ss_tot_val = np.sum((y_val - y_mean)**2)
r2_val = 1 - ss_res_val / ss_tot_val

print(f'训练集 R² = {r2_train:.6f}')
print(f'验证集 R² = {r2_val:.6f}')
