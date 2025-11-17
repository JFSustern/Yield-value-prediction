import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- 1  机理参数 ----------
R     = 8.314
Ea    = 60e3          # J/mol
A     = 1e8           # arb. units
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
    network = (X / (1 - X))**2                      # 颗粒网络强度 ∝ (Φ/Φ_m) 幂律

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_tensor = torch.from_numpy(X).to(device)
y_tensor = torch.from_numpy(y).to(device)

net = nn.Sequential(
        nn.Linear(4, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1)).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

loader = DataLoader(TensorDataset(X_tensor, y_tensor),
                    batch_size=256, shuffle=True)

# ---------- 6  训练 ----------
for epoch in range(400):
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(net(xb), yb)
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f'Epoch {epoch:3d}  loss={loss.item():.4f}')

# ---------- 7  预测 ----------
net.eval()
with torch.no_grad():
    y_pred = net(X_tensor).cpu().numpy()

print('生成样本 R² =', 1 - np.var(y_pred-y)/np.var(y))