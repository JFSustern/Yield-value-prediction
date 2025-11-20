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
    # 3.1 有效最大堆积（混合功）
    Phi_eff = Phi_m * (1 - np.exp(-E / 400))  # 同原逻辑
    X = np.clip(Phi / Phi_eff, 1e-3, 0.99)  # 防边界

    # 3.1.1 严格 KD 相对粘度（无量纲）
    eta = 2.5  # 球形颗粒本征粘度
    eta_r = (1 - X) ** (-eta * Phi_m)  # KD 核心式

    # 3.1.2 屈服应力 ∝ (η_r - 1)^p   （文献幂律标度）
    p = 0.8  # 幂次可先验 0.6-1.0
    C = 30.0  # Pa  经验标尺（可调）
    tau0_base = C * (eta_r - 1) ** p  # 单位：Pa

    # 3.2 Arrhenius 固化增强
    k = A * np.exp(-Ea / (R * T))
    alpha = 1 - np.exp(-k * t)
    chem_boost = 1 + 5 * alpha  # 化学交联倍数

    # 3.3 最终屈服应力
    tau0 = tau0_base * chem_boost
    return tau0, alpha

tau0, alpha = tau0_mechanism(T, Phi, E, t)

# ---------- 4  构造 DataFrame ----------
df = pd.DataFrame({'T':T, 'Phi':Phi, 'E':E, 't':t, 'tau0':tau0,
                   '1/T':1/T, 'X':Phi/Phi_m, 'logt':np.log1p(t),
                   'alpha':alpha})

print("="*80)
print("数据生成完成")
print("="*80)
print(f"样本数量: {N}")
print(f"\n输入变量范围:")
print(f"  温度 T:        [{T.min()-273.15:.1f}, {T.max()-273.15:.1f}] °C  (绝对温度: [{T.min():.1f}, {T.max():.1f}] K)")
print(f"  体积分数 Φ:    [{Phi.min():.3f}, {Phi.max():.3f}]")
print(f"  混合功 E:      [{E.min():.1f}, {E.max():.1f}] kJ")
print(f"  静置时间 t:    [{t.min():.1f}, {t.max():.1f}] s  ({t.min()/3600:.2f}, {t.max()/3600:.2f} h)")
print(f"\n输出变量范围:")
print(f"  屈服应力 τ₀:  [{tau0.min():.2f}, {tau0.max():.2f}] Pa")
print(f"  固化度 α:      [{alpha.min():.4f}, {alpha.max():.4f}]")

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
# 每5个样本中，前4个给训练集，最后1个给验证集
train_idx = []
val_idx = []
for i in range(0, len(sorted_idx), 5):
    train_idx.extend(sorted_idx[i:i+4])  # 前4个
    if i+4 < len(sorted_idx):
        val_idx.append(sorted_idx[i+4])  # 第5个
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)

X_train_tensor = X_tensor[train_idx]
y_train_tensor = y_tensor[train_idx]
X_val_tensor = X_tensor[val_idx]
y_val_tensor = y_tensor[val_idx]

print("\n" + "="*80)
print("数据预处理完成")
print("="*80)
print(f"训练集样本数: {len(train_idx)}")
print(f"验证集样本数: {len(val_idx)}")

net = nn.Sequential(
        nn.Linear(4, 32), nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(16, 1)).to(device)

print("\n" + "="*80)
print("神经网络架构")
print("="*80)
print(net)
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)

loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                    batch_size=256, shuffle=True)

# ---------- 6  训练（带 Early Stopping） ----------
print("\n" + "="*80)
print("开始训练")
print("="*80)
print(f"最大训练轮数: 4000")
print(f"批次大小: 256")
print(f"学习率: 1e-3")
print(f"权重衰减: 5e-4")
print(f"Early Stopping 轮数: 50")
print("="*80)

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

# 计算 RMSE 和 MAE
rmse_train = np.sqrt(np.mean((y_train - y_train_pred)**2))
mae_train = np.mean(np.abs(y_train - y_train_pred))
rmse_val = np.sqrt(np.mean((y_val - y_val_pred)**2))
mae_val = np.mean(np.abs(y_val - y_val_pred))
rmse_all = np.sqrt(np.mean((y - y_pred)**2))
mae_all = np.mean(np.abs(y - y_pred))

print("\n" + "="*80)
print("训练完成 - 模型性能评估")
print("="*80)
print(f"\n全样本性能:")
print(f"  R² = {r2:.6f}")
print(f"  RMSE = {rmse_all:.4f} Pa")
print(f"  MAE = {mae_all:.4f} Pa")
print(f"\n训练集性能:")
print(f"  R² = {r2_train:.6f}")
print(f"  RMSE = {rmse_train:.4f} Pa")
print(f"  MAE = {mae_train:.4f} Pa")
print(f"\n验证集性能:")
print(f"  R² = {r2_val:.6f}")
print(f"  RMSE = {rmse_val:.4f} Pa")
print(f"  MAE = {mae_val:.4f} Pa")
print(f"\n过拟合检查:")
print(f"  训练集与验证集 R² 差异: {abs(r2_train - r2_val):.6f}")
if abs(r2_train - r2_val) < 0.05:
    print(f"  ✓ 模型泛化良好")
elif abs(r2_train - r2_val) < 0.1:
    print(f"  ⚠ 存在轻微过拟合")
else:
    print(f"  ✗ 存在明显过拟合")
