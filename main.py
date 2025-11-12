import deepxde as dde
import numpy as np

# 设置 DeepXDE 使用 float32
dde.config.set_default_float("float32")

# 1. 机理生成（使用对数空间避免数值溢出）
def yield_WLF_log(T, edot):
    """计算 log10(σ_y)，避免数值溢出"""
    logaT = -8.86*(T-25)/(101.6+T-25)
    sig_ref, alpha, m = 2.1e6, 0.78, 0.11

    # 在对数空间计算: log10(σ) = log10(sig_ref) + alpha*logaT + m*log10(edot/1e-2)
    log_sig = np.log10(sig_ref) + alpha * logaT + m * np.log10(edot/1e-2)
    return log_sig

def yield_WLF(T, edot):
    """计算 σ_y [Pa]"""
    return 10 ** yield_WLF_log(T, edot)

# 生成训练数据（限制温度范围避免极端值）
np.random.seed(42)
T  = np.random.uniform(0, 60, 2000).astype(np.float32)  # 限制在 0-60°C
edot = 10**(np.random.uniform(-3, 1, 2000)).astype(np.float32)  # 1e-3 到 10 s⁻¹

# 使用对数空间
log_sig_y = yield_WLF_log(T, edot).astype(np.float32)
log_edot = np.log10(edot).astype(np.float32)

X = np.c_[T, log_edot].astype(np.float32)
y = log_sig_y.reshape(-1,1).astype(np.float32)  # 输出是 log10(σ_y)

print(f"输入范围: T=[{T.min():.1f}, {T.max():.1f}]°C, log10(edot)=[{log_edot.min():.2f}, {log_edot.max():.2f}]")
print(f"输出范围: log10(σ_y)=[{y.min():.2f}, {y.max():.2f}]")
print(f"对应 σ_y=[{10**y.min():.2e}, {10**y.max():.2e}] Pa")

# 2. PINN 定义物理损失（在对数空间）
def pde(x, u):
    """u 是预测的 log10(σ_y)"""
    T, log_edot = x[:,0:1], x[:,1:2]

    # WLF 公式在对数空间
    logaT = -8.86*(T-25)/(101.6+T-25)
    sig_ref, alpha, m = 2.1e6, 0.78, 0.11  #sig_ref(温度25℃，剪切率0.01 s⁻¹的屈服应力)

    # 使用 numpy 的 log10 常量值
    log10_sig_ref = 6.322219  # np.log10(基准屈服应力)
    log10_1e_minus_2 = -2.0   # np.log10(1e-2)

    log_sig_wlf = log10_sig_ref + alpha * logaT + m * np.log10(edot/1e-2)

    return u - log_sig_wlf  # 强制网络输出等于 WLF 真值

# 定义几何域
geom = dde.geometry.Rectangle([0, -3], [60, 1])  # T: 0-60°C, log10(edot): -3 to 1

# 添加观测数据约束
data_constraint = dde.icbc.PointSetBC(X, y, component=0)

# 创建 PDE 数据对象
data = dde.data.PDE(
    geom,
    pde,
    [data_constraint],
    num_domain=500,
    num_boundary=0,
    train_distribution="uniform"
)

# 构建网络
net = dde.nn.FNN([2] + [64]*3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# 编译模型
model.compile("adam", lr=1e-3, loss_weights=[1, 100])

# 训练
print("\n开始训练...")
losshistory, train_state = model.train(iterations=10000, display_every=1000)

# 使用 L-BFGS 进一步优化
print("\n使用 L-BFGS 优化...")
model.compile("L-BFGS", loss_weights=[1, 100])  # 保持与 Adam 相同的权重
losshistory, train_state = model.train()

# 3. 推理
x_test = np.array([
    # 边界条件测试
    [0, np.log10(1e-3)],     # 最低温度 + 最低剪切率
    [0, np.log10(1e0)],      # 最低温度 + 中等剪切率
    [60, np.log10(1e-3)],    # 最高温度 + 最低剪切率
    [60, np.log10(10)],      # 最高温度 + 最高剪切率

    # 中间区域测试
    [10, np.log10(1e-2)],    # 低温区
    [25, np.log10(1e-2)],    # 参考温度
    [40, np.log10(1e-1)],    # 中温区
    [50, np.log10(1e0)],     # 高温区

    # 对角线测试（温度和应变率同时变化）
    [15, np.log10(5e-3)],
    [30, np.log10(5e-2)],
    [45, np.log10(5e-1)],
]).astype(np.float32)

pred_log = model.predict(x_test)
pred = 10 ** pred_log  # 转换回线性空间

true_vals = yield_WLF(x_test[:,0], 10**x_test[:,1])


print("条件\t\t\t\t\t\t\t预测值(MPa)\t\t真实值(MPa)\t\t误差(%)")
print("-" * 80)
for i, (x, p, t) in enumerate(zip(x_test, pred, true_vals)):
    T_val, edot_val = x[0], 10**x[1]
    error = abs(p[0] - t) / t * 100
    print(f"T={T_val:5.1f}°C, ε̇={edot_val:.1e} s⁻¹\t{p[0]/1e6:.6f}\t\t{t/1e6:.6f}\t\t{error:.6f}")