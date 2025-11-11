import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

# 设置 DeepXDE 使用 float32
dde.config.set_default_float("float32")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# 2. 评估WLF机理模型的置信度（用于权重设置）
def evaluate_mechanism_confidence(T_samples, edot_samples, noise_level=0.02):
    """
    评估WLF机理模型的置信度

    参数:
        T_samples: 温度样本
        edot_samples: 应变率样本
        noise_level: 假设的实验数据噪声水平（相对误差）

    返回:
        confidence_score: 置信度分数（0-1），用于确定物理损失权重
        mechanism_error: 机理模型的估计误差
    """
    # 计算WLF预测值
    log_sig_wlf = yield_WLF_log(T_samples, edot_samples)

    # 模拟实验数据（添加噪声）
    log_sig_exp = log_sig_wlf + np.random.normal(0, noise_level, len(log_sig_wlf))

    # 计算机理模型的拟合误差
    mape = np.mean(np.abs(log_sig_wlf - log_sig_exp) / np.abs(log_sig_exp))
    r2 = r2_score(log_sig_exp, log_sig_wlf)

    # 置信度评分：R²越高、MAPE越低，置信度越高
    confidence_score = r2 * (1 - mape)
    mechanism_error = mape

    print(f"\n=== WLF机理模型置信度评估 ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape*100:.2f}%")
    print(f"置信度分数: {confidence_score:.4f}")
    print(f"建议物理损失权重: {confidence_score*100:.4f}")
    print(f"建议数据损失权重: {(1-confidence_score)*100:.4f}")

    return confidence_score, mechanism_error

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

# 评估机理模型置信度并确定权重
confidence, mech_error = evaluate_mechanism_confidence(T, edot, noise_level=0.02)

# 根据置信度自适应设置权重
# 置信度高 -> 物理损失权重高；置信度低 -> 数据损失权重高
physics_weight = float(confidence * 100)  # 物理损失权重
data_weight = float((1 - confidence) * 100)  # 数据损失权重

print(f"\n=== 自适应权重设置 ===")
print(f"物理损失权重: {physics_weight:.4f}")
print(f"数据损失权重: {data_weight:.4f}")
print(f"权重比 (物理:数据) = 1:{data_weight/physics_weight:.4f}")

# 3. PINN 定义物理损失（在对数空间）
def pde(x, u):
    """u 是预测的 log10(σ_y)"""
    T, log_edot = x[:,0:1], x[:,1:2]

    # WLF 公式在对数空间
    logaT = -8.86*(T-25)/(101.6+T-25)
    sig_ref, alpha, m = 2.1e6, 0.78, 0.11

    # 使用 numpy 的 log10 常量值
    log10_sig_ref = 6.322219  # np.log10(2.1e6)
    log10_1e_minus_2 = -2.0   # np.log10(1e-2)

    log_sig_wlf = log10_sig_ref + alpha * logaT + m * log_edot - m * log10_1e_minus_2

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

# 构建网络（针对燃料屈服值的非线性特性优化）
# 使用更深的网络结构和Swish激活函数以更好地捕捉WLF方程的强非线性
# 改进策略：
# - 增加网络深度（3层 -> 5层）以提升对WLF方程强非线性的拟合能力
# - 使用Swish激活函数（比tanh更适合深度网络，梯度流动更好）
# - 增加每层节点数（64 -> 128）以提升表达能力
# - Glorot初始化保证训练稳定性
net = dde.nn.FNN([2] + [128]*5 + [1], "swish", "Glorot normal")
model = dde.Model(data, net)

# 编译模型（使用自适应权重）
model.compile("adam", lr=1e-3, loss_weights=[physics_weight, data_weight])

# 训练
print("\n开始训练...")
losshistory, train_state = model.train(iterations=10000, display_every=1000)

# 使用 L-BFGS 进一步优化
print("\n使用 L-BFGS 优化...")
model.compile("L-BFGS", loss_weights=[physics_weight, data_weight])
losshistory, train_state = model.train()

# 4. 全面测试集设计
# 覆盖边界条件和中间区域
x_test = np.array([
    # 边界条件测试
    [0, np.log10(1e-3)],     # 最低温度 + 最低应变率
    [0, np.log10(1e0)],      # 最低温度 + 中等应变率
    [60, np.log10(1e-3)],    # 最高温度 + 最低应变率
    [60, np.log10(10)],      # 最高温度 + 最高应变率

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

# 计算评估指标
errors = np.abs(pred.flatten() - true_vals) / true_vals * 100
mape_test = np.mean(errors)
max_error = np.max(errors)
r2_test = r2_score(true_vals, pred.flatten())

print("\n" + "="*80)
print("测试集评估结果")
print("="*80)
print(f"平均绝对百分比误差 (MAPE): {mape_test:.4f}%")
print(f"最大相对误差: {max_error:.4f}%")
print(f"R² Score: {r2_test:.6f}")
print("="*80)

print("\n条件\t\t\t\t\t\t\t预测值(MPa)\t\t真实值(MPa)\t\t误差(%)")
print("-" * 80)
for i, (x, p, t) in enumerate(zip(x_test, pred, true_vals)):
    T_val, edot_val = x[0], 10**x[1]
    error = abs(p[0] - t) / t * 100
    print(f"T={T_val:5.1f}°C, ε̇={edot_val:.1e} s⁻¹\t{p[0]/1e6:.6f}\t\t{t/1e6:.6f}\t\t{error:.6f}")

