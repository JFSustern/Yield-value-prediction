import deepxde as dde
import matplotlib.pyplot as plt
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

# 构建网络
net = dde.nn.FNN([2] + [64]*3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# 编译模型
model.compile(
    "adam",
    lr=1e-3,
    loss_weights=[1, 100]   ## 物理损失与数据损失的权重
)

# 训练
print("\n开始训练...")
losshistory_adam, train_state = model.train(iterations=10000, display_every=1000)

# 使用 L-BFGS 进一步优化
print("\n使用 L-BFGS 优化...")
model.compile("L-BFGS")
losshistory_lbfgs, train_state = model.train()

# 合并两个阶段的损失历史
loss_train_adam = np.array(losshistory_adam.loss_train)
loss_train_lbfgs = np.array(losshistory_lbfgs.loss_train)

print(f"\nAdam 阶段训练了 {len(loss_train_adam)} 轮记录")
print(f"L-BFGS 阶段训练了 {len(loss_train_lbfgs)} 轮")

# 对 L-BFGS 数据进行下采样，每 5 步取一个点（使图表更清晰）
# 跳过第一个点（索引 0），因为它与 Adam 的最后一个点重合
downsample_rate = 5
# 从索引1开始，每隔downsample_rate取一个点
lbfgs_indices = np.arange(1, len(loss_train_lbfgs), downsample_rate)
loss_train_lbfgs_downsampled = loss_train_lbfgs[lbfgs_indices]

# 创建迭代次数数组
# Adam: 每1000次记录一次，所以迭代次数是 0, 1000, 2000, ..., 10000
iterations_adam = np.arange(len(loss_train_adam)) * 1000
# L-BFGS: 每次都记录，迭代次数是 10001, 10002, ..., 10000+len(loss_train_lbfgs)
# 下采样后的迭代次数
iterations_lbfgs_downsampled = lbfgs_indices + 10000

print(f"Adam 迭代次数范围: {iterations_adam[0]} - {iterations_adam[-1]}")
print(f"L-BFGS 迭代次数范围: {iterations_lbfgs_downsampled[0]} - {iterations_lbfgs_downsampled[-1]}")
print(f"绘图数据点数: Adam={len(iterations_adam)}, L-BFGS={len(iterations_lbfgs_downsampled)}")

# 3. 推理
x_test = np.array([
    [10, np.log10(1e-2)],    # 10 °C, 0.01 s⁻¹
    [25, np.log10(1e-2)],    # 25 °C, 0.01 s⁻¹ (参考条件)
    [50, np.log10(1e0)],     # 50 °C, 1 s⁻¹
]).astype(np.float32)

pred_log = model.predict(x_test)
pred = 10 ** pred_log  # 转换回线性空间

true_vals = yield_WLF(x_test[:,0], 10**x_test[:,1])

print("\n预测结果对比:")
print("条件\t\t\t预测值(MPa)\t真实值(MPa)\t误差(%)")
print("-" * 60)
for i, (x, p, t) in enumerate(zip(x_test, pred, true_vals)):
    T_val, edot_val = x[0], 10**x[1]
    error = abs(p[0] - t) / t * 100
    print(f"T={T_val:5.1f}°C, ε̇={edot_val:.1e} s⁻¹\t{p[0]/1e6:.2f}\t\t{t/1e6:.2f}\t\t{error:.2f}")

# 4. 可视化
print("\n生成可视化图表...")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 12))

# ============ 图 1：训练损失曲线 ============
ax1 = plt.subplot(2, 3, 1)
# 分别绘制Adam和L-BFGS阶段，避免连线问题
ax1.semilogy(iterations_adam, loss_train_adam[:, 0], label='物理损失 (PDE)', linewidth=2, marker='o', markersize=4, color='C0')
ax1.semilogy(iterations_lbfgs_downsampled, loss_train_lbfgs_downsampled[:, 0], linewidth=2, marker='o', markersize=4, color='C0')
ax1.semilogy(iterations_adam, loss_train_adam[:, 1], label='数据损失 (Data)', linewidth=2, marker='s', markersize=4, color='C1')
ax1.semilogy(iterations_lbfgs_downsampled, loss_train_lbfgs_downsampled[:, 1], linewidth=2, marker='s', markersize=4, color='C1')
ax1.axvline(x=10000, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Adam → L-BFGS')
ax1.set_xlabel('迭代次数', fontsize=11)
ax1.set_ylabel('损失值', fontsize=11)
ax1.set_title('训练损失曲线（两阶段）', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# ============ 图 2：总损失曲线 ============
ax2 = plt.subplot(2, 3, 2)
# 分别计算两个阶段的总损失
total_loss_adam = loss_train_adam[:, 0] + loss_train_adam[:, 1]
total_loss_lbfgs = loss_train_lbfgs_downsampled[:, 0] + loss_train_lbfgs_downsampled[:, 1]
# 分别绘制，避免连线
ax2.semilogy(iterations_adam, total_loss_adam, color='purple', linewidth=2, marker='o', markersize=4, label='总损失')
ax2.semilogy(iterations_lbfgs_downsampled, total_loss_lbfgs, color='purple', linewidth=2, marker='o', markersize=4)
ax2.axvline(x=10000, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Adam → L-BFGS')
ax2.set_xlabel('迭代次数', fontsize=11)
ax2.set_ylabel('总损失值', fontsize=11)
ax2.set_title('总损失曲线（两阶段）', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# ============ 图 3：预测值 vs 真实值（测试点） ============
ax3 = plt.subplot(2, 3, 3)
x_labels = [f'T={int(x[0])}°C\nε̇={10**x[1]:.1e}' for x in x_test]
x_pos = np.arange(len(x_labels))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, pred.flatten()/1e6, width, label='PINN 预测', alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, true_vals/1e6, width, label='WLF 真实值', alpha=0.8)

ax3.set_ylabel('屈服强度 (MPa)', fontsize=11)
ax3.set_title('预测值 vs 真实值', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(x_labels, fontsize=9)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# ============ 图 4：预测误差 ============
ax4 = plt.subplot(2, 3, 4)
errors = np.abs(pred.flatten() - true_vals) / true_vals * 100
colors = ['green' if e < 0.5 else 'orange' if e < 1 else 'red' for e in errors]
bars = ax4.bar(x_labels, errors, color=colors, alpha=0.7)
ax4.set_ylabel('相对误差 (%)', fontsize=11)
ax4.set_title('预测相对误差', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 添加误差值标签
for i, (bar, error) in enumerate(zip(bars, errors)):
    ax4.text(bar.get_x() + bar.get_width()/2., error,
            f'{error:.3f}%', ha='center', va='bottom', fontsize=9)

# ============ 图 5：3D 曲面图（PINN 预测） ============
ax5 = plt.subplot(2, 3, 5, projection='3d')

# 生成网格数据
T_grid = np.linspace(0, 60, 30)
log_edot_grid = np.linspace(-3, 1, 30)
T_mesh, log_edot_mesh = np.meshgrid(T_grid, log_edot_grid)

# 预测
X_grid = np.c_[T_mesh.flatten(), log_edot_mesh.flatten()].astype(np.float32)
pred_grid = model.predict(X_grid)
pred_grid = 10 ** pred_grid.reshape(T_mesh.shape)  # 转换回线性空间

# 绘制曲面
surf = ax5.plot_surface(T_mesh, log_edot_mesh, pred_grid/1e6, cmap='viridis', alpha=0.8)
ax5.set_xlabel('温度 (°C)', fontsize=10)
ax5.set_ylabel('log10(应变速率)', fontsize=10)
ax5.set_zlabel('屈服强度 (MPa)', fontsize=10)
ax5.set_title('PINN 预测曲面', fontsize=12, fontweight='bold')
fig.colorbar(surf, ax=ax5, label='屈服强度 (MPa)', shrink=0.5)

# ============ 图 6：3D 曲面图（WLF 真实值） ============
ax6 = plt.subplot(2, 3, 6, projection='3d')

# 计算 WLF 真实值
true_grid = yield_WLF(T_mesh, 10**log_edot_mesh)

# 绘制曲面
surf2 = ax6.plot_surface(T_mesh, log_edot_mesh, true_grid/1e6, cmap='plasma', alpha=0.8)
ax6.set_xlabel('温度 (°C)', fontsize=10)
ax6.set_ylabel('log10(应变速率)', fontsize=10)
ax6.set_zlabel('屈服强度 (MPa)', fontsize=10)
ax6.set_title('WLF 真实值曲面', fontsize=12, fontweight='bold')
fig.colorbar(surf2, ax=ax6, label='屈服强度 (MPa)', shrink=0.5)

plt.tight_layout()
plt.savefig('/Users/sunjifei/Desktop/文献/Project/training_visualization.png', dpi=300, bbox_inches='tight')
print("✓ 可视化图表已保存到：training_visualization.png")
plt.show()

# ============ 额外的评估指标 ============
print("\n" + "="*60)
print("模型评估指标")
print("="*60)

# 在全部训练数据上评估
pred_log_train = model.predict(X)
pred_train = 10 ** pred_log_train
y_true = 10 ** y

mae = np.mean(np.abs(pred_train - y_true))
mse = np.mean((pred_train - y_true) ** 2)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((pred_train - y_true) / y_true)) * 100
r2 = 1 - np.sum((pred_train - y_true) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

print(f"平均绝对误差 (MAE):        {mae:.6e} Pa")
print(f"均方误差 (MSE):            {mse:.6e} Pa²")
print(f"均方根误差 (RMSE):         {rmse:.6e} Pa")
print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
print(f"决定系数 (R²):             {r2:.6f}")
print("="*60)

