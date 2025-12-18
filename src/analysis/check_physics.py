
import matplotlib.pyplot as plt
import pandas as pd


def analyze_physics_correlations():
    # 读取测试结果
    df = pd.read_csv("data/synthetic/test_results_dual.csv")

    # 1. 分析 Phi_peak 的物理规律
    # 计算预测的增量
    df['Delta_Pred'] = df['Phi_peak_Pred'] - df['Phi_final(固含量)']

    # 理论上的增量 (根据体积守恒公式)
    phi = df['Phi_final(固含量)']
    ratio = df['Curing_Ratio(固化剂比例)']
    denom = phi + (1 - ratio) * (1 - phi)
    df['Phi_peak_Theory'] = phi / denom
    df['Delta_Theory'] = df['Phi_peak_Theory'] - phi

    plt.figure(figsize=(18, 10))

    # 图1: Delta vs Ratio (验证稀释效应)
    plt.subplot(2, 3, 1)
    plt.scatter(df['Curing_Ratio(固化剂比例)'], df['Delta_Pred'], alpha=0.6, c='blue')
    plt.xlabel('Curing Agent Ratio')
    plt.ylabel('Predicted Delta (Phi_peak - Phi_final)')
    plt.title('1. Dilution Effect: Ratio vs Delta')
    plt.grid(True)

    # 图2: Phi_m vs Sigma (验证级配效应)
    plt.subplot(2, 3, 2)
    plt.scatter(df['sigma(几何标准差)'], df['Phi_m_Pred'], alpha=0.6, c='green')
    plt.xlabel('Sigma (Particle Distribution Width)')
    plt.ylabel('Predicted Phi_m')
    plt.title('2. Packing Efficiency: Sigma vs Phi_m')
    plt.grid(True)

    # 图3: Phi_m vs Emix (验证混合效应)
    plt.subplot(2, 3, 3)
    plt.scatter(df['Emix(混合功_J)'], df['Phi_m_Pred'], alpha=0.6, c='purple')
    plt.xlabel('Mixing Energy (J)')
    plt.ylabel('Predicted Phi_m')
    plt.title('3. Process Effect: Emix vs Phi_m')
    plt.grid(True)

    # 图4: Phi_m 分布直方图 (验证多样性)
    plt.subplot(2, 3, 4)
    plt.hist(df['Phi_m_Pred'], bins=30, color='orange', alpha=0.7)
    plt.xlabel('Predicted Phi_m')
    plt.ylabel('Count')
    plt.title(f'4. Phi_m Distribution (Std: {df["Phi_m_Pred"].std():.4f})')
    plt.grid(True)

    # 图5: Phi_peak vs Phi_final (验证基础关系)
    plt.subplot(2, 3, 5)
    plt.scatter(df['Phi_final(固含量)'], df['Phi_peak_Pred'], alpha=0.6, c='red')
    plt.plot([0.6, 0.8], [0.6, 0.8], 'k--', label='y=x')
    plt.xlabel('Phi_final (Input)')
    plt.ylabel('Phi_peak (Predicted)')
    plt.title('5. Phi_peak vs Phi_final')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = "data/synthetic/plots/physics_correlations.png"
    plt.savefig(save_path)
    print(f"Physics correlation analysis saved to {save_path}")

    # 打印统计信息
    print("\nStatistics:")
    print(f"Phi_m Std Dev: {df['Phi_m_Pred'].std():.6f} (If 0, it's constant)")
    print(f"Phi_peak Std Dev: {df['Phi_peak_Pred'].std():.6f}")

if __name__ == "__main__":
    analyze_physics_correlations()

