
import torch
import torch.nn as nn

from src.physics.yodel import yodel_mechanism, calc_m1, calc_phi_c


class YodelPINN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        """
        Physics-Integrated Neural Network for Dual-Stage Yield Stress Prediction

        Args:
            input_dim: 输入维度 [Phi_final, d50, sigma, Emix, Temp, Ratio_curing]
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 1. 神经网络部分 (Parameter Estimator)
        # 预测中间物理参数:
        # 1. Phi_m_pred: 最大堆积密度
        # 2. G_max_factor: 相互作用力修正系数
        # 3. Phi_peak_delta: Peak阶段固含量相对于Final的增量 (全黑盒预测)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3) # 输出: [Phi_m, G_max, Phi_peak_delta]
        )

        # G_max 的基准值
        self.register_buffer('g_max_base', torch.tensor(80000.0))

    def forward(self, x):
        """
        Args:
            x: [batch, 6] -> (Phi_final, d50, sigma, Emix, Temp, Ratio_curing)
        """
        # 1. 输入预处理 (归一化) 给 NN
        scale = torch.tensor([1.0, 0.03, 0.6, 2.0e-9, 0.02, 5.0], device=x.device)
        x_norm = x * scale

        # NN 预测
        out = self.net(x_norm)

        # 解析原始物理量
        phi_final = x[:, 0]
        d50 = x[:, 1]
        sigma = x[:, 2]
        # ratio_curing = x[:, 5] # 模型现在通过黑盒方式学习 ratio 对 phi_peak 的影响

        raw_phi_m = out[:, 0]
        raw_g_max = out[:, 1]
        raw_phi_peak_delta = out[:, 2]

        # 2. 物理参数解码

        # A. 全黑盒预测 Phi_peak
        # 物理约束：Phi_peak 必须大于 Phi_final (因为少了液体)
        # 策略：预测增量 delta > 0
        # 使用 Softplus 保证正数，且梯度平滑
        phi_peak_delta = torch.nn.functional.softplus(raw_phi_peak_delta)
        phi_peak_pred = phi_final + phi_peak_delta

        # B. Phi_m 解码
        # 物理约束：Phi_m 必须大于 Phi_peak_pred (否则堵塞)
        # 限制在 [Phi_peak_pred, 0.80] 之间
        max_packing = 0.80
        # 使用 Sigmoid 保证在合理区间
        phi_m_pred = phi_peak_pred + torch.sigmoid(raw_phi_m) * (max_packing - phi_peak_pred)

        # C. G_max 解码
        g_max_pred = torch.nn.functional.softplus(raw_g_max) * self.g_max_base

        # 3. 物理层 (Differentiable Physics Layer)

        # 计算 m1
        m1_pred = calc_m1(d50, g_max_pred)

        # 计算 Phi_c
        phi_c = calc_phi_c(d50, sigma)

        # 调用 YODEL 主方程 (双阶段)
        # Peak 阶段使用黑盒预测的 Phi_peak_pred
        tau0_peak = yodel_mechanism(phi_peak_pred, phi_m_pred, phi_c, m1_pred)

        # Final 阶段使用输入的 Phi_final
        tau0_final = yodel_mechanism(phi_final, phi_m_pred, phi_c, m1_pred)

        # 拼接输出 [batch, 2]
        tau0_pred = torch.stack([tau0_peak, tau0_final], dim=1)

        # 返回参数供监控
        return tau0_pred, (phi_m_pred, m1_pred, g_max_pred, phi_peak_pred, phi_peak_delta)

if __name__ == "__main__":
    # 测试模型
    model = YodelPINN()
    x = torch.tensor([[0.68, 20.0, 1.5, 1e5, 50.0, 0.15]])
    tau0, params = model(x)
    print(f"Input: {x}")
    print(f"Predicted Tau0: {tau0.detach().numpy()} Pa")
    print(f"Phi_peak (Black-box): {params[3].item():.4f} (Delta: {params[4].item():.4f})")

