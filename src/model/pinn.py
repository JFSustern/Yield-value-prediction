
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
        # 预测中间物理参数 (Phi_m_eff, G_max_factor)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # 输出: [Phi_m_pred, G_max_factor]
        )

        # G_max 的基准值
        self.register_buffer('g_max_base', torch.tensor(80000.0))

    def forward(self, x):
        """
        Args:
            x: [batch, 6] -> (Phi_final, d50, sigma, Emix, Temp, Ratio_curing)
        """
        # 1. 输入预处理 (归一化) 给 NN
        # Scale: [1, 0.03, 0.6, 2e-9, 0.02, 5.0]
        # Ratio_curing ~ 0.2 -> * 5.0 = 1.0
        scale = torch.tensor([1.0, 0.03, 0.6, 2.0e-9, 0.02, 5.0], device=x.device)
        x_norm = x * scale

        # NN 预测
        out = self.net(x_norm)

        # 解析原始物理量
        phi_final = x[:, 0]
        d50 = x[:, 1]
        sigma = x[:, 2]
        ratio_curing = x[:, 5]

        raw_phi_m = out[:, 0]
        raw_g_max = out[:, 1]

        # 2. 物理参数解码

        # 计算 Phi_peak (基于稀释原理)
        # Phi_peak = Phi_final / (Phi_final + (1-ratio)*(1-Phi_final))
        # 注意：这里假设 Phi_final 是最终固含量，Ratio 是固化剂占液相比例
        v_solid = phi_final
        v_liquid_total = 1.0 - phi_final
        v_liquid_other = v_liquid_total * (1.0 - ratio_curing)
        phi_peak = v_solid / (v_solid + v_liquid_other)

        # Phi_m: 必须大于 Phi_peak (因为 Phi_peak > Phi_final)
        # 限制在 [Phi_peak, 0.74] 之间
        max_packing = 0.74
        # 使用 Softplus 保证正数差值
        phi_m_pred = phi_peak + torch.sigmoid(raw_phi_m) * (max_packing - phi_peak)

        # G_max
        g_max_pred = torch.nn.functional.softplus(raw_g_max) * self.g_max_base

        # 3. 物理层 (Differentiable Physics Layer)

        # 计算 m1
        m1_pred = calc_m1(d50, g_max_pred)

        # 计算 Phi_c
        phi_c = calc_phi_c(d50, sigma)

        # 调用 YODEL 主方程 (双阶段)
        tau0_peak = yodel_mechanism(phi_peak, phi_m_pred, phi_c, m1_pred)
        tau0_final = yodel_mechanism(phi_final, phi_m_pred, phi_c, m1_pred)

        # 拼接输出 [batch, 2]
        tau0_pred = torch.stack([tau0_peak, tau0_final], dim=1)

        return tau0_pred, (phi_m_pred, m1_pred, g_max_pred, phi_peak)

if __name__ == "__main__":
    # 测试模型
    model = YodelPINN()
    # Phi=0.68, d50=20, sigma=1.5, Emix=1e5, T=50, Ratio=0.15
    x = torch.tensor([[0.68, 20.0, 1.5, 1e5, 50.0, 0.15]])
    tau0, params = model(x)
    print(f"Input: {x}")
    print(f"Predicted Tau0 (Peak, Final): {tau0.detach().numpy()} Pa")
    print(f"Phi_peak: {params[3].item():.4f}")

