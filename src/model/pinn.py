# src/model/pinn.py

import torch
import torch.nn as nn

from src.physics.yodel import yodel_mechanism, calc_m1, calc_phi_c


class YodelPINN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        """
        Physics-Integrated Neural Network for Yield Stress Prediction

        Args:
            input_dim: 输入维度 [Phi, d50, sigma, Emix, Temp]
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 1. 神经网络部分 (Parameter Estimator)
        # 任务：从输入预测难以测量的中间物理参数 (Phi_m_eff, G_max_factor)
        # 注意：不再直接预测 m1，而是预测 G_max 的修正系数
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # 输出: [Phi_m_pred, G_max_factor]
        )

        # G_max 的基准值 (与 generator.py 保持一致)
        # 对应真实物理力 ~80 nN
        self.register_buffer('g_max_base', torch.tensor(80000.0))

    def forward(self, x):
        """
        Args:
            x: [batch, 5] -> (Phi, d50, sigma, Emix, Temp)
            注意：为了训练稳定性，建议输入已经过归一化。
            但物理层需要真实物理量。
            这里假设输入 x 是【原始物理量】。
            为了防止 NN 饱和，我们在送入 net 之前手动归一化。
        """
        # 1. 输入预处理 (归一化) 给 NN
        # 粗略统计特征: Phi~0.7, d50~30, sigma~1.5, Emix~1e6, Temp~50
        # Scale: [1, 0.03, 0.6, 1e-6, 0.02]
        scale = torch.tensor([1.0, 0.03, 0.6, 1e-6, 0.02], device=x.device)
        x_norm = x * scale

        # NN 预测
        out = self.net(x_norm)

        # 解析原始物理量给物理层
        phi = x[:, 0]
        d50 = x[:, 1]
        sigma = x[:, 2]
        # emix = x[:, 3] # 暂时没用到，因为 Phi_m 由网络直接预测了综合效应
        # temp = x[:, 4] # 暂时没用到，因为 G_max 由网络预测了综合效应

        raw_phi_m = out[:, 0]
        raw_g_max = out[:, 1]

        # 2. 物理参数解码

        # Phi_m: 必须大于当前的 Phi
        # 使用 Sigmoid + 缩放，限制在 [Phi, 0.74] 之间
        # 改进：Phi_m = Phi + Sigmoid(raw) * (0.74 - Phi)
        # 这样保证了 Phi < Phi_m < 0.74 (FCC极限)
        # 避免了之前的 clamp 导致的梯度消失
        max_packing = 0.74
        phi_m_pred = phi + torch.sigmoid(raw_phi_m) * (max_packing - phi)

        # G_max: 必须 > 0
        # 网络预测的是相对于基准值 80000 的修正系数
        # 使用 Softplus 保证正数，且初始值接近 1.0 (Softplus(0.55) ~ 1.0)
        g_max_pred = torch.nn.functional.softplus(raw_g_max) * self.g_max_base

        # 3. 物理层 (Differentiable Physics Layer)

        # 计算 m1 (使用物理公式!)
        m1_pred = calc_m1(d50, g_max_pred)

        # 计算 Phi_c (确定性几何关系)
        # Phi_c ~ 0.28 * (1 + CV)
        phi_c = calc_phi_c(d50, sigma)

        # 调用 YODEL 主方程
        tau0_pred = yodel_mechanism(phi, phi_m_pred, phi_c, m1_pred)

        return tau0_pred, (phi_m_pred, m1_pred, g_max_pred)

if __name__ == "__main__":
    # 测试模型前向传播
    model = YodelPINN()
    # 模拟输入: Phi=0.68, d50=20, sigma=1.5, Emix=1e5, T=50
    x = torch.tensor([[0.68, 20.0, 1.5, 1e5, 50.0]])
    tau0, params = model(x)
    print(f"Input: {x}")
    print(f"Predicted Tau0: {tau0.item():.2f} Pa")
    print(f"Predicted Params: Phi_m={params[0].item():.4f}, m1={params[1].item():.2f}, G_max={params[2].item():.2f}")

