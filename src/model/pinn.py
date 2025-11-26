# src/model/pinn.py

import torch
import torch.nn as nn

from src.physics.yodel import yodel_mechanism


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
        # 任务：从输入预测难以测量的中间物理参数 (Phi_m_eff, m1_eff)
        # 注意：虽然 m1 可以由 d50 算出来，但这里让网络学习综合效应
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Tanh 通常比 ReLU 更适合物理回归 (平滑)
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2) # 输出: [Phi_m_pred, m1_pred]
        )

        # 物理参数的缩放因子 (用于归一化输出)
        self.register_buffer('phi_m_scale', torch.tensor(1.0))
        self.register_buffer('m1_scale', torch.tensor(1000.0)) # m1 通常在 Pa 量级

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

        # 激活函数确保物理意义
        # Phi_m 必须在 (Phi, 1.0) 之间 -> Sigmoid + 缩放
        # m1 必须 > 0 -> Softplus

        raw_phi_m = out[:, 0]
        raw_m1 = out[:, 1]

        # 物理约束：Phi_m 必须大于当前的 Phi
        # Phi_m_pred = Phi + Softplus(raw) + epsilon
        phi_m_pred = phi + torch.nn.functional.softplus(raw_phi_m) + 1e-3
        # 同时也应该小于 0.74 (FCC) 或 1.0
        phi_m_pred = torch.clamp(phi_m_pred, max=0.99)

        m1_pred = torch.nn.functional.softplus(raw_m1) * self.m1_scale

        # 2. 物理层 (Differentiable Physics Layer)
        # 计算 Phi_c (确定性几何关系)
        # 注意：这里需要引入 yodel.py 中的 calc_phi_c
        # 为了保持梯度，直接在这里实现或调用
        # Phi_c ~ 0.28 * (1 + (sigma - 1)*0.5)
        phi_c = 0.28 * (1 + (sigma - 1.0) * 0.5)

        # 调用 YODEL 主方程
        tau0_pred = yodel_mechanism(phi, phi_m_pred, phi_c, m1_pred)

        return tau0_pred, (phi_m_pred, m1_pred)

if __name__ == "__main__":
    # 测试模型前向传播
    model = YodelPINN()
    # 模拟输入: Phi=0.68, d50=20, sigma=1.5, Emix=1e5, T=50
    x = torch.tensor([[0.68, 20.0, 1.5, 1e5, 50.0]])
    tau0, params = model(x)
    print(f"Input: {x}")
    print(f"Predicted Tau0: {tau0.item():.2f} Pa")
    print(f"Predicted Params: Phi_m={params[0].item():.4f}, m1={params[1].item():.2f}")

