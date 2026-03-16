
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
            x: [batch, 5] -> (Phi, d50, sigma, Emix, Temp)
        """
        # 1. 输入预处理 (归一化) 给 NN
        # 粗略统计特征: Phi~0.7, d50~30, sigma~1.5, Emix~3e7, Temp~50
        # Scale: [1, 0.03, 0.6, 3e-8, 0.02]
        scale = torch.tensor([1.0, 0.03, 0.6, 3.0e-8, 0.02], device=x.device)
        x_norm = x * scale

        # NN 预测
        out = self.net(x_norm)

        # 解析原始物理量给物理层
        phi = x[:, 0]
        d50 = x[:, 1]
        sigma = x[:, 2]

        raw_phi_m = out[:, 0]
        raw_g_max = out[:, 1]

        # 2. 物理参数解码

        # Phi_m: 必须大于当前的 Phi
        # 使用 Sigmoid + 缩放，限制在 [Phi, 0.76] 之间
        max_packing = 0.76
        phi_m_pred = phi + torch.sigmoid(raw_phi_m) * (max_packing - phi)

        # G_max: 必须 > 0
        g_max_pred = torch.nn.functional.softplus(raw_g_max) * self.g_max_base

        # 3. 物理层 (Differentiable Physics Layer)

        # 计算 m1 (使用物理公式!)
        m1_pred = calc_m1(d50, g_max_pred)

        # 计算 Phi_c (确定性几何关系)
        phi_c = calc_phi_c(d50, sigma)

        # 调用 YODEL 主方程
        tau0_pred = yodel_mechanism(phi, phi_m_pred, phi_c, m1_pred)

        return tau0_pred, (phi_m_pred, m1_pred, g_max_pred)

if __name__ == "__main__":
    # 测试模型前向传播
    model = YodelPINN()
    # 模拟输入: Phi=0.68, d50=20, sigma=1.5, Emix=3e7, T=50
    x = torch.tensor([[0.68, 20.0, 1.5, 3e7, 50.0]])
    tau0, params = model(x)
    print(f"Input: {x}")
    print(f"Predicted Tau0: {tau0.item():.2f} Pa")
    print(f"Predicted Params: Phi_m={params[0].item():.4f}, m1={params[1].item():.2f}, G_max={params[2].item():.2f}")

