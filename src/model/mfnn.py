"""
Multi-Fidelity Neural Network (MFNN) for Yield Stress Prediction

基于论文: "A composite neural network that learns from multi-fidelity data"
核心思想: 结合低保真度数据(合成数据)和高保真度数据(真实实验数据)

架构:
    y_hf(x) = y_lf(x) + α(x) × [y_lf(x) - y_lf_mean]

其中:
    - y_lf: 低保真度模型 (基于合成数据训练)
    - α(x): 非线性缩放因子 (学习低保真度和高保真度之间的关系)
    - y_hf: 高保真度输出
"""

import torch
import torch.nn as nn

from src.physics.yodel import yodel_mechanism, calc_m1, calc_phi_c


class LowFidelityNet(nn.Module):
    """低保真度网络 - 基于YODEL物理模型"""

    def __init__(self, input_dim=5, hidden_dim=128):
        """
        Args:
            input_dim: 输入维度 [Phi, d50, sigma, Emix, Temp]
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 神经网络预测物理参数
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # 输出: [Phi_m_raw, G_max_raw]
        )

        # G_max 的基准值
        self.register_buffer('g_max_base', torch.tensor(80000.0))

    def forward(self, x):
        """
        Args:
            x: [batch, 5] -> (Phi, d50, sigma, Emix, Temp)
        Returns:
            tau0_pred: 低保真度预测 [batch, 1]
            params: (phi_m, m1, g_max)
        """
        # 归一化
        scale = torch.tensor([1.0, 0.03, 0.6, 3.0e-8, 0.02], device=x.device)
        x_norm = x * scale

        # NN 预测
        out = self.net(x_norm)

        # 解析输入
        phi = x[:, 0]
        d50 = x[:, 1]
        sigma = x[:, 2]

        raw_phi_m = out[:, 0]
        raw_g_max = out[:, 1]

        # 物理约束
        max_packing = 0.76
        phi_m_pred = phi + torch.sigmoid(raw_phi_m) * (max_packing - phi)
        g_max_pred = torch.nn.functional.softplus(raw_g_max) * self.g_max_base

        # 物理计算
        m1_pred = calc_m1(d50, g_max_pred)
        phi_c = calc_phi_c(d50, sigma)

        # YODEL 主方程
        tau0_pred = yodel_mechanism(phi, phi_m_pred, phi_c, m1_pred)

        return tau0_pred, (phi_m_pred, m1_pred, g_max_pred)


class NonlinearScalingNet(nn.Module):
    """非线性缩放网络 - 学习低保真度到高保真度的映射"""

    def __init__(self, input_dim=5, hidden_dim=64):
        """
        Args:
            input_dim: 输入维度 [Phi, d50, sigma, Emix, Temp]
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 输入: x + y_lf (原始特征 + 低保真度预测)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # 输出: α(x)
        )

    def forward(self, x, y_lf):
        """
        Args:
            x: 输入特征 [batch, input_dim]
            y_lf: 低保真度预测 [batch, 1]
        Returns:
            alpha: 缩放因子 [batch, 1]
        """
        # 归一化
        scale = torch.tensor([1.0, 0.03, 0.6, 3.0e-8, 0.02], device=x.device)
        x_norm = x * scale

        # 拼接输入和低保真度预测
        combined = torch.cat([x_norm, y_lf], dim=-1)

        # 预测缩放因子
        alpha = self.net(combined)

        return alpha


class YodelMFNN(nn.Module):
    """
    Multi-Fidelity Neural Network for Yield Stress Prediction

    架构:
        y_hf(x) = y_lf(x) + α(x) × [y_lf(x) - y_lf_mean]

    训练策略:
        1. 第一阶段: 用大量低保真度数据(合成数据)训练
        2. 第二阶段: 固定 LowFidelityNet, 用少量高保真度数据训练 NonlinearScalingNet
        3. (可选) 第三阶段: 联合微调两个网络
    """

    def __init__(self, input_dim=5, lf_hidden_dim=128, alpha_hidden_dim=64):
        """
        Args:
            input_dim: 输入维度
            lf_hidden_dim: 低保真度网络隐藏层维度
            alpha_hidden_dim: 缩放网络隐藏层维度
        """
        super().__init__()

        # 低保真度网络
        self.lf_net = LowFidelityNet(input_dim, lf_hidden_dim)

        # 非线性缩放网络
        self.alpha_net = NonlinearScalingNet(input_dim, alpha_hidden_dim)

        # 低保真度预测的均值 (在训练时计算)
        self.register_buffer('y_lf_mean', torch.tensor(0.0))

    def forward(self, x, use_hf=True):
        """
        Args:
            x: [batch, 5] -> (Phi, d50, sigma, Emix, Temp)
            use_hf: 是否使用高保真度修正
        Returns:
            y_pred: 预测的屈服应力 [batch, 1]
            y_lf: 低保真度预测 [batch, 1]
            alpha: 缩放因子 [batch, 1] (仅当use_hf=True时返回)
            params: 物理参数 (phi_m, m1, g_max)
        """
        # 1. 低保真度预测
        y_lf, params = self.lf_net(x)

        if y_lf.dim() == 1:
            y_lf = y_lf.unsqueeze(1)

        if not use_hf:
            # 仅使用低保真度模型
            return y_lf, y_lf, None, params

        # 2. 高保真度修正
        alpha = self.alpha_net(x, y_lf)

        # 3. 组合预测
        # y_hf = y_lf + α × (y_lf - y_lf_mean)
        y_hf = y_lf + alpha * (y_lf - self.y_lf_mean)

        return y_hf, y_lf, alpha, params

    def set_y_lf_mean(self, mean_value):
        """设置低保真度预测的均值"""
        self.y_lf_mean = torch.tensor(mean_value, dtype=torch.float32)

    def freeze_lf_net(self):
        """冻结低保真度网络参数"""
        for param in self.lf_net.parameters():
            param.requires_grad = False

    def unfreeze_lf_net(self):
        """解冻低保真度网络参数"""
        for param in self.lf_net.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("Testing Multi-Fidelity Neural Network")
    print("=" * 60)

    model = YodelMFNN()

    # 模拟输入
    x = torch.tensor([[0.68, 20.0, 1.5, 3e7, 50.0],
                      [0.65, 25.0, 1.6, 4e7, 48.0]])

    print(f"\nInput shape: {x.shape}")
    print(f"Input:\n{x}")

    # 低保真度预测
    print("\n" + "-" * 60)
    print("Low-Fidelity Prediction (use_hf=False)")
    print("-" * 60)
    y_lf, _, _, params_lf = model(x, use_hf=False)
    print(f"Predicted Tau0 (LF): {y_lf.squeeze().detach().numpy()}")
    print(f"Phi_m: {params_lf[0].detach().numpy()}")
    print(f"G_max: {params_lf[2].detach().numpy()}")

    # 设置低保真度均值
    model.set_y_lf_mean(100.0)

    # 高保真度预测
    print("\n" + "-" * 60)
    print("High-Fidelity Prediction (use_hf=True)")
    print("-" * 60)
    y_hf, y_lf, alpha, params_hf = model(x, use_hf=True)
    print(f"Predicted Tau0 (LF): {y_lf.squeeze().detach().numpy()}")
    print(f"Predicted Tau0 (HF): {y_hf.squeeze().detach().numpy()}")
    print(f"Scaling Factor α: {alpha.squeeze().detach().numpy()}")
    print(f"Correction: {(y_hf - y_lf).squeeze().detach().numpy()}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
