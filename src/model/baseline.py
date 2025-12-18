import torch
import torch.nn as nn

class PureNN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128):
        """
        Pure Data-Driven Neural Network (Baseline)
        直接从输入特征映射到双阶段屈服值，不包含任何物理层。
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # 使用标准的 ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 直接输出 [Tau0_peak, Tau0_final]
        )

    def forward(self, x):
        # 输入归一化 (与 PINN 保持一致，公平对比)
        scale = torch.tensor([1.0, 0.03, 0.6, 2.0e-9, 0.02, 5.0], device=x.device)
        x_norm = x * scale

        # 直接预测
        out = self.net(x_norm)

        # 强制非负 (物理量不能为负)
        return torch.nn.functional.softplus(out)

