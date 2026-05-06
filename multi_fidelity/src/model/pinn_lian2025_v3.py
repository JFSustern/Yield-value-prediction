"""
LianPINN v3 - 可配置网络结构

在 v2 基础上支持:
  - 可变宽度 (hidden_dim)
  - 可变深度 (n_hidden_layers: 隐藏层数量，不含输入/输出层)
  - 可选激活函数 (tanh / gelu / silu)

物理公式与 v2 完全相同:
  τ = m1 * φ³ / [φ_max(SP) * (φ_max(SP) - φ)]
  m1 = 0.72 Pa (固定)
"""

import torch
import torch.nn as nn


_ACTIVATIONS = {
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
}


class LianPINN_v3(nn.Module):

    M1 = 0.72  # Pa

    def __init__(self, hidden_dim=64, n_hidden_layers=3, activation='tanh'):
        """
        Args:
            hidden_dim:       每个隐藏层的宽度
            n_hidden_layers:  隐藏层数量 (不含输入投影层和输出层)
                              n_hidden_layers=3 等价于 v2 的 4 层 Linear 结构
            activation:       'tanh' | 'gelu' | 'silu'
        """
        super().__init__()
        assert activation in _ACTIVATIONS, f"activation 须为 {list(_ACTIVATIONS)}"
        assert n_hidden_layers >= 1

        act_cls = _ACTIVATIONS[activation]

        layers = []
        # 输入层
        layers += [nn.Linear(2, hidden_dim), act_cls()]
        # 隐藏层
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_cls()]
        # 输出层
        layers += [nn.Linear(hidden_dim, 1)]

        self.net = nn.Sequential(*layers)

        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation

    def forward(self, x):
        """
        Args:
            x: [batch, 2]  -> (Phi, SP_percent)
        Returns:
            tau0_pred:    [batch]
            phi_max_pred: [batch]
        """
        phi = x[:, 0]

        scale = torch.tensor([2.0, 1.5], device=x.device, dtype=x.dtype)
        x_norm = x * scale

        raw = self.net(x_norm).squeeze(-1)

        phi_max_pred = phi + 0.05 + torch.sigmoid(raw) * (0.95 - phi - 0.05)

        epsilon = 1e-6
        diff_m = torch.relu(phi_max_pred - phi) + epsilon
        tau0_pred = self.M1 * phi**3 / (phi_max_pred * diff_m)

        return tau0_pred, phi_max_pred

    def describe(self):
        total = sum(p.numel() for p in self.parameters())
        return (f"LianPINN_v3(hidden={self.hidden_dim}, "
                f"layers={self.n_hidden_layers}, act={self.activation}) "
                f"| {total:,} params")


if __name__ == "__main__":
    configs = [
        (64,  3, 'tanh'),   # v2 基线
        (128, 3, 'tanh'),   # 加宽
        (64,  5, 'tanh'),   # 加深
        (128, 4, 'gelu'),   # 宽+深+gelu
        (64,  3, 'silu'),   # 换激活
    ]
    for h, l, a in configs:
        m = LianPINN_v3(h, l, a)
        print(m.describe())
