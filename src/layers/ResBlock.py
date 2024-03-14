import torch
import torch.nn as nn


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup = None, expand_ratio = 1.5, act = nn.GELU):
        super(InvertedResidual, self).__init__()

        if oup is None:
            oup = inp

        hidden_dim = int(inp * expand_ratio)

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            act(),
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            act(),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            # nn.BatchNorm2d(oup),
        )


    def forward(self, x):
        return x + self.conv(x)
