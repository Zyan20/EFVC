import torch
from torch import nn

def make_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = kernel_size,
        stride = stride,
        padding = kernel_size // 2,
    )


def make_deconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size = kernel_size,
        stride = stride,
        output_padding = stride - 1,
        padding = kernel_size // 2,
    )


class ResidualUnit(nn.Module):
    """Simple residual unit"""

    def __init__(self, N: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            make_conv(N, N // 2, kernel_size = 1),
            nn.GELU(),
            make_conv(N // 2, N // 2, kernel_size = 3),
            nn.GELU(),
            make_conv(N // 2, N, kernel_size = 1),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out += identity
        out = self.activation(out)
        return out


def make_res_units(channels, layers = 3) -> nn.Sequential:
    return nn.Sequential(
        *[ResidualUnit(channels) for _ in range(layers)]
    )