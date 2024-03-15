import torch
from torch import nn
import torch.nn.functional as F

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


class CondConv(nn.Module):
    def __init__(self, 
        conv: nn.Module, out_channels = None,
        q_nums = 64, q_embed_dim = 64,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = conv.weight.shape[0]
        self.out_channels = out_channels

        self.conv = conv

        self.embed = nn.Embedding(q_nums, q_embed_dim)
        self.scale = nn.Linear(q_embed_dim, out_channels)
        self.shift = nn.Linear(q_embed_dim, out_channels)

    def forward(self, x, q_index):
        x = self.conv(x)

        B = x.shape[0]
        q_index_tensor = torch.tensor([[q_index]]).to(x.device)
        q_index_tensor = q_index_tensor.repeat_interleave(B, dim = 0)

        embeded_q = self.embed(q_index_tensor)
        scale = F.softplus(self.scale(embeded_q)).reshape(B, self.out_channels, 1, 1)
        shift = self.shift(embeded_q).reshape(B, self.out_channels, 1, 1)

        x = scale * x + shift

        return x


def make_res_units(channels, layers = 3) -> nn.Sequential:
    return nn.Sequential(
        *[ResidualUnit(channels) for _ in range(layers)]
    )