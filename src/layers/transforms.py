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
        q_nums = 64, reverse = False
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = conv.weight.shape[0]

        self.out_channels = out_channels

        self.conv = conv
        self.q_nums = q_nums
        self.reverse = reverse

        self.scale = nn.Sequential(
            nn.Linear(q_nums, out_channels),
            nn.LeakyReLU(inplace = True),
            nn.Linear(out_channels, int(out_channels)),
        )
        # self.shift = nn.Sequential(
        #     nn.Linear(q_nums, out_channels),
        #     nn.LeakyReLU(inplace = True),
        #     nn.Linear(out_channels, out_channels),
        # )

    def forward(self, x, q_index):
        B = x.shape[0]
        q_index_tensor = torch.tensor([[q_index]]).to(x.device)
        embeded_q = F.one_hot(q_index_tensor, self.q_nums).float().expand(B, 1, self.q_nums)    # [B, 1, q_embed_dim]
        scale = F.softplus(self.scale(embeded_q)).view(B, self.out_channels, 1, 1)

        if self.reverse:
            x = scale * x
            x = self.conv(x)

        else:
            x = self.conv(x)
            x = scale * x

        return x

    


def make_res_units(channels, layers = 3) -> nn.Sequential:
    return nn.Sequential(
        *[ResidualUnit(channels) for _ in range(layers)]
    )