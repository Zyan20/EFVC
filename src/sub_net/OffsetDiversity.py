import torch
from torch import nn
from .video_net import bilinearupsacling, flow_warp

class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=64, aux_feature_num=64+3+2,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, 64, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(64, 3 * group_num * offset_num, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

    def forward(self, x, aux_feature, flow):
        B, C, H, W = x.shape
        out = self.conv_offset(aux_feature)
        out = bilinearupsacling(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        x = x.view(B * self.group_num, C // self.group_num, H, W)
        x = x.repeat(self.offset_num, 1, 1, 1)
        x = flow_warp(x, offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)
        x = self.fusion(x)

        return x