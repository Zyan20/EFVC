import torch
from torch import nn

from .model import EFVC
from .sub_net.video_net import ResBlock, bilineardownsacling


class Warpper(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.warpper = nn.Sequential(
            nn.Conv2d(64 + 2, 64, 5, 1, 2),
            ResBlock(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            ResBlock(64)
        )

    def forward(self, x):
        return self.warpper(x)



class EFVC_Destilation(EFVC):
    """offset diversity"""
    def __init__(self):
        super(EFVC_Destilation, self).__init__()

        self.warpper = Warpper()

    
    def motion_compensation(self, ref, feature, mv):
        # warpframe = flow_warp(ref, mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref, feature)

        context1 = self.warpper(torch.cat([ref_feature1, mv], dim = 1))
        context2 = self.warpper(torch.cat([ref_feature2, mv2], dim = 1))
        context3 = self.warpper(torch.cat([ref_feature3, mv3], dim = 1))
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, None
    
    