import torch

from .model import EFVC
from sub_net.OffsetDiversity import OffsetDiversity
from .sub_net.video_net import ME_Spynet, GDN, flow_warp, ResBlock, bilineardownsacling


class EFVC_OD(EFVC):
    """offset diversity"""
    def __init__(self):
        super(EFVC_OD, self).__init__()

        self.align = OffsetDiversity()

    
    def motion_compensation(self, ref, feature, mv):
        warpframe = flow_warp(ref, mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref, feature)
        context1_init = flow_warp(ref_feature1, mv)
        context1 = self.align(ref_feature1, torch.cat(
            (context1_init, warpframe, mv), dim=1), mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe
