import math
import torch
from torch import mv, nn
from .sub_net.video_net import ME_Spynet, GDN, flow_warp, ResBlock, bilineardownsacling
from compressai.layers import subpel_conv3x3
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from .layers.ResBlock import InvertedResidual


def ResAct(channel, layers = 3):
    return nn.Sequential(
        *[InvertedResidual(channel) for _ in range(layers)]
    )


class FeatureExtractor(nn.Module):
    def __init__(self, channel = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = InvertedResidual(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = InvertedResidual(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = InvertedResidual(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in = 64, channel_out = 64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = InvertedResidual(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding = 1)
        self.res_block3_out = InvertedResidual(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = InvertedResidual(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding = 1)
        self.res_block2_out = InvertedResidual(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding = 1)
        self.res_block1_out = InvertedResidual(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class ContextualEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.gdn1 = ResAct(channel_N)
        self.res1 = InvertedResidual(channel_N * 2)
        
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.gdn2 = ResAct(channel_N)
        self.res2 = InvertedResidual(channel_N * 2)
        
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.gdn3 = ResAct(channel_N)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.gdn1(feature)
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.gdn2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.gdn1 = ResAct(channel_N)

        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.gdn2 = ResAct(channel_N)
        self.res1 = InvertedResidual(channel_N * 2)
                             
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.gdn3 = ResAct(channel_N)
        self.res2 = InvertedResidual(channel_N * 2)

        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.gdn1(feature)
        feature = self.up2(feature)
        feature = self.gdn2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.gdn3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class TemporalPriorEncoder(nn.Module):
    def __init__(self, channel_N = 64, channel_M = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1)
        self.gdn1 = ResAct(channel_N)

        self.conv2 = nn.Conv2d(channel_N * 2, channel_M, 3, stride=2, padding=1)
        self.gdn2 = ResAct(channel_M)

        self.conv3 = nn.Conv2d(channel_M + channel_N, channel_M * 3 // 2, 3, stride=2, padding=1)
        self.gdn3 = ResAct(channel_M * 3 // 2)

        self.conv4 = nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1)

    def forward(self, context1, context2, context3):
        feature = self.conv1(context1)
        feature = self.gdn1(feature)
        feature = self.conv2(torch.cat([feature, context2], dim=1))
        feature = self.gdn2(feature)
        feature = self.conv3(torch.cat([feature, context3], dim=1))
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1),
            InvertedResidual(channel, expand_ratio = 2),
            InvertedResidual(channel, expand_ratio = 2),
        )
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.feature_conv(torch.cat((ctx, res), dim=1))
        recon = self.recon_conv(feature)
        return feature, recon

def get_hyper_codec(channel_in, channel_out, expand_ratio = 1):
    encoder = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, 3, stride = 1, padding = 1),
        ResAct(channel_out),

        nn.Conv2d(channel_out, channel_out, 3, stride = 2, padding = 1),
        ResAct(channel_out),

        nn.Conv2d(channel_out, channel_out, 3, stride = 2, padding = 1),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(channel_out, channel_out, 3, stride = 2, padding = 1, output_padding = 1),
        ResAct(channel_out),

        nn.ConvTranspose2d(channel_out, channel_out, 3, stride = 2, padding = 1, output_padding = 1),
        ResAct(channel_out),

        nn.ConvTranspose2d(channel_out, channel_in * 2, 3, stride = 1, padding = 1),
    )

    return encoder, decoder

def get_codec(channel_in, channel_out, expand_ratio = 1):
    encoder = nn.Sequential(
        nn.Conv2d(channel_in, channel_out, 3, stride = 2, padding = 1),
        ResAct(channel_out),

        nn.Conv2d(channel_out, channel_out, 3, stride = 2, padding = 1),
        ResAct(channel_out),

        nn.Conv2d(channel_out, channel_out, 3, stride = 2, padding = 1),
        ResAct(channel_out),

        nn.Conv2d(channel_out, channel_out, 3, stride = 2, padding=1),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(channel_out, channel_out, 3, stride = 2, padding = 1, output_padding = 1),
        ResAct(channel_out),
        
        nn.ConvTranspose2d(channel_out, channel_out, 3, stride = 2, padding = 1, output_padding = 1),
        ResAct(channel_out),

        nn.ConvTranspose2d(channel_out, channel_out, 3, stride = 2, padding = 1, output_padding = 1),
        ResAct(channel_out),

        nn.ConvTranspose2d(channel_out, channel_in, 3, stride = 2, padding = 1, output_padding = 1),
    )

    return encoder, decoder


class EFVC(nn.Module):
    channel_mv = 128
    channel_N = 64
    channel_M = 96

    def __init__(self):
        super().__init__()

        self.optic_flow = ME_Spynet()

        self.mv_encoder, self.mv_decoder = get_codec(2, self.channel_mv)
        self.mv_prior_encoder, self.mv_prior_decoder = get_hyper_codec(self.channel_mv, self.channel_N)


        self.feature_adaptor_I = nn.Conv2d(3, self.channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(self.channel_N, self.channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.contextual_encoder = ContextualEncoder()


        self.contextual_hyper_prior_encoder, self.contextual_hyper_prior_decoder = get_hyper_codec(self.channel_M, self.channel_N)

        self.temporal_prior_encoder = TemporalPriorEncoder()

        self.contextual_entropy_parameter = nn.Sequential(
            InvertedResidual(self.channel_M * 4, self.channel_M * 4, expand_ratio = 1.5),
            InvertedResidual(self.channel_M * 4, self.channel_M * 4, expand_ratio = 2),
            nn.Conv2d(self.channel_M * 4, self.channel_M * 2, 3, stride = 1, padding = 1),
        )

        self.contextual_decoder = ContextualDecoder()

        self.recon_generation_net = ReconGeneration()

        self.entropy_coder = None
        self.bit_estimator_z = EntropyBottleneck(self.channel_N)
        self.bit_estimator_z_mv = EntropyBottleneck(self.channel_N)
        self.gaussian_encoder = GaussianConditional(None)


    def multi_scale_feature_extractor(self, ref, feature):
        if feature is None:
            feature = self.feature_adaptor_I(ref)
        else:
            feature = self.feature_adaptor_P(feature)
        return self.feature_extractor(feature)

    def motion_compensation(self, ref, feature, mv):
        warpframe = flow_warp(ref, mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref, feature)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe
    
    def _calc_bpp(self, likelihoods, num_pixels):
        return torch.sum(torch.clamp(-1.0 * torch.log(likelihoods + 1e-5) / math.log(2.0), 0, 50)) / num_pixels

    def forward(self, input_frame, ref_frame, ref_feature):
        est_mv = self.optic_flow(input_frame, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_z = self.mv_prior_encoder(mv_y)
        mv_z_hat, mv_z_likelihood = self.bit_estimator_z_mv(mv_z)
        mv_params = self.mv_prior_decoder(mv_z_hat)
        mv_scales_hat, mv_means_hat = mv_params.chunk(2, 1)

        mv_y_hat, mv_y_likelihood = self.gaussian_encoder(mv_y, scales = mv_scales_hat, means = mv_means_hat)
        mv_hat = self.mv_decoder(mv_y_hat)

        context1, context2, context3, warp_frame = self.motion_compensation(
            ref_frame, ref_feature, mv_hat)

        y = self.contextual_encoder(input_frame, context1, context2, context3)
        z = self.contextual_hyper_prior_encoder(y)

        z_hat, z_likelihood = self.bit_estimator_z(z)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)

        temporal_params = self.temporal_prior_encoder(context1, context2, context3)
        params = torch.cat((temporal_params, hierarchical_params), dim=1)
        gaussian_params = self.contextual_entropy_parameter(params)
        
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihood = self.gaussian_encoder(y, scales = scales_hat, means = means_hat)

        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        im_shape = input_frame.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = self._calc_bpp(y_likelihood, pixel_num)
        bpp_z = self._calc_bpp(z_likelihood, pixel_num)
        bpp_mv_y = self._calc_bpp(mv_y_likelihood, pixel_num)
        bpp_mv_z = self._calc_bpp(mv_z_likelihood, pixel_num)

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {
            "bpp_mv_y": bpp_mv_y,
            "bpp_mv_z": bpp_mv_z,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp,

            "recon_image": recon_image,
            "feature": feature,
            "warpped_image": warp_frame,


            "y": y,
            "z": z,
            "y_likeli": y_likelihood,

            "mv_y": mv_y,
            "mv_z": mv_z,
            "mv_y_likeli": mv_y_likelihood,
        }
