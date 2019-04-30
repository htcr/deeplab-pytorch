#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer

import numpy as np

from guided_filter_pytorch.guided_filter import FastGuidedFilter

class _Stem(nn.Sequential):
    """
    The 1st Residual Layer.
    Note that the max pooling is different from both MSRA and FAIR ResNet.
    """

    def __init__(self):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(6, 64, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class _UPSAMPLE(nn.Module):
    """
    Upsampling for finer mask
    """
    
    def __init__(self, in_ch):
        super(_UPSAMPLE, self).__init__()
        """
        self.deconv1 = nn.ConvTranspose2d(in_ch, 128, kernel_size=5, padding=2, stride=2, bias=True)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2, stride=2, bias=True)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=5, padding=2, stride=2, bias=True)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=5, padding=2, bias=True)
        """
        self.deconv1 = nn.Conv2d(in_ch, 128, kernel_size=5, padding=2, stride=1, bias=True)
        self.deconv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1, bias=True)
        self.deconv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1, bias=True)
        self.conv_out = nn.Conv2d(32, 1, kernel_size=5, padding=2, stride=1, bias=True)

    def forward(self, x, comp_fg, bg):
        x = F.upsample(x, scale_factor=2.0, mode='bilinear')
        x = F.relu(self.deconv1(x))
        
        x = F.upsample(x, scale_factor=2.0, mode='bilinear')
        x = F.relu(self.deconv2(x))

        x = F.upsample(x, scale_factor=2.0, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.conv_out(x)
        
        # mask = F.sigmoid(x)
        return x


class _UPSAMPLE_CASCADE(nn.Module):
    """
    Upsampling for finer mask
    """
    
    def __init__(self, in_ch):
        super(_UPSAMPLE_CASCADE, self).__init__()
        self.trans1 = nn.Conv2d(7, 256, kernel_size=5, padding=2, stride=1, bias=True)
        self.deconv1 = nn.Conv2d(256, 128, kernel_size=5, padding=2, stride=1, bias=True)
        self.pred1 = nn.Conv2d(128, 1, kernel_size=5, padding=2, stride=1, bias=True)

        self.trans2 = nn.Conv2d(7, 128, kernel_size=5, padding=2, stride=1, bias=True)
        self.deconv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2, stride=1, bias=True)
        self.pred2 = nn.Conv2d(64, 1, kernel_size=5, padding=2, stride=1, bias=True)

        self.trans3 = nn.Conv2d(7, 64, kernel_size=5, padding=2, stride=1, bias=True)
        self.deconv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2, stride=1, bias=True)
        self.pred3 = nn.Conv2d(32, 1, kernel_size=5, padding=2, stride=1, bias=True)


    def forward(self, mask8, f8, fg, bg):
        fg = fg / 128.0
        bg = bg / 128.0
        
        mask4_raw = F.upsample(mask8, scale_factor=2.0, mode='bilinear')
        _, _, h4, w4 = mask4_raw.shape
        fg4 = F.interpolate(fg, (h4, w4), mode='bilinear')
        bg4 = F.interpolate(bg, (h4, w4), mode='bilinear')
        cat4 = torch.cat((mask4_raw, fg4, bg4), dim=1)
        cat4trans = self.trans1(cat4)
        f4_raw = F.upsample(f8, (h4, w4), mode='bilinear')

        f4 = F.relu(self.deconv1(f4_raw + cat4trans))
        mask4 = F.sigmoid(self.pred1(f4))


        mask2_raw = F.upsample(mask4, scale_factor=2.0, mode='bilinear')
        _, _, h2, w2 = mask2_raw.shape
        fg2 = F.interpolate(fg, (h2, w2), mode='bilinear')
        bg2 = F.interpolate(bg, (h2, w2), mode='bilinear')
        cat2 = torch.cat((mask2_raw, fg2, bg2), dim=1)
        cat2trans = self.trans2(cat2)
        f2_raw = F.upsample(f4, (h2, w2), mode='bilinear')

        f2 = F.relu(self.deconv2(f2_raw + cat2trans))
        mask2 = F.sigmoid(self.pred2(f2))
        

        mask1_raw = F.upsample(mask2, scale_factor=2.0, mode='bilinear')
        _, _, h1, w1 = mask1_raw.shape
        fg1 = F.interpolate(fg, (h1, w1), mode='bilinear')
        bg1 = F.interpolate(bg, (h1, w1), mode='bilinear')
        cat1 = torch.cat((mask1_raw, fg1, bg1), dim=1)
        cat1trans = self.trans3(cat1)
        f1_raw = F.upsample(f2, (h1, w1), mode='bilinear')

        f1 = F.relu(self.deconv3(f1_raw + cat1trans))
        mask1 = F.sigmoid(self.pred3(f1))
        
        return [mask4, mask2, mask1]


dump_feature = False


class GuidedUpsampleUnit(nn.Module):
    def __init__(self, embed_size):
        super(GuidedUpsampleUnit, self).__init__()
        self.embed_size = embed_size
        self.conv1 = nn.Conv2d(self.embed_size+7, self.embed_size, kernel_size=1)
        self.conv2 = nn.Conv2d(self.embed_size, self.embed_size, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.embed_size, self.embed_size, kernel_size=1)
        self.mask_pred = nn.Conv2d(self.embed_size, 1, kernel_size=1)

    def forward(self, fg, bg, mask, feat):
        # fg, bg shall be normalized
        # feat shall be transformed to embed_size
        mask_2x_raw = F.upsample(mask, scale_factor=2.0, mode='bilinear')
        _, _, h2, w2 = mask_2x_raw.shape
        feat_2x_raw = F.interpolate(feat, (h2, w2), mode='bilinear')
        fg_2x = F.interpolate(fg, (h2, w2), mode='bilinear')
        bg_2x = F.interpolate(bg, (h2, w2), mode='bilinear')

        x = torch.cat((mask_2x_raw, fg_2x, bg_2x, feat_2x_raw), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feat_2x = F.relu(self.conv3(x))
        
        mask_2x = F.sigmoid(self.mask_pred(feat_2x))

        return mask_2x, feat_2x


from cv2.ximgproc import guidedFilter
import matplotlib.pyplot as plt

class GuidedUpsampleUnitGF(nn.Module):
    def __init__(self, embed_size):
        super(GuidedUpsampleUnitGF, self).__init__()
        self.embed_size = embed_size
        self.conv1 = nn.Conv2d(self.embed_size+7, self.embed_size, kernel_size=1)
        self.conv2 = nn.Conv2d(self.embed_size, self.embed_size, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.embed_size, self.embed_size, kernel_size=1)
        self.mask_pred = nn.Conv2d(self.embed_size, 1, kernel_size=1)

    def forward(self, fg, bg, mask, feat):
        # fg, bg shall be normalized
        # feat shall be transformed to embed_size
        mask_2x_raw = F.upsample(mask, scale_factor=2.0, mode='bilinear')
        _, _, h2, w2 = mask_2x_raw.shape
        feat_2x_raw = F.interpolate(feat, (h2, w2), mode='bilinear')
        fg_2x = F.interpolate(fg, (h2, w2), mode='bilinear')
        bg_2x = F.interpolate(bg, (h2, w2), mode='bilinear')

        x = torch.cat((mask_2x_raw, fg_2x, bg_2x, feat_2x_raw), dim=1)
        x = F.relu(self.conv1(x))
        feat_2x = F.relu(self.conv2(x))
        
        ## apply guided filter on feature map
        guide_fg_np = torch.squeeze(torch.mean(fg_2x, dim=1), dim=0).detach().cpu().numpy().astype(np.float32)*128.0
        feat_2x_np = torch.squeeze(feat_2x, dim=0).detach().cpu().numpy().astype(np.float32)
        filtered_channels = list()
        _, C, H, W = feat_2x.shape
        for cid in range(C):
            # print('applying GF')
            channel = feat_2x_np[cid, :, :]
            filtered_channel = np.zeros((H, W), dtype=np.float32)
            guidedFilter(guide=guide_fg_np, src=channel, dst=filtered_channel, radius=int(0.01*H), eps=0.01)
            # plt.imshow(channel)
            # plt.show()
            # plt.imshow(filtered_channel)
            # plt.show()
            filtered_channels.append(filtered_channel)
        filtered_feat_2x_np = np.stack(filtered_channels, axis=0)
        filtered_feat_2x = torch.Tensor(filtered_feat_2x_np).to(fg_2x.get_device())
        filtered_feat_2x = torch.unsqueeze(filtered_feat_2x, dim=0)
        feat_2x = filtered_feat_2x    


        feat_2x = F.relu(self.conv3(feat_2x))
        mask_2x = F.sigmoid(self.mask_pred(feat_2x))

        return mask_2x, feat_2x


class GuidedUpsampleUnitGFFast(nn.Module):
    def __init__(self, embed_size):
        super(GuidedUpsampleUnitGFFast, self).__init__()
        self.embed_size = embed_size
        self.conv1 = nn.Conv2d(self.embed_size+7, self.embed_size, kernel_size=1)
        self.conv2 = nn.Conv2d(self.embed_size, self.embed_size, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(self.embed_size, self.embed_size, kernel_size=1)
        self.mask_pred = nn.Conv2d(self.embed_size, 1, kernel_size=1)

    def forward(self, fg, bg, mask, feat):
        # fg, bg shall be normalized
        # feat shall be transformed to embed_size
        mask_2x_raw = F.upsample(mask, scale_factor=2.0, mode='bilinear')
        _, _, h2, w2 = mask_2x_raw.shape
        feat_2x_raw = F.interpolate(feat, (h2, w2), mode='bilinear')
        fg_2x = F.interpolate(fg, (h2, w2), mode='bilinear')
        bg_2x = F.interpolate(bg, (h2, w2), mode='bilinear')

        x = torch.cat((mask_2x_raw, fg_2x, bg_2x, feat_2x_raw), dim=1)
        x = F.relu(self.conv1(x))
        feat_2x = F.relu(self.conv2(x))
        
        _, C, H, W = feat_2x.shape
        guide_fg = torch.mean(fg_2x, dim=1, keepdim=True)*128.0
        feat_2x = FastGuidedFilter(r=int(0.01*H), eps=0.01)(guide_fg, feat_2x, guide_fg)

        feat_2x = F.relu(self.conv3(feat_2x))
        mask_2x = F.sigmoid(self.mask_pred(feat_2x))

        return mask_2x, feat_2x
        

class CascadedUpsampler(nn.Module):
    def __init__(self, input_feature_size, embed_size):
        super(CascadedUpsampler, self).__init__()
        self.input_feature_size = input_feature_size
        self.embed_size = embed_size
        
        self.input_feat_trans = nn.Conv2d(self.input_feature_size, self.embed_size, kernel_size=3, padding=1)
        
        self.upsample = GuidedUpsampleUnit(self.embed_size)
        
    def forward(self, fg, bg, init_mask, init_feat):
        fg = fg / 128.0
        bg = bg / 128.0
        
        init_feat = F.relu(self.input_feat_trans(init_feat))
        mask_2x, feat_2x = self.upsample(fg, bg, init_mask, init_feat)
        mask_4x, feat_4x = self.upsample(fg, bg, mask_2x, feat_2x)
        mask_8x, feat_8x = self.upsample(fg, bg, mask_4x, feat_4x)

        """
        if dump_feature:
            feat_8x_np = feat_8x.detach().cpu().numpy()
            print('saving feature of shape: ' + str(feat_8x_np.shape))
            np.save('feat_8x.npy', feat_8x_np)
        """ 

        return [mask_2x, mask_4x, mask_8x]


class CascadedUpsamplerGF(nn.Module):
    def __init__(self, input_feature_size, embed_size):
        super(CascadedUpsamplerGF, self).__init__()
        self.input_feature_size = input_feature_size
        self.embed_size = embed_size
        
        self.input_feat_trans = nn.Conv2d(self.input_feature_size, self.embed_size, kernel_size=3, padding=1)
        
        self.upsample = GuidedUpsampleUnitGFFast(self.embed_size)
        
    def forward(self, fg, bg, init_mask, init_feat):
        fg = fg / 128.0
        bg = bg / 128.0
        
        init_feat = F.relu(self.input_feat_trans(init_feat))
        mask_2x, feat_2x = self.upsample(fg, bg, init_mask, init_feat)
        mask_4x, feat_4x = self.upsample(fg, bg, mask_2x, feat_2x)
        mask_8x, feat_8x = self.upsample(fg, bg, mask_4x, feat_4x)

        """
        if dump_feature:
            feat_8x_np = feat_8x.detach().cpu().numpy()
            print('saving feature of shape: ' + str(feat_8x_np.shape))
            np.save('feat_8x.npy', feat_8x_np)
        """ 

        return [mask_2x, mask_4x, mask_8x]  
        

class DeepLabV2JointBKS(nn.Module):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2JointBKS, self).__init__()
        self.layer1 = _Stem()
        self.layer2 = _ResLayer(n_blocks[0], 64, 64, 256, 1, 1)
        self.layer3 = _ResLayer(n_blocks[1], 256, 128, 512, 2, 1)
        self.layer4 = _ResLayer(n_blocks[2], 512, 256, 1024, 1, 2)
        self.layer5 = _ResLayer(n_blocks[3], 1024, 512, 2048, 1, 4)

        self.aspp = _ASPP(2048, n_classes, atrous_rates)

        matting_feature_channel = 256

        self.feature_trans = nn.Conv2d(2048, matting_feature_channel, kernel_size=3, padding=1, bias=True)
        #self.logit_trans = nn.Conv2d(2, matting_feature_channel, kernel_size=3, padding=1, bias=True)

        self.upsample = _UPSAMPLE_CASCADE(matting_feature_channel)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def forward(self, fgs, bgs):
        x = torch.cat((fgs, bgs), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = self.layer5(x)
        
        logits = self.aspp(feature)

        matting_feature = self.feature_trans(feature)
        # logit_feature = self.logit_trans(logits)

        #matting_feature = F.relu(matting_feature + logit_feature)
        
        probs = F.softmax(logits, dim=1)
        human_probs = probs[:, 1:, :, :]

        mask8 = F.sigmoid(human_probs)
        cascade_masks = self.upsample(mask8, matting_feature, fgs, bgs)

        return logits, cascade_masks


class DeepLabV2JointBKSV2(nn.Module):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2JointBKSV2, self).__init__()
        self.layer1 = _Stem()
        self.layer2 = _ResLayer(n_blocks[0], 64, 64, 256, 1, 1)
        self.layer3 = _ResLayer(n_blocks[1], 256, 128, 512, 2, 1)
        self.layer4 = _ResLayer(n_blocks[2], 512, 256, 1024, 1, 2)
        self.layer5 = _ResLayer(n_blocks[3], 1024, 512, 2048, 1, 4)

        self.aspp = _ASPP(2048, n_classes, atrous_rates)

        self.upsample = CascadedUpsamplerGF(2048, 32)
        # self.upsample = CascadedUpsamplerRecurrent(2048, 32)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def forward(self, fgs, bgs):

        _, _, inH, inW = fgs.shape

        x = torch.cat((fgs, bgs), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = self.layer5(x)
        
        logits = self.aspp(feature)

        probs = F.softmax(logits, dim=1)
        human_probs = probs[:, 1:, :, :]

        # mask8 = F.sigmoid(human_probs)
        mask8 = human_probs
        cascade_masks = self.upsample(fgs, bgs, mask8.detach(), feature)

        # try to fix misalignment
        logits_out = F.interpolate(logits, (inH, inW))
        cascade_masks_out = [F.interpolate(mask, (inH, inW)) for mask in cascade_masks]

        return logits_out, cascade_masks_out


class DeepLabV2JointBKSV2GF(nn.Module):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2JointBKSV2GF, self).__init__()
        self.layer1 = _Stem()
        self.layer2 = _ResLayer(n_blocks[0], 64, 64, 256, 1, 1)
        self.layer3 = _ResLayer(n_blocks[1], 256, 128, 512, 2, 1)
        self.layer4 = _ResLayer(n_blocks[2], 512, 256, 1024, 1, 2)
        self.layer5 = _ResLayer(n_blocks[3], 1024, 512, 2048, 1, 4)

        self.aspp = _ASPP(2048, n_classes, atrous_rates)

        self.upsample = CascadedUpsampler(2048, 32)
        self.upsample_gf = CascadedUpsamplerGF(2048, 32)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def load_state_dict(self, state_dict):
        full_state_dict = dict()
        for key, val in state_dict.items():
            print(key)
            full_state_dict[key] = val
            key_spt = key.split('.')
            if key_spt[0] == 'upsample':
                key_spt[0] = 'upsample_gf'
                new_key = '.'.join(key_spt)
                full_state_dict[new_key] = val
                print(new_key)
        super(DeepLabV2JointBKSV2GF, self).load_state_dict(full_state_dict)


    def forward(self, fgs, bgs):

        _, _, inH, inW = fgs.shape

        x = torch.cat((fgs, bgs), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feature = self.layer5(x)
        
        logits = self.aspp(feature)

        probs = F.softmax(logits, dim=1)
        human_probs = probs[:, 1:, :, :]

        # mask8 = F.sigmoid(human_probs)
        mask8 = human_probs
        cascade_masks = self.upsample(fgs, bgs, mask8.detach(), feature)
        cascade_masks_gf = self.upsample_gf(fgs, bgs, mask8.detach(), feature)

        logits_out = F.interpolate(logits, (inH, inW))

        out_masks = [torch.max(cascade_masks[i], cascade_masks_gf[i]) for i in range(len(cascade_masks))]

        cascade_masks_out = [F.interpolate(mask, (inH, inW)) for mask in out_masks]

        return logits_out, cascade_masks_out


def test():
    model = DeepLabV2JointBKSV2(
        n_classes=2, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)
    bg = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)

    logits, masks = model(image, bg)
    print("logits:", logits.shape)
    for mask in masks:
        print("mask:", mask.shape)
