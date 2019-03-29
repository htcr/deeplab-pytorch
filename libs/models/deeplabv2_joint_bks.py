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


def test():
    model = DeepLabV2JointBKS(
        n_classes=2, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)
    bg = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)

    logits, mask = model(image, bg)
    print("logits:", logits.shape)
    print("mask:", mask.shape)
