import torch
from torch import nn
from torch.nn import functional as F

from . import layers as L


class UNet(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, act: str):
        super().__init__()
        self.down_conv1 = L.ConvBlock(in_dims, 64, act=act)
        self.down_conv2 = L.ConvBlock(64, 128, act=act)
        self.down_conv3 = L.ConvBlock(128, 256, act=act)
        self.down_conv4 = L.ConvBlock(256, 512, act=act)

        self.blottleneck = L.SeparableConv2d(512, 512)

        self.up1 = L.UpscalingBlock(512, 256, act=act)
        self.up2 = L.UpscalingBlock(256, 128, act=act)
        self.up3 = L.UpscalingBlock(128, 64, act=act)
        self.up4 = L.UpscalingBlock(64, 32, act=act)

        self.unc256 = L.SeparableConv2d(256, out_dims)
        self.unc128 = L.SeparableConv2d(128, out_dims)
        self.unc64 = L.SeparableConv2d(64, out_dims)
        self.unc32 = L.SeparableConv2d(32, out_dims)

        self.scale_weights = nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4]))

    def forward(self, x):
        xd1 = F.max_pool2d(self.down_conv1(x), 2)
        xd2 = F.max_pool2d(self.down_conv2(xd1), 2)
        xd3 = F.max_pool2d(self.down_conv3(xd2), 2)
        xd4 = F.max_pool2d(self.down_conv4(xd3), 2)
        xd4 = self.blottleneck(xd4)

        xu1 = self.up1(xd4) + xd3
        unc256 = self.unc256(xu1)

        xu2 = self.up2(xu1) + xd2
        unc128 = self.unc128(xu2)

        xu3 = self.up3(xu2) + xd1
        unc64 = self.unc64(xu3)

        xu4 = self.up4(xu3)
        unc32 = self.unc32(xu4)

        target_size = unc32.shape[-2:]
        unc256 = F.interpolate(
            unc256, target_size, mode="bilinear", align_corners=False
        )
        unc128 = F.interpolate(
            unc128, target_size, mode="bilinear", align_corners=False
        )
        unc64 = F.interpolate(unc64, target_size, mode="bilinear", align_corners=False)

        weights = F.softmax(self.scale_weights, dim=0)
        return (
            weights[0] * unc256
            + weights[1] * unc128
            + weights[2] * unc64
            + weights[3] * unc32
        )
