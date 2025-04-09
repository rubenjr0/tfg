import torch
from torch import nn

from . import layers as L


class VAE(nn.Module):
    def __init__(self, in_dims: int, out_dims: int = 1):
        super().__init__()
        self.down_conv1 = L.ConvBlock(in_dims, 64)
        self.down_conv2 = L.ConvBlock(64, 128)
        self.down_conv3 = L.ConvBlock(128, 256)
        self.down_conv4 = L.ConvBlock(256, 512)
        self.down_conv5 = L.ConvBlock(512, 768)

        self.mu = nn.Conv2d(768, 1024, kernel_size=1)
        self.logvar = nn.Conv2d(768, 1024, kernel_size=1)

        self.downcast = nn.Conv2d(1024, 768, kernel_size=1)

        self.up1 = L.UpscalingBlock(768, 512)
        self.up2 = L.UpscalingBlock(512, 256)
        self.up3 = L.UpscalingBlock(256, 128)
        self.up4 = L.UpscalingBlock(128, 64)
        self.up5 = L.UpscalingBlock(64, 32)
        self.out_conv = nn.Conv2d(32, out_dims, kernel_size=1)

    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x3 = self.down_conv3(x2)
        x4 = self.down_conv4(x3)
        x5 = self.down_conv5(x4)

        mu = self.mu(x5)
        logvar = self.logvar(x5)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x = mu + eps * std
        x = self.downcast(x)

        x = self.up1(x) + x4
        x = self.up2(x) + x3
        x = self.up3(x) + x2
        x = self.up4(x) + x1
        x = self.up5(x)

        x = self.out_conv(x)
        return x
