from torch import nn
from torch.nn import functional as F

from . import layers as L


class UNet(nn.Module):
    def __init__(self, in_dims: int, out_dims: int = 1):
        super().__init__()
        self.down_conv1 = L.ConvBlock(in_dims, 64)
        self.down_conv2 = L.ConvBlock(64, 128)
        self.down_conv3 = L.ConvBlock(128, 256)
        self.down_conv4 = L.ConvBlock(256, 512)

        self.up1 = L.UpscalingBlock(512, 256)
        self.up2 = L.UpscalingBlock(256, 128)
        self.up3 = L.UpscalingBlock(128, 64)
        self.up4 = L.UpscalingBlock(64, 32)

        self.out_conv = nn.Conv2d(32, out_dims, kernel_size=1)

    def forward(self, x):
        xd1 = F.max_pool2d(self.down_conv1(x), 2)
        xd2 = F.max_pool2d(self.down_conv2(xd1), 2)
        xd3 = F.max_pool2d(self.down_conv3(xd2), 2)
        xd4 = F.max_pool2d(self.down_conv4(xd3), 2)

        xu1 = self.up1(xd4) + xd3
        xu2 = self.up2(xu1) + xd2
        xu3 = self.up3(xu2) + xd1
        xu4 = self.up4(xu3)

        x = self.out_conv(xu4)
        return x
