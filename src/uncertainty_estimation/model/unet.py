from torch import nn
from torch.nn import functional as F

from . import layers as L


class UNet(nn.Module):
    def __init__(self, in_dims: int, out_dims: int = 1):
        super().__init__()
        self.down_conv1 = L.ConvBlock(in_dims, 64)
        self.down_conv2 = L.ConvBlock(64, 128)
        self.down_conv3 = L.ConvBlock(128, 256)

        self.up1 = L.UpscalingBlock(256, 128)
        self.up2 = L.UpscalingBlock(128, 64)
        self.up3 = L.UpscalingBlock(64, 32)

        self.out_conv = nn.Conv2d(32, out_dims, kernel_size=1)

    def forward(self, x):
        xd1 = F.max_pool2d(self.down_conv1(x), 2)
        xd2 = F.max_pool2d(self.down_conv2(xd1), 2)
        xd3 = F.max_pool2d(self.down_conv3(xd2), 2)

        xu1 = self.up1(xd3) + xd2
        xu2 = self.up2(xu1) + xd1
        xu3 = self.up3(xu2)

        x = self.out_conv(xu3)
        return x
