from torch import nn
from torch.nn import functional as F

from .separable_convolution import SeparableConv2d, SeparableConvTranspose2d


class ConvBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.conv = SeparableConv2d(in_dims, out_dims)
        self.norm = nn.InstanceNorm2d(out_dims)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = F.mish(x)
        x = self.drop(x)
        x = F.max_pool2d(x, kernel_size=2)
        return x


class UpscalingBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.upsample = nn.Conv2d(in_dims, in_dims * 4, kernel_size=1)
        self.up_res = SeparableConvTranspose2d(
            in_dims * 4,
            in_dims,
        )
        self.downsample = nn.Conv2d(
            in_dims * 4,
            in_dims,
            kernel_size=1,
        )
        self.out_conv = SeparableConv2d(
            in_dims,
            out_dims,
        )
        self.norm = nn.InstanceNorm2d(out_dims)

    def forward(self, x):
        x = self.upsample(x)
        res = self.up_res(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.downsample(x) + res
        x = self.out_conv(x)
        x = self.norm(x)
        x = F.mish(x)
        return x
