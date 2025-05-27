import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_dims, in_dims, kernel_size, padding="same", groups=in_dims
        )
        self.pointwise = nn.Conv2d(in_dims, out_dims, 1)

    def forward(self, x):
        x = self.depthwise(x) + x
        x = self.pointwise(x)
        return x


class SeparableConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_dims: int,
        out_dims: int,
    ):
        super(SeparableConvTranspose2d, self).__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_dims,
            in_dims,
            kernel_size=3,
            stride=2,
            output_padding=1,
            padding=1,
            groups=in_dims,
        )
        self.pointwise = nn.Conv2d(in_dims, out_dims, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super(ConvBlock, self).__init__()
        self.conv = SeparableConv2d(in_dims, out_dims)
        self.norm = nn.InstanceNorm2d(out_dims)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
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
            groups=4,
            kernel_size=1,
        )
        self.out_conv = SeparableConv2d(
            in_dims,
            out_dims,
        )
        self.norm = nn.InstanceNorm2d(out_dims)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        res = self.up_res(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.downsample(x) + res
        x = self.out_conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    """
    Takes (N, C, H, W) and returns encoded features to be merged with other features.
    """

    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.conv1 = SeparableConv2d(in_dims, out_dims)
        self.norm1 = nn.InstanceNorm2d(out_dims)
        self.act1 = nn.GELU()
        self.conv2 = SeparableConv2d(out_dims, out_dims)
        self.norm2 = nn.InstanceNorm2d(out_dims)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x
