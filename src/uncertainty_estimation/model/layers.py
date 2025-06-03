import torch
from torch import nn
from torch.nn import functional as F


def get_act(act: str):
    return (
        nn.GELU()
        if act == "gelu"
        else nn.SiLU()
        if act == "silu"
        else nn.Mish()
        if act == "mish"
        else nn.ReLU()
        if act == "relu"
        else ValueError(f"Unknown activation function: {act}")
    )


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


class SpatialAttention(nn.Module):
    def __init__(self, in_dims: int, act: str):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(in_dims, in_dims // 8, 1),
            get_act(act),
            nn.Conv2d(in_dims // 8, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.att(x)


class ConvBlock(nn.Module):
    def __init__(
        self, in_dims: int, out_dims: int, act: str, stochastic_prob: float = 0.0
    ):
        super(ConvBlock, self).__init__()
        self.conv = SeparableConv2d(in_dims, out_dims)
        self.norm = nn.BatchNorm2d(out_dims)
        self.act = get_act(act)
        self.drop = nn.Dropout2d(0.2)
        self.stochastic_prob = stochastic_prob
        self.shortcut = (
            nn.Identity() if in_dims == out_dims else nn.Conv2d(in_dims, out_dims, 1)
        )

    def forward(self, x):
        id = self.shortcut(x)
        if self.training and self.stochastic_prob > 0.0:
            if torch.rand(1) < self.stochastic_prob:
                return id
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x + id


class UpscalingBlock(nn.Module):
    def __init__(self, in_dims: int, out_dims: int, act: str):
        super().__init__()
        self.shuffle_proj = SeparableConv2d(in_dims, out_dims * 4)
        self.bil_proj = SeparableConv2d(in_dims, out_dims)
        self.shuffle = nn.PixelShuffle(2)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.BatchNorm2d(out_dims)
        self.act = get_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bil = self.bil_proj(x)
        bil = F.interpolate(bil, scale_factor=2, mode="bilinear", align_corners=False)

        shuffle = self.shuffle_proj(x)
        shuffle = self.shuffle(shuffle)

        x = self.alpha * shuffle + (1 - self.alpha) * bil
        x = self.norm(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    """
    Takes (N, C, H, W) and returns encoded features to be merged with other features.
    """

    def __init__(self, in_dims: int, out_dims: int, act: str):
        super().__init__()
        self.conv1 = SeparableConv2d(in_dims, out_dims)
        self.norm1 = nn.BatchNorm2d(out_dims)
        self.conv2 = SeparableConv2d(out_dims, out_dims)
        self.norm2 = nn.BatchNorm2d(out_dims)
        self.act = get_act(act)
        self.dropout = nn.Dropout2d(0.1)
        self.att = SpatialAttention(out_dims, act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.att(x)
        return x
