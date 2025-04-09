from torch import nn


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
