import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(
    in_dims: int, h_dims: int, out_dims: int, depth: int, kernel_size=9, patch_size=7
):
    return nn.Sequential(
        nn.Conv2d(in_dims, h_dims, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(h_dims),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            h_dims, h_dims, kernel_size, groups=h_dims, padding="same"
                        ),
                        nn.GELU(),
                        nn.BatchNorm2d(h_dims),
                    )
                ),
                nn.Conv2d(h_dims, h_dims, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(h_dims),
            )
            for _ in range(depth)
        ],
        nn.Conv2d(h_dims, out_dims, kernel_size=1),
    )
