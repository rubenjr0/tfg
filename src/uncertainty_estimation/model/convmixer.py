import torch.nn as nn

from .layers import get_act


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(h_dims: int, depth: int, act: str, kernel_size=9, patch_size=7):
    return nn.Sequential(
        nn.Conv2d(32, h_dims, kernel_size=patch_size, stride=patch_size),
        get_act(act),
        nn.BatchNorm2d(h_dims),
        *[
            nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(
                            h_dims, h_dims, kernel_size, groups=h_dims, padding="same"
                        ),
                        get_act(act),
                        nn.BatchNorm2d(h_dims),
                    )
                ),
                nn.Conv2d(h_dims, h_dims, kernel_size=1),
                get_act(act),
                nn.BatchNorm2d(h_dims),
            )
            for _ in range(depth)
        ],
        nn.Conv2d(h_dims, 1, kernel_size=1),
    )
