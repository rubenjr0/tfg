import torch
import torch.nn as nn

from . import layers as L


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, h_dims: int, depth: int, act: str, kernel_size=9, patch_size=7):
        self.rgb_encoder = L.Encoder(in_dims=3, out_dims=16, act=act)
        self.stack_encoder = L.Encoder(in_dims=3, out_dims=16, act=act)
        self.mixer = nn.Sequential(
            nn.Conv2d(32, h_dims, kernel_size=patch_size, stride=patch_size),
            L.get_act(act),
            nn.BatchNorm2d(h_dims),
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                h_dims,
                                h_dims,
                                kernel_size,
                                groups=h_dims,
                                padding="same",
                            ),
                            L.get_act(act),
                            nn.BatchNorm2d(h_dims),
                        )
                    ),
                    nn.Conv2d(h_dims, h_dims, kernel_size=1),
                    L.get_act(act),
                    nn.BatchNorm2d(h_dims),
                )
                for _ in range(depth)
            ],
            nn.Conv2d(h_dims, 1, kernel_size=1),
        )

    def forward(self, rgb, depth, depth_edges, depth_laplacian):
        depth_stack = torch.cat([depth, depth_edges, depth_laplacian], dim=1)
        rgb = self.rgb_encoder(rgb)
        depth_stack = self.stack_encoder(depth_stack)
        x = torch.cat([rgb, depth_stack], dim=1)
        out = self.mixer(x)
        return out.clamp(-6, 6).exp()
