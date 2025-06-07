import torch
from torch import nn
from torch.nn import functional as F

from . import layers as L


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, h_dims: int, depth: int, act: str, kernel_size=9, patch_size=7):
        super().__init__()
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
        )
        self.up1 = L.UpscalingBlock(h_dims, h_dims, act)
        self.skip1 = L.SeparableConv2d(h_dims, 1)
        self.up2 = L.UpscalingBlock(h_dims, h_dims, act)
        self.skip2 = L.SeparableConv2d(h_dims, 1)
        self.up3 = L.UpscalingBlock(h_dims, h_dims, act)
        self.skip3 = L.SeparableConv2d(h_dims, 1)
        self.scale_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

    def forward(self, rgb, depth, depth_edges, depth_laplacian):
        target_size = rgb.shape[-2:]
        depth_stack = torch.cat([depth, depth_edges, depth_laplacian], dim=1)
        rgb = self.rgb_encoder(rgb)
        depth_stack = self.stack_encoder(depth_stack)
        x = torch.cat([rgb, depth_stack], dim=1)
        x = self.mixer(x)
        up1 = self.up1(x)
        s1 = self.skip1(up1)
        up2 = self.up2(up1)
        s2 = self.skip2(up2)
        up3 = self.up3(up2)
        s3 = self.skip3(up3)
        s3 = F.interpolate(s3, target_size, mode="bilinear", align_corners=False)
        s2 = F.interpolate(s2, target_size, mode="bilinear", align_corners=False)
        s1 = F.interpolate(s1, target_size, mode="bilinear", align_corners=False)
        weights = F.softmax(self.scale_weights, dim=0)
        out = weights[0] * s1 + weights[1] * s2 + weights[2] * s3
        return out.clamp(-6, 6).exp()


if __name__ == "__main__":
    rgb = torch.rand(16, 3, 256, 256)
    depth = torch.rand(16, 1, 256, 256)
    depth_edges = torch.rand(16, 1, 256, 256)
    depth_laplacian = torch.rand(16, 1, 256, 256)
    mixer = ConvMixer(64, 12, "siren")
    z = mixer(rgb, depth, depth_edges, depth_laplacian)
    print(z.shape)
