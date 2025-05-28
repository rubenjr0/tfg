import torch
from torch import nn
from torch.nn import functional as F

from .layers import ConvBlock, UpscalingBlock


class ConvVAE(nn.Module):
    """
    A simple feedforward neural network for flow prediction.
    """

    def __init__(
        self,
        in_dims: int,
        latent_dims: int = 256,
        out_dims: int = 1,
        min_size: int = 32,
    ):
        super(ConvVAE, self).__init__()
        self.in_dims = in_dims

        # Encoder
        self.down_conv1 = ConvBlock(in_dims, 64)  # 256x256, 64
        self.down_conv2 = ConvBlock(64, 128)  # 128x128, 128
        self.down_conv3 = ConvBlock(128, 256)  # 64x64, 256

        # Bottleneck
        self.bottleneck_size = min_size
        bottleneck_dims = 256 * self.bottleneck_size * self.bottleneck_size
        self.mu = nn.Linear(bottleneck_dims, latent_dims)
        self.log_var = nn.Linear(bottleneck_dims, latent_dims)
        self.decode_proj = nn.Linear(latent_dims, bottleneck_dims)

        # Decoder
        self.up1 = UpscalingBlock(256, 128)  # 32x32 -> 64x64
        self.up2 = UpscalingBlock(128, 64)  # 64x64 -> 128x128
        self.up3 = UpscalingBlock(64, 32)  # 128x128 -> 256x256

        # Multiscale heads
        self.unc_64 = nn.Conv2d(128, out_dims, 1)
        self.unc_128 = nn.Conv2d(64, out_dims, 1)
        self.unc_256 = nn.Conv2d(32, out_dims, 1)

        # skips
        self.skip1 = nn.Conv2d(64, 32, 1)
        self.skip2 = nn.Conv2d(128, 64, 1)

        self.scale_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)

        # encoder
        down1 = self.down_conv1(x)
        down2 = self.down_conv2(F.max_pool2d(down1, 2))
        bottleneck = self.down_conv3(F.max_pool2d(down2, 4))

        # bottleneck
        flat = bottleneck.flatten(1)
        mu = self.mu(flat)
        log_var = self.log_var(flat)
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        z = eps * std + mu
        dec = self.decode_proj(z).view(
            batch_size, 256, self.bottleneck_size, self.bottleneck_size
        )

        # decoder
        up1 = self.up1(dec)
        unc_64 = self.unc_64(up1)

        up2 = self.up2(up1) + self.skip2(down2)
        unc_128 = self.unc_128(up2)

        up3 = self.up3(up2) + self.skip1(down1)
        unc_256 = self.unc_256(up3)

        # multi-scale
        target_size = unc_256.shape[2:]
        unc_64_up = F.interpolate(
            unc_64, target_size, mode="bilinear", align_corners=False
        )
        unc_128_up = F.interpolate(
            unc_128, target_size, mode="bilinear", align_corners=False
        )

        weights = F.softmax(self.scale_weights, dim=0)
        z = weights[0] * unc_64_up + weights[1] * unc_128_up + weights[2] * unc_256
        return z, mu, log_var

    def kl_div(self, mu, log_var, free_bits=2.0):
        log_var = torch.clamp(log_var, -6, 6)
        kl = -0.5 * (1 + log_var - mu**2 - log_var.exp())
        kl = torch.clamp(kl - free_bits / mu.size(-1), 0)
        return kl.sum(dim=1).mean()
