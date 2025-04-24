import torch
from torch import nn
from .layers import ConvBlock




class ConvVAE(nn.Module):
    """
    A simple feedforward neural network for flow prediction.
    """

    def __init__(self, in_dims: int, latent_dims: int):
        super(ConvVAE, self).__init__()
        self.in_convs = nn.Sequential(
            ConvBlock(in_dims, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
        )

        self.mu = nn.Conv2d(64, latent_dims, 7)
        self.log_var = nn.Conv2d(64, latent_dims, 7)

        self.out_convs = nn.Sequential(
            ConvBlock(latent_dims, 64 * 4),
            nn.PixelShuffle(2),
            ConvBlock(64, 32 * 4),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        x = self.in_convs(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        z = eps * std + mu
        x = self.out_convs(z)
        return x
