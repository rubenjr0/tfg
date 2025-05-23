import torch
from torch import nn
from .layers import ConvBlock


class ConvVAE(nn.Module):
    """
    A simple feedforward neural network for flow prediction.
    """

    def __init__(self, in_dims: int, latent_dims: int, out_dims: int = 1):
        super(ConvVAE, self).__init__()
        self.MIN_SIZE = 16
        self.in_dims = in_dims
        self.in_convs = nn.Sequential(
            ConvBlock(in_dims, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((self.MIN_SIZE, self.MIN_SIZE)),
            nn.Flatten(),
        )

        self.mu = nn.Linear(128 * self.MIN_SIZE**2, latent_dims)
        self.log_var = nn.Linear(128 * self.MIN_SIZE**2, latent_dims)
        self.out_conv = nn.Conv2d(
            latent_dims // (self.MIN_SIZE**2), out_dims, kernel_size=1
        )

    def forward(self, x):
        n_batches = x.size(0)
        x = self.in_convs(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        z = eps * std + mu
        z = z.view(n_batches, -1, self.MIN_SIZE, self.MIN_SIZE)
        z = self.out_conv(z)
        return z
