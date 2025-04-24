import rerun as rr
import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from .layers import Encoder, SeparableConv2d
from .conv_vae import ConvVAE


class UncertaintyEstimator(LightningModule):
    def __init__(self, in_dims: int, out_dims: int = 1, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.rgb_encoder = Encoder(3, 32)
        self.est_encoder = Encoder(3, 32)
        self.merge_conv = SeparableConv2d(32, 32)
        self.norm = nn.InstanceNorm2d(32)
        self.model = ConvVAE(in_dims=32, latent_dims=128)

    def forward(self, rgb, est):
        in_size = rgb.shape[2:]
        rgb = self.rgb_encoder(rgb)
        est = self.est_encoder(est)
        x = rgb + est
        x = self.merge_conv(x)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.model(x)
        x = F.interpolate(x, size=in_size, mode="bilinear", align_corners=False)
        return x

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        depth = batch["depth"]
        est = batch["est"]
        est_edges = batch["est_edges"]
        est_laplacian = batch["est_laplacian"]

        stack = torch.cat([est, est_edges, est_laplacian], dim=1)
        est_log_var = self(image, stack)
        est_var = torch.clamp(est_log_var, -6, 6).exp()

        nll_loss = torch.mean(
            (
                F.huber_loss(est, depth, reduction="none") / (2 * est_var)
                + 0.5 * est_log_var
            )
        )
        diff = (est - depth) ** 2
        # diff_loss = F.smooth_l1_loss(est_var, diff)
        ratio_loss = torch.mean(torch.abs(diff / est_var - 1.0))
        diff_loss = ratio_loss

        # regularization, penalize very large values
        reg = torch.mean(torch.abs(est_log_var))

        # Apply weights to the losses
        nll_loss = nll_loss * 1.0
        diff_loss = diff_loss * 0.3
        reg = reg * 0.01

        loss = nll_loss + diff_loss + reg

        rr.log("train/loss", rr.Scalar(loss.detach().cpu()))
        rr.log("train/loss/nll", rr.Scalar(nll_loss.detach().cpu()))
        rr.log("train/loss/diff", rr.Scalar(diff_loss.detach().cpu()))

        self.last_diff = diff
        self.last_image = image
        self.last_est = est
        self.last_est_var = est_var
        self.last_depth = depth
        return loss

    def on_train_epoch_end(self):
        image = (
            self.last_image[0]
            .squeeze()
            .detach()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)
            .clip(0, 255)
            .astype("uint8")
        )
        depth_img = self.last_depth[0, 0].cpu().numpy()
        rr.log("image/", rr.Image(image))
        rr.log("image/depth", rr.DepthImage(depth_img))
        last_est = self.last_est[0].cpu().numpy()
        rr.log("image/est", rr.DepthImage(last_est))
        last_diff = self.last_diff[0].cpu().numpy()
        rr.log("image/diff", rr.Tensor(last_diff))
        est_var_img: torch.Tensor = (
            self.last_est_var[0].squeeze().detach().cpu().numpy()
        )
        rr.log("image/est/var", rr.Tensor(est_var_img))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
        return opt


if __name__ == "__main__":
    model = UncertaintyEstimator(6)
    x = torch.randn(2, 5, 256, 256)
    y = model(x)
    print(y.shape)
