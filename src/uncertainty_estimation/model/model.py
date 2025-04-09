import rerun as rr
import torch
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from .layers import Encoder, SeparableConv2d
from .convmixer import ConvMixer


class UncertaintyEstimator(LightningModule):
    def __init__(self, in_dims: int, out_dims: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.rgb_encoder = Encoder(3, 32)
        self.est_encoder = Encoder(3, 32)
        self.merge_conv = SeparableConv2d(32, 32)
        self.norm = nn.InstanceNorm2d(32)
        self.model = ConvMixer(
            in_dims=32,
            h_dims=64,
            out_dims=out_dims,
            depth=32,
            kernel_size=7,
            patch_size=7,
        )

    def forward(self, rgb, est):
        in_size = rgb.shape[2:]
        rgb = self.rgb_encoder(rgb)
        est = self.est_encoder(est)
        x = rgb + est
        x = self.merge_conv(x)
        x = self.norm(x)
        x = F.gelu(x)
        log_var = self.model(x)
        log_var = F.interpolate(
            log_var, size=in_size, mode="bilinear", align_corners=False
        )
        var = log_var.clamp(-6, 6).exp()
        return var

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        depth = batch["depth"]
        est = batch["est"]
        est_edges = batch["est_edges"]
        est_laplacian = batch["est_laplacian"]

        stack = torch.cat([est, est_edges, est_laplacian], dim=1)
        est_var = self(image, stack)

        nll_loss = torch.mean(
            (
                F.huber_loss(est, depth, reduction="none") / (2 * est_var)
                + 0.5 * est_var.log()
            )
        )
        diff = torch.abs(est - depth)
        diff_loss = F.huber_loss(est_var, diff) * 0.5

        # regularization, penalize very large values
        reg = torch.mean(est_var) * 0.2

        # We penalize the estimated variance if it is too far from the computed variance
        loss = nll_loss + diff_loss + reg

        rr.log("train/loss", rr.Scalar(loss.detach().cpu()))
        rr.log("train/loss/nll", rr.Scalar(nll_loss.detach().cpu()))
        rr.log("train/loss/diff", rr.Scalar(diff_loss.detach().cpu()))

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
        est_var_img: torch.Tensor = (
            self.last_est_var[0].squeeze().detach().cpu().numpy()
        )
        rr.log("image/est/var", rr.Tensor(est_var_img))

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=1e-2, weight_decay=1e-5)
        return opt


if __name__ == "__main__":
    model = UncertaintyEstimator(6)
    x = torch.randn(2, 5, 256, 256)
    y = model(x)
    print(y.shape)
