import rerun as rr
import torch
from lightning import LightningModule
from torch.nn import functional as F
from torchvision.transforms.v2 import GaussianBlur

from .layers import Encoder
from .convmixer import ConvMixer


class UncertaintyEstimator(LightningModule):
    def __init__(self, in_dims: int, out_dims: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.rgb_encoder = Encoder(3, 16)
        self.est_encoder = Encoder(3, 16)
        self.model = ConvMixer(in_dims=16, h_dims=32, out_dims=out_dims, depth=8)
        # Very heavy Gaussian blur
        self.blur = GaussianBlur(kernel_size=9)

    def forward(self, rgb, est):
        in_shape = est.shape[2:]
        rgb = self.rgb_encoder(rgb)
        est = self.est_encoder(est)
        x = rgb + est
        x = self.model(x)
        x = F.interpolate(x, size=in_shape, mode="bilinear", align_corners=False)
        # x = self.blur(x)
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
        diff = F.avg_pool2d((est - depth) ** 2, 4)
        diff = self.blur(diff)
        diff = F.interpolate(
            diff, size=est.shape[2:], mode="bilinear", align_corners=False
        )
        diff_loss = F.smooth_l1_loss(est_var, diff)

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
        if (self.trainer.global_step - 1) % 10 != 0:
            return
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
        opt = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-7,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": sched,
        }


if __name__ == "__main__":
    model = UncertaintyEstimator(8)
    x = torch.randn(2, 8, 256, 256)
    y = model(x)
    print(y.shape)
