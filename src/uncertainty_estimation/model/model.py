import rerun as rr
import torch
from lightning import LightningModule
from torch.nn import functional as F
from torchvision.transforms.v2 import GaussianBlur

from .layers import Encoder

# from .convmixer import ConvMixer
# from .unet import UNet
from .conv_vae import ConvVAE


class UncertaintyEstimator(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.rgb_encoder = Encoder(3, 16)
        self.est_encoder = Encoder(3, 16)
        # self.model = ConvMixer()
        # self.model = UNet(in_dims=32, out_dims=1)
        self.model = ConvVAE(in_dims=32, latent_dims=512, out_dims=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((32, 32))
        self.blur = GaussianBlur(kernel_size=(5, 5), sigma=1.2)

    def blur_scale(self, x, shape):
        x = self.blur(x)
        x = F.interpolate(x, size=shape, mode="bilinear", align_corners=False)
        return x

    def forward(self, rgb, est):
        in_shape = est.shape[2:]
        rgb = self.rgb_encoder(rgb)
        est = self.est_encoder(est)
        x = torch.cat([rgb, est], dim=1)
        x = self.model(x)
        x = self.pool(x)
        x = self.blur_scale(x, in_shape).clamp(-3, 3).exp()
        return x

    def training_step(self, batch, _batch_idx):
        image = batch["image"]
        depth = batch["depth"]
        depth_edges = batch["depth_edges"]
        depth_laplacian = batch["depth_laplacian"]
        observation = batch["est"]
        observation_edges = batch["est_edges"]
        observation_laplacian = batch["est_laplacian"]

        observation_stack = torch.cat(
            [observation, observation_edges, observation_laplacian], dim=1
        )
        reference_stack = torch.cat([depth, depth_edges, depth_laplacian], dim=1)
        estimated_variance = self(image, observation_stack)
        reference_variance = self(image, reference_stack)

        eps = 1e-6
        difference = (depth - observation) ** 2
        estimated_nll_loss = torch.mean(
            0.5 * torch.log(2 * torch.pi * (estimated_variance + eps))
            + difference / (2 * (estimated_variance + eps))
        )
        reference_nll_loss = torch.mean(
            0.5 * torch.log(2 * torch.pi * (reference_variance + eps))
        )

        # regularization, penalize very large values
        reg = torch.mean(torch.abs(estimated_variance)) * 0.01

        alpha = 1.0
        beta = 0.1
        loss = alpha * estimated_nll_loss + beta * reference_nll_loss + reg

        rr.log("train/loss", rr.Scalar(loss.detach().cpu()))
        # rr.log("train/loss/nll", rr.Scalar(nll_loss.detach().cpu()))
        # rr.log("train/loss/diff", rr.Scalar(diff_loss.detach().cpu()))

        self.last_image = image
        self.last_obs = observation
        self.last_est_var = estimated_variance
        self.last_ref_var = reference_variance
        self.last_depth = depth
        return loss

    def on_train_epoch_end(self):
        if self.trainer.current_epoch % 5 != 0:
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
        last_obs = self.last_obs[0].cpu().numpy()
        rr.log("image/est", rr.DepthImage(last_obs))
        est_var_img: torch.Tensor = (
            self.last_est_var[0].squeeze().detach().cpu().numpy()
        )
        rr.log("image/est/var", rr.Tensor(est_var_img))
        ref_var_img: torch.Tensor = (
            self.last_ref_var[0].squeeze().detach().cpu().numpy()
        )
        rr.log("image/ref/var", rr.Tensor(ref_var_img))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt,
            T_0=self.trainer.estimated_stepping_batches // 5,
            T_mult=1,
            eta_min=1e-6,
        )
        return {"optimizer": opt, "scheduler": sched}


if __name__ == "__main__":
    model = UncertaintyEstimator(8)
    x = torch.randn(2, 8, 256, 256)
    y = model(x)
    print(y.shape)
