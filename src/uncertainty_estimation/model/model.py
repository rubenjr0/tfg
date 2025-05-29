import rerun as rr
import torch
from lightning import LightningModule

# from torch.nn import functional as F
from ranger21 import Ranger21

from .layers import Encoder

from .unet import UNet
# from .convmixer import ConvMixer
# from .conv_vae import ConvVAE


class UncertaintyEstimator(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.rgb_encoder = Encoder(in_dims=3, out_dims=16)
        self.stack_encoder = Encoder(in_dims=3, out_dims=16)
        # self.model = ConvVAE(in_dims=32)
        # self.model = ConvMixer(in_dims=32, h_dims=128, out_dims=1, depth=20)
        self.model = UNet(in_dims=32)

        self.estimated_w = 1.0
        self.reference_w = 0.1
        self.tv_w = 0.01
        self.reg_w = 0.01

    def forward(self, rgb, depth, depth_edges, depth_laplacian):
        # in_shape = rgb.shape[2:]
        stack = torch.cat([depth, depth_edges, depth_laplacian], dim=1)
        rgb = self.rgb_encoder(rgb)
        stack = self.stack_encoder(stack)
        x = torch.cat([rgb, stack], dim=1)
        x = self.model(x)
        x = x.clamp(-6, 6).exp()
        return x

    def nll_loss(
        self,
        estimated_variance: torch.Tensor,
        target: torch.Tensor,
        eps=1e-6,
    ):
        safe_var = estimated_variance + eps
        log = 0.5 * torch.log(2 * torch.pi * safe_var)
        quad = target / (2 * safe_var)
        nll_loss = (log + quad).mean()
        return nll_loss

    def total_variance_loss(self, variance_map):
        return torch.mean(
            (variance_map[:, :, 1:, :] - variance_map[:, :, :-1, :]) ** 2
        ) + torch.mean((variance_map[:, :, :, 1:] - variance_map[:, :, :, :-1]) ** 2)

    def step(self, split: str, batch):
        image = batch["image"]
        depth = batch["depth"]
        depth_edges = batch["depth_edges"]
        depth_laplacian = batch["depth_laplacian"]
        observation = batch["est"]
        observation_edges = batch["est_edges"]
        observation_laplacian = batch["est_laplacian"]

        estimated_variance = self(
            image, observation, observation_edges, observation_laplacian
        )
        reference_variance = self(image, depth, depth_edges, depth_laplacian)

        difference = (depth - observation) ** 2

        # Estimated depth variance loss
        estimated_nll_loss = self.nll_loss(estimated_variance, difference)

        # Ground truth depth variance loss (should be close to zero)
        reference_nll_loss = self.nll_loss(
            reference_variance, torch.full_like(difference, 0.01)
        )

        # regularization, penalize large values
        reg = torch.mean(torch.abs(estimated_variance)) + torch.mean(
            torch.abs(reference_variance)
        )

        # Smoothness loss
        tv_loss = self.total_variance_loss(estimated_variance)

        loss = (
            self.estimated_w * estimated_nll_loss
            + self.reference_w * reference_nll_loss
            + self.tv_w * tv_loss
            + self.reg_w * reg
        )

        rr.log(f"{split}/loss", rr.Scalars(loss.detach().cpu()))
        rr.log(f"{split}/loss/estimated", rr.Scalars(estimated_nll_loss.detach().cpu()))
        rr.log(f"{split}/loss/reference", rr.Scalars(reference_nll_loss.detach().cpu()))
        rr.log(f"{split}/loss/total", rr.Scalars(tv_loss.detach().cpu()))
        return loss, image, observation, estimated_variance, reference_variance, depth

    def training_step(self, batch, _batch_idx):
        loss, _, _, _, _, _ = self.step("train", batch)
        return loss

    def validation_step(self, batch, _batch_idx):
        loss, image, observation, estimated_variance, reference_variance, depth = (
            self.step("val", batch)
        )
        acc = torch.mean((depth - observation) ** 2 / (estimated_variance + 1e-6))
        rr.log("val/acc", rr.Scalars(acc.detach().cpu()))

        self.last_image = image
        self.last_obs = observation
        self.last_est_var = estimated_variance
        self.last_ref_var = reference_variance
        self.last_depth = depth
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.current_epoch % self.trainer.log_every_n_steps != 0:
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
        batches_per_epoch = (
            self.trainer.estimated_stepping_batches / self.trainer.max_epochs
        )
        opt = Ranger21(
            self.parameters(),
            lr=2e-4,
            num_epochs=self.trainer.max_epochs,
            num_batches_per_epoch=batches_per_epoch,
            weight_decay=1e-3,
        )
        return opt


if __name__ == "__main__":
    model = UncertaintyEstimator()
    x = torch.randn(2, 8, 256, 256)
    y = model(x)
    print(y.shape)
