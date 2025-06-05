from os import getenv

import numpy as np
import rerun as rr
import torch
from lightning import LightningModule
from neptune.types import File
from prodigyopt import Prodigy
from ranger21 import Ranger21

from .layers import Encoder
from .unet import UNet


class UncertaintyEstimator(LightningModule):
    def __init__(self, activation: str, optimizer: str, curriculum_epochs: int = 20):
        super().__init__()
        self.save_hyperparameters("activation", "optimizer", "curriculum_epochs")
        self.rgb_encoder = Encoder(in_dims=3, out_dims=16, act=activation)
        self.stack_encoder = Encoder(in_dims=3, out_dims=16, act=activation)
        self.model = UNet(in_dims=32, out_dims=1, act=activation)
        self.activation = activation
        self.optimizer_name = optimizer

        self.estimated_w = 1.0
        self.reference_w = 1e-3
        self.curriculum_epochs = curriculum_epochs
        self.rerun_logging = getenv("USE_RERUN", "false").lower() == "true"

    def forward(self, rgb, depth, depth_edges, depth_laplacian):
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
        ground_truth: torch.Tensor,
        observation: torch.Tensor,
        eps: float = 1e-6,
    ):
        safe_var = torch.maximum(torch.tensor(eps), estimated_variance)
        return torch.mean(
            torch.log(safe_var) + torch.abs(ground_truth - observation) / safe_var
        )

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

        # Estimated depth variance loss
        estimated_nll_loss = self.nll_loss(estimated_variance, depth, observation)

        # Ground truth depth variance loss (should be close to zero)
        reference_nll_loss = self.nll_loss(reference_variance, depth, depth)
        reference_w = (
            min(1.0, self.trainer.current_epoch / self.curriculum_epochs)
            * self.reference_w
        )

        loss = self.estimated_w * estimated_nll_loss + reference_w * reference_nll_loss

        if self.rerun_logging:
            rr.log(f"{split}/loss", rr.Scalars(loss.detach().cpu()))
            rr.log(
                f"{split}/loss/estimated_var",
                rr.Scalars(estimated_nll_loss.detach().cpu()),
            )
            rr.log(
                f"{split}/loss/reference_var",
                rr.Scalars(reference_nll_loss.detach().cpu()),
            )
        self.log(f"{split}/loss", loss, sync_dist=True)
        self.log(f"{split}/estimated_var_loss", estimated_nll_loss, sync_dist=True)
        self.log(f"{split}/reference_var_loss", reference_nll_loss, sync_dist=True)
        return loss, image, observation, estimated_variance, reference_variance, depth

    def training_step(self, batch, _batch_idx):
        loss, _, _, _, _, _ = self.step("train", batch)
        return loss

    def validation_step(self, batch, _batch_idx):
        loss, image, observation, estimated_variance, reference_variance, depth = (
            self.step("val", batch)
        )
        corr = torch.mean((depth - observation) ** 2 / (estimated_variance + 1e-6))
        if self.rerun_logging:
            rr.log("val/corr", rr.Scalars(corr.detach().cpu()))
        self.log("val/corr", corr, sync_dist=True)

        self.last_image = image
        self.last_obs = observation
        self.last_est_var = estimated_variance
        self.last_ref_var = reference_variance
        self.last_depth = depth
        return loss

    def test_step(self, batch, _batch_idx):
        loss, image, observation, estimated_variance, reference_variance, depth = (
            self.step("test", batch)
        )
        corr = torch.mean((depth - observation) ** 2 / (estimated_variance + 1e-6))
        if self.rerun_logging:
            rr.log("test/corr", rr.Scalars(corr.detach().cpu()))
        self.log("test/corr", corr, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
            return tensor.squeeze().detach().cpu().numpy()

        def to_img(tensor: np.ndarray) -> np.ndarray:
            if tensor.ndim == 2:
                tensor = tensor[:, :, np.newaxis]
            elif tensor.ndim == 3:
                tensor = tensor.transpose(1, 2, 0)
            return (tensor / (tensor.max() + 1e-6) * 255).clip(0, 255).astype(np.uint8)

        if self.trainer.current_epoch % self.trainer.log_every_n_steps != 0:
            return
        image = to_img(tensor_to_numpy(self.last_image[0]))
        depth = tensor_to_numpy(self.last_depth[0])
        obs = tensor_to_numpy(self.last_obs[0])
        est_var = tensor_to_numpy(self.last_est_var[0])
        ref_var = tensor_to_numpy(self.last_ref_var[0])

        if self.rerun_logging:
            rr.log("image/", rr.Image(image))
            rr.log("image/depth", rr.DepthImage(depth))
            rr.log("image/est", rr.DepthImage(obs))
            rr.log("image/est/var", rr.Tensor(est_var))
            rr.log("image/ref/var", rr.Tensor(ref_var))
        logger = self.logger
        if logger is not None:
            logger.experiment["val/image"].append(File.as_image(image))
            logger.experiment["val/depth"].append(File.as_image(to_img(depth)))
            logger.experiment["val/est"].append(File.as_image(to_img(obs)))
            logger.experiment["val/est_var"].append(File.as_image(to_img(est_var)))
            logger.experiment["val/ref_var"].append(File.as_image(to_img(ref_var)))

    def configure_optimizers(self):
        if self.optimizer_name == "prodigy":
            opt = Prodigy(self.parameters(), lr=1.0, weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.estimated_stepping_batches
            )
            return [opt], [sched]
        elif self.optimizer_name == "ranger":
            batches_per_epoch = (
                self.trainer.estimated_stepping_batches / self.trainer.max_epochs
            )
            opt = Ranger21(
                self.parameters(),
                lr=5e-5,
                use_madgrad=True,
                num_epochs=self.trainer.max_epochs,
                num_batches_per_epoch=batches_per_epoch,
                weight_decay=5e-4,
            )
            return opt
        elif self.optimizer_name == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=5e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="min",
                factor=0.7,
                patience=3,
                min_lr=1e-6,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "scheduler": sched,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val/loss",
                    "strict": True,
                },
            }
