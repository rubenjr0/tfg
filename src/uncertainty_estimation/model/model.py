from os import getenv

import numpy as np
import rerun as rr
import torch
from lightning import LightningModule
from neptune.types import File
from prodigyopt import Prodigy
from ranger21 import Ranger21

from .unet import UNet


class UncertaintyEstimator(LightningModule):
    def __init__(
        self,
        model: None | UNet = None,
        activation_name: str = "relu",
        optimizer_name: str = "adamw",
        estimated_loss_w: float = 1.0,
        reference_loss_w: float = 1e-3,
        curriculum_epochs: int = 15,
        learning_rate: float = 1e-4,
        batch_size: int | None = None,
    ):
        super().__init__()
        if model is None:
            model = UNet(activation_name)
        self.model = model
        self.activation_name = activation_name
        self.optimizer_name = optimizer_name
        self.estimated_loss_w = estimated_loss_w
        self.reference_loss_w = reference_loss_w
        self.curriculum_epochs = curriculum_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters(ignore="model")
        self.rerun_logging = getenv("USE_RERUN", "false").lower() == "true"

    def forward(self, rgb, depth, depth_edges, depth_laplacian):
        return self.model(rgb, depth, depth_edges, depth_laplacian)

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
        reference_loss_w = (
            min(1.0, self.trainer.current_epoch / self.curriculum_epochs)
            * self.reference_loss_w
            / image.size(0)
        )

        loss = (
            self.estimated_loss_w * estimated_nll_loss
            + reference_loss_w * reference_nll_loss
        )

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
        if self.optimizer_name == "ranger":
            batches_per_epoch = (
                self.trainer.estimated_stepping_batches / self.trainer.max_epochs
            )
            opt = Ranger21(
                self.parameters(),
                lr=self.learning_rate,
                use_madgrad=True,
                num_epochs=self.trainer.max_epochs,
                num_batches_per_epoch=batches_per_epoch,
                weight_decay=1e-3,
            )
            return opt
        else:
            if self.optimizer_name == "adam":
                opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
            elif self.optimizer_name == "adamw":
                opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
            elif self.optimizer_name == "radam":
                opt = torch.optim.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
            elif self.optimizer_name == "prodigy":
                opt = Prodigy(self.parameters(), lr=1.0, d_coef=0.1, weight_decay=1e-3)
            else:
                return ValueError("No valid optimizer was specified")
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.estimated_stepping_batches
            )
            return [opt], [sched]
         