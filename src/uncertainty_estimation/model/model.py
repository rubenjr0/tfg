import rerun as rr
import torch
from torch import nn
from torch.nn import functional as F
from lightning import LightningModule

from .unet import UNet
from uncertainty_estimation.utils.pipeline import Pipeline


class UncertaintyEstimator(LightningModule):
    def __init__(self, in_dims: int, out_dims: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.norm = nn.BatchNorm2d(in_dims)
        self.model = UNet(
            in_dims=in_dims,
            out_dims=out_dims,
        )

    def forward(self, x):
        x = self.norm(x)
        log_var = self.model(x)
        var = log_var.clamp(-6, 6).exp()
        return var

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        y = batch["depth"]
        y_var = batch["depth_var"]
        est = batch["est"]
        mask = batch["mask"]

        stack = torch.cat([est * 1.0, image * 0.25], dim=1)
        est_var = self(stack)
        new_mu, new_var = Pipeline.refine(mu_1=y, var_1=y_var, mu_2=est, var_2=est_var)

        #
        nll_loss = torch.mean(
            (
                F.huber_loss(est, y, reduction="none") / (2 * est_var)
                + 0.5 * est_var.log()
            )
            * mask
        )

        # The estimated variance should have a similar range as the computed variance
        range_loss = F.huber_loss(est_var, y_var, reduction="none") * mask
        range_loss = torch.mean(range_loss) * 0.1

        # The refined mean should be close to the estimated depth
        rec = F.mse_loss(new_mu, est)

        # regularization, penalize very large values
        reg = torch.mean(est_var) * 0.01

        # We penalize the estimated variance if it is too far from the computed variance
        loss = nll_loss + rec + range_loss + reg

        rr.log("train/loss", rr.Scalar(loss.detach().cpu()))
        rr.log("train/loss/nll", rr.Scalar(nll_loss.detach().cpu()))
        rr.log("train/loss/range", rr.Scalar(range_loss.detach().cpu()))
        rr.log("train/loss/rec", rr.Scalar(rec.detach().cpu()))

        self.last_image = image
        self.last_est = est
        self.last_est_var = est_var
        self.last_y = y
        self.last_y_var = y_var
        self.last_mask = mask
        self.last_new_mu = new_mu
        self.last_new_var = new_var
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
        y_img = self.last_y[0, 0].cpu().numpy()
        y_var_img = self.last_y_var[0, 0].cpu().numpy()
        mask_img = self.last_mask[0, 0].cpu().numpy()
        rr.log("image/", rr.Image(image))
        rr.log("image/depth", rr.DepthImage(y_img))
        rr.log("image/depth/var", rr.Tensor(y_var_img))
        rr.log("image/mask", rr.Tensor(mask_img))
        last_est = self.last_est[0].cpu().numpy()
        rr.log("image/est", rr.DepthImage(last_est))
        est_var_img: torch.Tensor = (
            self.last_est_var[0].squeeze().detach().cpu().numpy()
        )
        rr.log("image/est/var", rr.Tensor(est_var_img))
        new_mu_img = self.last_new_mu[0].squeeze().detach().cpu().numpy()
        new_var_img = self.last_new_var[0].squeeze().detach().cpu().numpy()
        rr.log("new/mu", rr.DepthImage(new_mu_img))
        rr.log("new/var", rr.Tensor(new_var_img))

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=2e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=20,
            eta_min=1e-6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": sched,
        }


if __name__ == "__main__":
    model = UncertaintyEstimator(6)
    x = torch.randn(2, 5, 256, 256)
    y = model(x)
    print(y.shape)
