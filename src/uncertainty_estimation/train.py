import lightning as L
import rerun as rr
import torch
from lightning.pytorch import callbacks as CB
from torch.utils.data import DataLoader

from uncertainty_estimation.data import ImageDepthDataset
from uncertainty_estimation.model import UncertaintyEstimator


def train():
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    dataset = ImageDepthDataset(root="data/")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=12)
    model = UncertaintyEstimator()

    rr.init("uncertainty-predictor", spawn=True)

    trainer = L.Trainer(
        logger=False,
        # fast_dev_run=True,
        log_every_n_steps=2,
        max_epochs=100,
        precision="16-mixed",
        # gradient_clip_val=1.0,
        detect_anomaly=True,
        callbacks=[
            CB.StochasticWeightAveraging(swa_lrs=1e-3),
        ],
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train()
