import lightning as L
import rerun as rr
import torch
from torch.utils.data import DataLoader
# from lightning.pytorch import callbacks as CB

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
        max_epochs=100,
        log_every_n_steps=2,
        logger=False,
        # fast_dev_run=True,
        # gradient_clip_val=1.0,
        detect_anomaly=False,
        precision="16-mixed",
        callbacks=[
            # CB.StochasticWeightAveraging(swa_lrs=1e-3),
        ],
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train()
