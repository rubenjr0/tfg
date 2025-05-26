import lightning as L
from lightning.pytorch import callbacks as CB
import rerun as rr
import torch
from torch.utils.data import DataLoader

from uncertainty_estimation.data import ImageDepthDataset
from uncertainty_estimation.model import UncertaintyEstimator


def train():
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    dataset = ImageDepthDataset(root="data/ai_001_001/images")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=12)
    model = UncertaintyEstimator()

    rr.init("uncertainty-predictor", spawn=True)

    trainer = L.Trainer(
        logger=False,
        # fast_dev_run=True,
        log_every_n_steps=25,
        max_epochs=100,
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=[
            CB.StochasticWeightAveraging(swa_lrs=1e-4),
        ],
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    train()
