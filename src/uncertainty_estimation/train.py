import lightning as L
import rerun as rr
import torch
from torch.utils.data import DataLoader

from uncertainty_estimation.data import ImageDepthDataset
from uncertainty_estimation.model.model import UncertaintyEstimator
from uncertainty_estimation.utils import Sensor


def train():
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    sensor = Sensor(0.3, 20, (0.0008, 0.0016, 0.0018))
    dataset = ImageDepthDataset(root="data/val/indoors", sensor=sensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=12)
    model = UncertaintyEstimator(in_dims=4)

    rr.init("uncertainty-predictor", spawn=True)

    trainer = L.Trainer(
        logger=False,
        # fast_dev_run=True,
        log_every_n_steps=5,
        max_epochs=2500,
        precision="16-mixed",
        gradient_clip_val=0.5,
    )

    trainer.fit(model, dataloader)
