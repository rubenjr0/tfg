from os import getenv
import lightning as L
import rerun as rr
import torch
from torch.utils.data import DataLoader, random_split
from dotenv import load_dotenv

# from lightning.pytorch import callbacks as CB

from uncertainty_estimation.data import ImageDepthDataset
from uncertainty_estimation.model import UncertaintyEstimator


def train():
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    dataset = ImageDepthDataset(root="data/")
    train_ds, val_ds = random_split(dataset, [0.7, 0.3])
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=12)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=12)
    model = UncertaintyEstimator()

    load_dotenv()
    logger = None
    neptune_key = getenv("NEPTUNE_API_TOKEN")
    if neptune_key is not None:
        import neptune  # noqa: F401
        from lightning.pytorch.loggers import NeptuneLogger

        project = getenv("NEPTUNE_PROJECT")
        logger = NeptuneLogger(
            api_key=neptune_key,
            project=project,
            tags=["uncertainty", "depth", "rgb"],
        )

    rr.init("uncertainty-predictor", spawn=True)

    trainer = L.Trainer(
        max_epochs=100,
        log_every_n_steps=2,
        logger=logger,
        # fast_dev_run=True,
        # gradient_clip_val=1.0,
        detect_anomaly=False,
        precision="16-mixed",
        callbacks=[
            # CB.StochasticWeightAveraging(swa_lrs=1e-3),
        ],
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    train()
