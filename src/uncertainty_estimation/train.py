from os import getenv

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.pytorch import callbacks as CB
from torch.utils.data import DataLoader, random_split

from uncertainty_estimation.data import ImageDepthDataset
from uncertainty_estimation.model import UncertaintyEstimator


def train():
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(42)

    load_dotenv()
    neptune_key = getenv("NEPTUNE_API_TOKEN")
    project = getenv("NEPTUNE_PROJECT")
    use_rerun = getenv("USE_RERUN", "false").lower() == "true"
    opt = getenv("OPTIMIZER", "adamw").lower()
    max_epochs = int(getenv("MAX_EPOCHS", "100"))
    batch_size = int(getenv("BATCH_SIZE", "16"))

    dataset = ImageDepthDataset(root="data/")
    train_ds, val_ds = random_split(dataset, [0.8, 0.2])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=12)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=12)

    logger = None
    if neptune_key is not None:
        import neptune  # noqa: F401
        from lightning.pytorch.loggers import NeptuneLogger

        logger = NeptuneLogger(
            api_key=neptune_key,
            project=project,
            tags=["uncertainty", "depth", "rgb"],
        )
    if use_rerun:
        import rerun as rr

        rr.init("uncertainty-predictor", spawn=True)

    model = UncertaintyEstimator(opt=opt, rerun_logging=use_rerun)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=2,
        logger=logger,
        # fast_dev_run=True,
        gradient_clip_val=1.0 if opt == "adamw" else None,
        detect_anomaly=False,
        precision="16-mixed",
        callbacks=[
            CB.StochasticWeightAveraging(swa_lrs=1e-3) if opt == "adamw" else None,
            CB.LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    train()
