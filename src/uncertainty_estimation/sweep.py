from os import getenv

import lightning as L
import neptune  # noqa: F401
import optuna
from dotenv import load_dotenv
from lightning.pytorch import callbacks as CB
from lightning.pytorch import strategies as ST
from lightning.pytorch.loggers import NeptuneLogger
from optuna_integration import PyTorchLightningPruningCallback

from uncertainty_estimation.data import UncertaintyDatamodule
from uncertainty_estimation.model import UncertaintyEstimator

SEED = 42

load_dotenv()
neptune_key = getenv("NEPTUNE_API_TOKEN")
project = getenv("NEPTUNE_PROJECT")


def objective(trial: optuna.Trial):
    activation: str = trial.suggest_categorical(
        "activation function", ["gelu", "silu", "mish", "relu"]
    )
    optimizer: str = trial.suggest_categorical(
        "optimizer", ["prodigy", "ranger", "adamw"]
    )
    batch_size: int = trial.suggest_categorical("batch size", [16, 32, 64])
    estimated_loss_w: float = trial.suggest_float("estimated loss weight", 0.5, 2.0)
    reference_loss_w: float = trial.suggest_float(
        "reference loss weight", 1e-6, 0.1, log=True
    )
    lightning_module = UncertaintyEstimator(
        model=None,
        activation_name=activation,
        optimizer_name=optimizer,
        estimated_loss_w=estimated_loss_w,
        reference_loss_w=reference_loss_w,
    )
    data_module = UncertaintyDatamodule(seed=SEED, batch_size=batch_size)
    logger = NeptuneLogger(api_key=neptune_key, project=project)
    pruner = PyTorchLightningPruningCallback(trial, monitor="val/corr")
    trainer = L.Trainer(
        max_epochs=30,
        logger=logger,
        precision="16-mixed",
        devices=-1,
        strategy=ST.DDPStrategy(find_unused_parameters=True),
        gradient_clip_val=1.0,
        callbacks=[
            CB.LearningRateMonitor(logging_interval="epoch"),
            CB.ModelCheckpoint(
                dirpath="checkpoints",
                filename="{epoch}_corr={val/corr:.2f}_loss={val/loss:.2f}",
                monitor="val/corr",
                mode="min",
                auto_insert_metric_name=False,
            ),
            CB.EarlyStopping(monitor="val/loss", patience=5, verbose=True),
            pruner,
        ],
    )
    trainer.fit(lightning_module, data_module)
    results = trainer.test(lightning_module, data_module)
    return results[0]["val/corr"]


def run_sweep(n_trials=10):
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    storage = optuna.storages.RDBStorage(
        url="sqlite:///:memory:",
    )
    study = optuna.create_study(direction="minimize", storage=storage, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"\tValue: {trial.value}")
    print("\tParameters:")
    for k, v in trial.params.items():
        print(f"\t\t{k}: {v}")

    return study


if __name__ == "__main__":
    run_sweep()
