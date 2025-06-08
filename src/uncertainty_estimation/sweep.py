from os import getenv

import joblib
import lightning as L
import neptune  # noqa: F401
import optuna
from dotenv import load_dotenv
from lightning.pytorch import callbacks as CB
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
        "activation function", ["relu", "gelu", "mish", "siren", "swish"]
    )
    optimizer: str = trial.suggest_categorical(
        "optimizer", ["adamw", "radam", "prodigy"]
    )
    # batch_size: int = trial.suggest_categorical("batch size", [16, 32])
    batch_size: int = 16
    if optimizer == "prodigy":
        learning_rate = 1.0
        trial.set_user_attr("learning rate", learning_rate)
    else:
        learning_rate: float = trial.suggest_float(
            "learning rate", 1e-5, 1e-3, log=True
        )
    estimated_loss_w: float = trial.suggest_float("estimated loss weight", 0.5, 2.0)
    reference_loss_w: float = trial.suggest_float(
        "reference loss weight",
        1e-8,
        0.1,
    )
    print("Running trial with:")
    print("\t- Activation:", activation)
    print("\t- Optimizer:", optimizer)
    print("\t- Batch Size:", batch_size)
    print("\t- Learning rate:", learning_rate)
    print("\t- Estimated loss weight:", estimated_loss_w)
    print("\t- Reference loss weight:", reference_loss_w)
    lightning_module = UncertaintyEstimator(
        model=None,
        activation_name=activation,
        optimizer_name=optimizer,
        estimated_loss_w=estimated_loss_w,
        reference_loss_w=reference_loss_w,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    data_module = UncertaintyDatamodule(seed=SEED, batch_size=batch_size)
    logger = NeptuneLogger(
        api_key=neptune_key, project=project, name=f"tfg-sweep-{trial.number}"
    )
    trainer = L.Trainer(
        max_epochs=10,
        logger=logger,
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        callbacks=[
            CB.LearningRateMonitor(logging_interval="epoch"),
            CB.ModelCheckpoint(
                dirpath=f"checkpoints/trial_{trial.number}",
                filename="best_{epoch}_corr={val/corr:.2f}_loss={val/loss:.2f}",
                monitor="val/corr",
                mode="min",
                auto_insert_metric_name=False,
            ),
            PyTorchLightningPruningCallback(trial, monitor="val/corr"),
        ],
    )
    trainer.fit(lightning_module, data_module)
    logger.finalize("success")
    return trainer.callback_metrics["val/corr"]


def run_sweep(n_trials=100):
    sampler = optuna.samplers.GPSampler(seed=SEED)
    pruner = optuna.pruners.SuccessiveHalvingPruner()
    storage = optuna.storages.RDBStorage(
        url="sqlite:///optuna_sweep_db.sqlite",
    )
    study = optuna.create_study(
        study_name="uncertainty_sweep_1",
        direction="minimize",
        storage=storage,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print("Best trial:")
    trial: optuna.Trial = study.best_trial
    print(f"\tValue: {trial.value}")
    print("\tParameters:")
    for k, v in trial.params.items():
        print(f"\t\t{k}: {v}")

    return study


if __name__ == "__main__":
    print("Running optuna sweep...")
    study = run_sweep()
    joblib.dump(study, "study.pkl")
