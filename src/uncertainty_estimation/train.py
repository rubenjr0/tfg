import sys
from os import getenv

import neptune  # noqa: F401
import optuna
import torch
from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import NeptuneLogger

from uncertainty_estimation.data import UncertaintyDatamodule
from uncertainty_estimation.model import UncertaintyEstimator


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--sweep', default=False)


SEED = 42


torch.set_float32_matmul_precision("medium")
torch.manual_seed(SEED)

load_dotenv()
neptune_key = getenv("NEPTUNE_API_TOKEN")
project = getenv("NEPTUNE_PROJECT")
use_rerun = getenv("USE_RERUN", "false").lower() == "true"

logger = NeptuneLogger(
    api_key=neptune_key,
    project=project,
)

if use_rerun:
    import rerun as rr

    rr.init("uncertainty-predictor", spawn=True)

cli = MyCLI(
    model_class=UncertaintyEstimator,
    datamodule_class=UncertaintyDatamodule,
    args=None,
    run=False,
    seed_everything_default=SEED,
    save_config_callback=None,
    save_config_kwargs={"overwrite": False},
    trainer_defaults={
        "max_epochs": 30,
        "logger": {
            "class_path": "lightning.pytorch.loggers.NeptuneLogger",
            "init_args": {
                "api_key": neptune_key,
                "project": project,
            },
        },
        "precision": "16-mixed",
        "strategy": {
            "class_path": "lightning.pytorch.strategies.DDPStrategy",
            "init_args": {"find_unused_parameters": True},
        },
        "gradient_clip_val": 1.0,
    },
)


def train():
    lightning_module = UncertaintyEstimator(
        model=None,
        activation_name=cli.model.activation_name,
        optimizer_name=cli.model.optimizer_name,
        estimated_loss_w=cli.model.estimated_loss_w,
        reference_loss_w=cli.model.reference_loss_w,
    )
    data_module = UncertaintyDatamodule(seed=SEED, batch_size=cli.datamodule.batch_size)

    cli.trainer.fit(lightning_module, data_module)
    cli.trainer.test(lightning_module, data_module)


def objective(trial: optuna.Trial):
    activation: str = trial.suggest_categorical(
        "activation function", ["gelu", "silu", "mish", "relu"]
    )
    optimizer: str = trial.suggest_categorical(
        "optimizer", ["prodigy", "ranger", "adamw"]
    )
    batch_size: int = trial.suggest_categorical("batch size", [16, 32])
    estimated_loss_w: float = trial.suggest_float("estimated loss weight", 0.5, 2.0)
    reference_loss_w: float = trial.suggest_loguniform(
        "reference loss weight", 1e-6, 0.1
    )
    lightning_module = UncertaintyEstimator(
        model=None,
        activation_name=activation,
        optimizer_name=optimizer,
        estimated_loss_w=estimated_loss_w,
        reference_loss_w=reference_loss_w,
    )
    data_module = UncertaintyDatamodule(seed=SEED, batch_size=batch_size)
    cli.trainer.fit(lightning_module, data_module)
    results = cli.trainer.test(lightning_module, data_module)
    return results["test/corr"]


def run_sweep(n_trials=10):
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"\tValue: {trial.value}")
    print("\tParameters:")
    for k, v in trial.params.items():
        print(f"\t\t{k}: {v}")

    return study


if __name__ == "__main__":
    if cli.config["sweep"]:
        run_sweep()
    else:
        train()
