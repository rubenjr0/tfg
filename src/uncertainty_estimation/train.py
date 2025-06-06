from os import getenv

import neptune  # noqa: F401
import torch
from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI

from uncertainty_estimation.data import UncertaintyDatamodule
from uncertainty_estimation.model import UncertaintyEstimator


SEED = 42


torch.set_float32_matmul_precision("medium")
torch.manual_seed(SEED)

load_dotenv()

use_rerun = getenv("USE_RERUN", "false").lower() == "true"


if use_rerun:
    import rerun as rr

    rr.init("uncertainty-predictor", spawn=True)


def init_cli():
    neptune_key = getenv("NEPTUNE_API_TOKEN")
    project = getenv("NEPTUNE_PROJECT")
    return LightningCLI(
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
    cli = init_cli()
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


if __name__ == "__main__":
    train()
