import math
import os
import sys

import pytorch_lightning as pl
import ray
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import PopulationBasedTraining

from nnclr import NNCLR


def train_nnclr_tune_checkpoint(config,
                                checkpoint_dir=None,
                                num_epochs=10,
                                num_gpus=0,
                                data_dir="~/data"):
    data_dir = os.path.expanduser(data_dir)
    kwargs = {
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "gpus": math.ceil(num_gpus),
        "logger": TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        "enable_progress_bar": False,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics={
                    "loss_step": "loss_step",
                    "loss_epoch": "loss_epoch"
                },
                filename="checkpoint",
                on="train_end")
        ]
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")

    model = NNCLR(**config)
    trainer = pl.Trainer(**kwargs)

    trainer.fit(model)


def tune_nnclr_pbt(num_samples=50, num_epochs=10, gpus_per_trial=1, data_dir="~/data"):
    config = {
        "arch": "conv4",
        "conv_4_out_planes": tune.choice([64, [96, 128, 256, 512]]),
        "projection_out_dim": tune.choice([64, 128, 512]),
        "lr": 1e-3,
        "bsize": 64,
        "num_workers": 2,
        "optimiser": tune.choice(["sgd", "adam"]),
        "scheduler": tune.choice(["cos", None])
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=4,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-1),
            "bsize": [64, 128, 256]
        })

    reporter = CLIReporter(
        parameter_columns=["projection_out_dim", "lr", "bsize"],
        metric_columns=["loss_step", "loss_epoch", "training_iteration"]
    )

    analysis = tune.run(
        tune.with_parameters(
            train_nnclr_tune_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            data_dir=data_dir),
        resources_per_trial={
            "cpu": 4,
            "gpu": gpus_per_trial
        },
        metric="loss_epoch",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_nnclr_pbt")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    redis_password = sys.argv[1]
    num_cpus = int(sys.argv[2])
    ray.init(address=os.environ["ip_head"], redis_password=redis_password)
    tune_nnclr_pbt(10, num_epochs=50, gpus_per_trial=1,
                   data_dir="/home/nfs/oshirekar/unsupervised_ml/data/miniimagenet_84/")
