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
from wasabi import msg

from clr_gat import CLRGAT
from dataloaders import UnlabelledDataModule


def train_gat_clr_tune_checkpoint(config,
                                  checkpoint_dir=None,
                                  num_epochs=50,
                                  num_gpus=0,
                                  data_dir="~/data"):
    kwargs = {
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "gpus": math.ceil(num_gpus),
        "logger": TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        "enable_progress_bar": False,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics={
                    "val/loss": "val/loss",
                    "val/accuracy": "val/accuracy"
                },
                filename="checkpoint",
                on="train_end")
        ]
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(checkpoint_dir, "checkpoint")

    datamodule = UnlabelledDataModule(dataset='miniimagenet',
                                      datapath=data_dir,
                                      split='test',
                                      img_size_orig=(84, 84),
                                      img_size_crop=(84, 84),
                                      eval_ways=5,
                                      eval_support_shots=5,
                                      eval_query_shots=15)

    model = CLRGAT(**config)
    trainer = pl.Trainer(**kwargs)

    trainer.fit(model, datamodule=datamodule)


def tune_gat_clr_pbt(num_samples=50, num_epochs=10, gpus_per_trial=1, data_dir="~/data"):
    config = {
        "arch": "conv4",
        "out_planes": tune.choice([64, [96, 128, 256, 512]]),
        "average_end": tune.choice([True, False]),
        "scl": False,
        "distance": tune.choice(["euclidean", "cosine"]),
        "att_feat_dim": 80,
        "gnn_type": "gat",
        "optim": "adam",
        "lr_sch": tune.choice(["step", "cos"]),
        "warmup_start_lr": 1e-3,
        "warmup_epochs": 250,
        "sup_finetune_lr": tune.loguniform(1e-6, 1e-3),
        "sup_finetune": "prototune",  # [prototune, std_proto, label_cleansing, sinkhorn]
        "sup_finetune_epochs": tune.uniform(15, 25),
        "eta_min": 5e-5,
        "lr": tune.loguniform(1e-6, 1e-3),
        "lr_decay_step": 25000,
        "lr_decay_rate": 0.5,
        "weight_decay": 6.059722614369727e-06,
        "dataset": "miniimagenet",
        "img_orig_size": (84, 84),
        "batch_size": 64,
        "n_support": 1,
        "n_query": 3,
        "use_projector": False,
        "projector_h_dim": 6400,
        "projector_out_dim": 1600,
        "use_hms": False,
        "label_cleansing_opts": {
            "use": False,
            "n_test_runs": 1000,
            "n_ways": 5,
            "n_shots": 5,
            "n_queries": 15,
            "unbalanced": False,
            "reduce": None,  # ['isomap', 'itsa', 'mds', 'lle', 'se', 'pca', 'none']
            "inference_semi": "transductive",  # ['transductive', 'inductive', 'inductive_sk']
            "d": 5,
            "alpha": 0.8,
            "K": 20,  # neighbours used for manifold creation
            "T": 3,  # power to raise probs matrix before sinkhorn algorithm
            "lr": 0.00001,  # learning rate of fine-tuning
            "denoising_iterations": 1000,
            "beta_pt": 0.5,  # power transform power
            "best_samples": 3,  # number of best samples per class chosen for pseudolabels
            "semi_inference_method": "transductive",  # ['transductive', 'inductive']
            "sinkhorn_iter": 1,
            "use_pt": True,  # [True, False]
        },
        "mpnn_loss_fn": "ce",
        "mpnn_dev": "cuda",
        "mpnn_opts": {
            "_use": True,
            "loss_cnn": True,
            "scaling_ce": tune.loguniform(1e-1, 1e0),
            "adapt": tune.choice(["ot", "instance"]),
            "temperature": tune.loguniform(1e-1, 1e0),
            "output_train_gnn": "plain",
            "graph_params": {
                "sim_type": "correlation",
                "thresh": "no",  # 0
                "set_negative": "hard"},
            "gnn_params": {
                "pretrained_path": "no",
                "red": 1,
                "cat": 0,
                "every": 0,
                "gnn": {
                    "num_layers": tune.randint(1, 4),
                    "aggregator": tune.choice(["add", "max", "mean"]),
                    "num_heads": tune.randint(1, 8),
                    "attention": "dot",
                    "mlp": 1,
                    "dropout_mlp": 0.1,
                    "norm1": 1,
                    "norm2": 1,
                    "res1": 1,
                    "res2": 1,
                    "dropout_1": 0.1,
                    "dropout_2": 0.1,
                    "mult_attr": 0},
                "classifier": {
                    "neck": 0,
                    "num_classes": 0,
                    "dropout_p": 0.4,
                    "use_batchnorm": 0}}
        },
        "feature_extractor": None
    }

    scheduler = PopulationBasedTraining(
        perturbation_interval=5,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-4, 1e-2),
        })

    reporter = CLIReporter(
        parameter_columns=["lr", "optim", "lr_sch", "out_planes"],
        metric_columns=["val/loss", "val/accuracy", "training_iteration"]
    )

    analysis = tune.run(
        tune.with_parameters(
            train_gat_clr_tune_checkpoint,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            data_dir=data_dir),
        resources_per_trial={
            "cpu": 6,
            "gpu": gpus_per_trial
        },
        metric="val/accuracy",
        mode="max",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_gat_clr_pbt",
        local_dir="./ray_results/"
    )

    # msg.info(f"Best hyperparameters found were: {analysis.best_config}")
    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    redis_password = sys.argv[1]
    num_cpus = int(sys.argv[2])
    with msg.loading("Init Ray"):
        ray.init(address=os.environ["ip_head"])
    tune_gat_clr_pbt(100, num_epochs=15, gpus_per_trial=1,
                     data_dir="/home/nfs/oshirekar/unsupervised_ml/data/miniimagenet_84/")
