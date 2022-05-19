import math
import os
import random
import string
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import umap
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import Callback
from scipy import stats
from torchvision.utils import make_grid
from tqdm import tqdm


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)


def random_string(letter_count=4, digit_count=4):
    tmp_random = random.Random(time.time())
    rand_str = "".join((tmp_random.choice(string.ascii_lowercase) for x in range(letter_count)))
    rand_str += "".join((tmp_random.choice(string.digits) for x in range(digit_count)))
    rand_str = list(rand_str)
    tmp_random.shuffle(rand_str)
    return "".join(rand_str)


class AutoUMAP(Callback):
    def __init__(
            self,
            args: Namespace,
            logdir: Union[str, Path] = Path("auto_umap"),
            frequency: int = 1,
            keep_previous: bool = False,
            color_palette: str = "hls",
    ):
        """UMAP callback that automatically runs UMAP on the validation dataset and uploads the
        figure to wandb.

        Args:
            args (Namespace): namespace object containing at least an attribute name.
            logdir (Union[str, Path], optional): base directory to store checkpoints.
                Defaults to Path("auto_umap").
            frequency (int, optional): number of epochs between each UMAP. Defaults to 1.
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
            keep_previous (bool, optional): whether to keep previous plots or not.
                Defaults to False.
        """

        super().__init__()

        self.args = args
        self.logdir = Path(logdir)
        self.frequency = frequency
        self.color_palette = color_palette
        self.keep_previous = keep_previous

    @staticmethod
    def add_auto_umap_args(parent_parser: ArgumentParser):
        """Adds user-required arguments to a parser.

        Args:
            parent_parser (ArgumentParser): parser to add new args to.
        """

        parser = parent_parser.add_argument_group("auto_umap")
        parser.add_argument("--auto_umap_dir", default=Path("auto_umap"), type=Path)
        parser.add_argument("--auto_umap_frequency", default=1, type=int)
        return parent_parser

    def initial_setup(self, trainer: pl.Trainer):
        """Creates the directories and does the initial setup needed.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        if trainer.logger is None:
            if self.logdir.exists():
                existing_versions = set(os.listdir(self.logdir))
            else:
                existing_versions = []
            version = "offline-" + random_string()
            while version in existing_versions:
                version = "offline-" + random_string()
        else:
            version = str(trainer.logger.version)
        if version is not None:
            self.path = self.logdir / version
            self.umap_placeholder = f"{self.args.name}-{version}" + "-ep={}.pdf"
        else:
            self.path = self.logdir
            self.umap_placeholder = f"{self.args.name}" + "-ep={}.pdf"
        self.last_ckpt: Optional[str] = None

        # create logging dirs
        if trainer.is_global_zero:
            os.makedirs(self.path, exist_ok=True)

    def on_train_start(self, trainer: pl.Trainer, _):
        """Performs initial setup on training start.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        self.initial_setup(trainer)

    def plot(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the module.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
            module (pl.LightningModule): current module object.
        """

        device = module.device
        data = []
        Y = []

        # set module to eval model and collect all feature representations
        module.eval()
        with torch.no_grad():
            for x, y in trainer.val_dataloaders[0]:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats = module(x)["feats"]

                feats = gather(feats)
                y = gather(y)

                data.append(feats.cpu())
                Y.append(y.cpu())
        module.train()

        if trainer.is_global_zero and len(data):
            data = torch.cat(data, dim=0).numpy()
            Y = torch.cat(Y, dim=0)
            num_classes = len(torch.unique(Y))
            Y = Y.numpy()

            data = umap.UMAP(n_components=2).fit_transform(data)

            # passing to dataframe
            df = pd.DataFrame()
            df["feat_1"] = data[:, 0]
            df["feat_2"] = data[:, 1]
            df["Y"] = Y
            plt.figure(figsize=(9, 9))
            ax = sns.scatterplot(
                x="feat_1",
                y="feat_2",
                hue="Y",
                palette=sns.color_palette(self.color_palette, num_classes),
                data=df,
                legend="full",
                alpha=0.3,
            )
            ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
            ax.tick_params(left=False, right=False, bottom=False, top=False)

            # manually improve quality of imagenet umaps
            if num_classes > 100:
                anchor = (0.5, 1.8)
            else:
                anchor = (0.5, 1.35)

            plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
            plt.tight_layout()

            if isinstance(trainer.logger, pl.loggers.WandbLogger):
                wandb.log(
                    {"validation_umap": wandb.Image(ax)},
                    commit=False,
                )

            # save plot locally as well
            epoch = trainer.current_epoch  # type: ignore
            plt.savefig(self.path / self.umap_placeholder.format(epoch))
            plt.close()

    def on_validation_end(self, trainer: pl.Trainer, module: pl.LightningModule):
        """Tries to generate an up-to-date UMAP visualization of the features
        at the end of each validation epoch.

        Args:
            trainer (pl.Trainer): pytorch lightning trainer object.
        """

        epoch = trainer.current_epoch  # type: ignore
        if epoch % self.frequency == 0 and not trainer.sanity_checking:
            self.plot(trainer, module)


class OfflineUMAP:
    def __init__(self, color_palette: str = "hls"):
        """Offline UMAP helper.

        Args:
            color_palette (str, optional): color scheme for the classes. Defaults to "hls".
        """

        self.color_palette = color_palette

    def plot(
            self,
            device: str,
            model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            plot_path: str,
    ):
        """Produces a UMAP visualization by forwarding all data of the
        first validation dataloader through the model.
        **Note: the model should produce features for the forward() function.

        Args:
            device (str): gpu/cpu device.
            model (nn.Module): current model.
            dataloader (torch.utils.data.Dataloader): current dataloader containing data.
            plot_path (str): path to save the figure.
        """

        data = []
        Y = []

        # set module to eval model and collect all feature representations
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Collecting features"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                feats = model(x)
                data.append(feats.cpu())
                Y.append(y.cpu())
        model.train()

        data = torch.cat(data, dim=0).numpy()
        Y = torch.cat(Y, dim=0)
        num_classes = len(torch.unique(Y))
        Y = Y.numpy()

        print("Creating UMAP")
        data = umap.UMAP(n_components=2).fit_transform(data)

        # passing to dataframe
        df = pd.DataFrame()
        df["feat_1"] = data[:, 0]
        df["feat_2"] = data[:, 1]
        df["Y"] = Y
        plt.figure(figsize=(9, 9))
        ax = sns.scatterplot(
            x="feat_1",
            y="feat_2",
            hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=df,
            legend="full",
            alpha=0.3,
        )
        ax.set(xlabel="", ylabel="", xticklabels=[], yticklabels=[])
        ax.tick_params(left=False, right=False, bottom=False, top=False)

        # manually improve quality of imagenet umaps
        if num_classes > 100:
            anchor = (0.5, 1.8)
        else:
            anchor = (0.5, 1.35)

        plt.legend(loc="upper center", bbox_to_anchor=anchor, ncol=math.ceil(num_classes / 10))
        plt.tight_layout()

        # save plot locally as well
        plt.savefig(plot_path)
        plt.close()


def get_train_images(ds, num):
    return torch.stack([ds[i]['data'][0] for i in range(num)], dim=0)


class GradLogger(pl.Callback):
    def __init__(self, every_n_steps=10):
        self.every_n_steps = every_n_steps

    def on_after_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if trainer.global_step % self.every_n_steps == 0:
            for name, param in pl_module.student.named_parameters():
                if "weight" in name and not "norm" in name and param.requires_grad:
                    pl_module.logger.experiment.log(
                        {f"{name}_grad": wandb.Histogram(param.grad.cpu())}
                    )


class WandbImageCallback(pl.Callback):
    """
    Logs the input and output images of a module.
    """

    def __init__(self, input_imgs, every_n_epochs=5):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                _, reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = make_grid(imgs, nrow=2, )  # normalize=True, range=(-1,1))
            trainer.logger.experiment.log({
                "reconstructions": wandb.Image(grid, caption='Reconstructions'),
                "global_step": trainer.global_step
            })
            # trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


# Cell
class TensorBoardImageCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=5):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                _, reconst_imgs = pl_module(input_imgs)
                pl_module.train()

            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = make_grid(imgs, nrow=2, )  # normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


# Cell
class ConfidenceIntervalCallback(pl.Callback):
    def __init__(self, log_to_wb=False) -> None:
        super().__init__()
        self.losses = []
        self.accuracies = []
        self.log_to_wb = log_to_wb

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
        loss, accuracy = outputs
        self.losses.append(loss)
        self.accuracies.append(accuracy)

    def on_test_end(self, trainer, pl_module) -> None:
        conf_interval = stats.t.interval(0.95, len(self.accuracies) - 1, loc=np.mean(self.accuracies),
                                         scale=stats.sem(self.accuracies))
        print(f"Confidence Interval: {conf_interval}")
        plt.ylabel("Average Test Accuracy")
        plt.errorbar([1], np.mean(self.accuracies), yerr=np.std(self.accuracies), fmt='o', color='black',
                     ecolor='lightgray', elinewidth=3, capsize=0)
        if self.log_to_wb:
            wandb.log({'Confidence Interval': conf_interval})
            wandb.log({
                'Average Test Accuracy with std dev': wandb.Image(plt)
            })
