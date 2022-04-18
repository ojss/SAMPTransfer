import copy
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from scipy import stats
from torchvision.utils import make_grid

# Cell
from protoclr_obow import PCLROBoW


def get_train_images(ds, num):
    return torch.stack([ds[i]['data'][0] for i in range(num)], dim=0)


class EmbeddingLogger(pl.Callback):
    def __init__(self, every_n_steps=10):
        super(EmbeddingLogger, self).__init__()
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: PCLROBoW,
            outputs,
            batch: Any,
            batch_idx: int,
            unused: Optional[int] = 0,
    ) -> None:
        if trainer.global_step % self.every_n_steps == 0:
            pl_module.eval()
            z = copy.deepcopy(outputs["embeddings"])
            ways = pl_module.batch_size
            y_query = torch.arange(ways).unsqueeze(
                0).unsqueeze(2)  # batch and shot dim
            y_query = y_query.repeat(1, 1, pl_module.n_query)
            y_query = y_query.view(1, -1).to(pl_module.device)

            y_support = torch.arange(ways).unsqueeze(
                0).unsqueeze(2)  # batch and shot dim
            y_support = y_support.repeat(1, 1, pl_module.n_support)
            y_support = y_support.view(1, -1).to(pl_module.device)
            labels = torch.cat([y_support, y_query], dim=-1)
            z = torch.cat([z, labels.T], dim=-1)
            wandb.log({
                "embeddings": wandb.Table(
                    columns=[f"D{c}" for c in range(z.shape[-1])],
                    data=z.cpu().tolist()
                )
            })
            pl_module.train()


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
