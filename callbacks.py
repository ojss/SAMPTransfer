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
    def __init__(self, every_n_steps=10, topk=10):
        super(EmbeddingLogger, self).__init__()
        self.every_n_steps = every_n_steps
        self.topk = 10

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        if trainer.global_step % self.every_n_steps == 0:
            valid_embeds = pl_module.teacher(batch["test"][0][0])
            valid_embeds = [pred for pred in valid_embeds]
            columns = ["image"] + [f"closest_{i + 1}" for i in range(self.topk)]
            indices = np.random.choice(len(self.valid_files), VALID_IMAGES, replace=False)


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
