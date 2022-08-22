from typing import Optional
from wasabi import msg

import pytorch_lightning as pl
import torch.cuda
import typer
from omegaconf import OmegaConf

from callbacks import ConfidenceIntervalCallback
from clr_gat import CLRGAT
from dataloaders.dataloaders import UnlabelledDataModule

app = typer.Typer()


# pl.seed_everything(72)


@app.command()
def clrgat(dataset: str, ckpt_path: str, datapath: str, eval_ways: int, eval_shots: int, query_shots: int,
           sup_finetune: str, config: Optional[str] = None, adapt: str = "ot", distance: str = "euclidean"):
    if torch.cuda.is_available():
        map_location = "cuda"
    else:
        map_location = "cpu"
    msg.divider("Eval Setting")
    msg.info(f"Eval Ways: {eval_ways}")
    msg.info(f"Eval Shots: {eval_shots}")
    if config is not None:
        cfg = OmegaConf.load(config)
    msg.divider("Model Setup")
    with msg.loading("Loading model"):
        # uncomment for older checkpoints
        model = CLRGAT.load_from_checkpoint(checkpoint_path=ckpt_path,
                                            mpnn_dev=map_location,
                                            arch="conv4",
                                            # out_planes=64,
                                            # average_end=False,
                                            label_cleansing_opts={
                                                "use": False,
                                            },
                                            distance=distance,
                                            use_hms=False,
                                            use_projector=False,
                                            projector_h_dim=2048,
                                            projector_out_dim=256,
                                            eval_ways=eval_ways,
                                            sup_finetune=sup_finetune,
                                            sup_finetune_epochs=25,
                                            # map_location=map_location,
                                            hparams_file=config)
        model.mpnn_opts["adapt"] = adapt

    msg.divider("Adaptation Type")
    msg.info(f" Adaptation type in use: {model.mpnn_opts['adapt']}")
    with msg.loading("Loading datamodule"):
        datamodule = UnlabelledDataModule(dataset=dataset,
                                          datapath=datapath,
                                          split='test',
                                          img_size_orig=(84, 84) if dataset == 'miniimagenet' else (28, 28),
                                          img_size_crop=(60, 60),
                                          eval_ways=eval_ways,
                                          eval_support_shots=eval_shots,
                                          eval_query_shots=query_shots)
    msg.divider("Beginning testing")
    trainer = pl.Trainer(
        gpus=-1,
        limit_test_batches=600,
        callbacks=[ConfidenceIntervalCallback(log_to_wb=False)]
    )
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    app()
