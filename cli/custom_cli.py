from typing import Optional, Type, Union, Callable, Dict, Any

import pytorch_lightning as pl
from jsonargparse import lazy_instance
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback

from bow.feature_extractor import CNN_4Layer


class MyCLI(LightningCLI):
    def __init__(self, model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
                 datamodule_class: Optional[
                     Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
                 save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
                 save_config_filename: str = "config.yaml",
                 save_config_overwrite: bool = True,
                 save_config_multifile: bool = False,
                 trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
                 trainer_defaults: Optional[Dict[str, Any]] = None,
                 seed_everything_default: Optional[int] = None,
                 description: str = "pytorch-lightning trainer command line tool",
                 env_prefix: str = "PL",
                 env_parse: bool = False,
                 parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
                 subclass_mode_model: bool = False,
                 subclass_mode_data: bool = False,
                 run: bool = True, ):
        super(MyCLI, self).__init__(model_class, datamodule_class, save_config_callback, save_config_filename,
                                    save_config_overwrite, save_config_multifile,
                                    trainer_class, trainer_defaults, seed_everything_default, description, env_prefix,
                                    env_parse, parser_kwargs, subclass_mode_model, subclass_mode_data, run)

    def add_arguments_to_parser(self, parser: pl.utilities.cli.LightningArgumentParser):
        # DEFAULTS
        parser.set_defaults(
            {
                "model.feature_extractor": lazy_instance(CNN_4Layer, in_channels=3, hidden_size=64, out_channels=64),
                "model.bow_clr": False,
                "model.clr_loss": True
            })

        parser.link_arguments("data.dataset", "model.dataset")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.n_support", "model.n_support")
        parser.link_arguments("data.n_query", "model.n_query")
        parser.link_arguments("data.img_size_orig", "model.img_orig_size")

        parser.add_argument("bow_extractor_opts.inv_delta", default=15)
        parser.add_argument("bow_extractor_opts.num_words", default=8192)

        parser.add_argument("bow_predictor_opts.kappa", default=8)
        parser.add_argument("--graph_conv_opts.m_scale.resizing", default='avg')

        parser.add_argument("job_name", default="local_dev_run", type=str, help="Job name")
        parser.add_argument(
            "--slurm.nodes", default=1, type=int, help="Number of nodes to request"
        )
        # parser.add_argument(
        #     "slurm.ngpus", default=1, type=int, help="Number of gpus to request on each node"
        # )
        parser.add_argument(
            "--slurm.timeout", default=72, type=int, help="Duration of the job, in hours"
        )
        parser.add_argument(
            "--slurm.partition", default="general", type=str, help="Partition where to submit"
        )
        parser.add_argument("--slurm.slurm_additional_parameters", type=dict)
        parser.add_argument(
            "--slurm.constraint",
            default="",
            type=str,
            help="Slurm constraint. Use 'volta32gb' for Tesla V100 with 32GB",
        )
        parser.add_argument(
            "--slurm.comment",
            default="",
            type=str,
            help="Comment to pass to scheduler, e.g. priority message")
