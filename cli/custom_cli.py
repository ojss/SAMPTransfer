import pytorch_lightning as pl
from jsonargparse import lazy_instance
from pytorch_lightning.utilities.cli import LightningCLI

from bow.feature_extractor import CNN_4Layer


class MyCLI(LightningCLI):
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

        parser.add_argument("bow_extractor_opts.inv_delta", default=15)
        parser.add_argument("bow_extractor_opts.num_words", default=8192)

        parser.add_argument("bow_predictor_opts.kappa", default=8)

        parser.add_argument("job_name", default="local_dev_run", type=str, help="Job name")
        parser.add_argument(
            "slurm.nodes", default=1, type=int, help="Number of nodes to request"
        )
        # parser.add_argument(
        #     "slurm.ngpus", default=1, type=int, help="Number of gpus to request on each node"
        # )
        parser.add_argument(
            "slurm.timeout", default=72, type=int, help="Duration of the job, in hours"
        )
        parser.add_argument(
            "slurm.partition", default="general", type=str, help="Partition where to submit"
        )
        parser.add_argument("slurm.slurm_additional_parameters", type=dict)
        parser.add_argument(
            "slurm.constraint",
            default="",
            type=str,
            help="Slurm constraint. Use 'volta32gb' for Tesla V100 with 32GB",
        )
        parser.add_argument(
            "slurm.comment",
            default="",
            type=str,
            help="Comment to pass to scheduler, e.g. priority message")
