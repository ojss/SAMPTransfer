# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import uuid
from pathlib import Path

import submitit
from omegaconf import OmegaConf

import protoclr_obow
from cli import custom_cli
from dataloaders import UnlabelledDataModule

UUID = uuid.uuid4()
OmegaConf.register_new_resolver("uuid", lambda: str(UUID))


def parse_args():
    cli = custom_cli.MyCLI(protoclr_obow.PCLROBoW, UnlabelledDataModule,
                           run=False,
                           save_config_overwrite=True,
                           save_config_filename=str(UUID),
                           parser_kwargs={"parser_mode": "omegaconf"})
    return cli


def main():
    cli = parse_args()
    args = cli.config["slurm"]
    ngpus = int(args["slurm_additional_parameters"]["gres"][-1])
    exp_dir = Path(f'./expts/{str(UUID)}')

    exp_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=exp_dir)

    executor.update_parameters(
        name=cli.config["job_name"],
        # gpus_per_node=args["ngpus"],
        mem_gb=12 * ngpus,
        tasks_per_node=ngpus,
        cpus_per_task=8,
        nodes=args["nodes"],
        timeout_min=args["timeout"] * 60,
        slurm_partition=args["partition"],
        slurm_signal_delay_s=120,
        slurm_constraint=args["constraint"],
        slurm_comment=args["comment"],
        slurm_additional_parameters=args["slurm_additional_parameters"]
    )

    job = executor.submit(protoclr_obow.slurm_main, cli)
    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
