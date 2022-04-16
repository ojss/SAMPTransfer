import uuid
from pathlib import Path

import submitit
import sys
from omegaconf import OmegaConf

import protoclr_obow

UUID = uuid.uuid4()
OmegaConf.register_new_resolver("uuid", lambda: str(UUID))


def parse_args():
    # TODO make this nicer somehow
    path = sys.argv[2]
    conf = OmegaConf.load(path)
    return conf, path


def main():
    conf, conf_path = parse_args()
    args = conf["slurm"]
    ngpus = int(args["slurm_additional_parameters"]["gres"][-1])
    exp_dir = Path(f'./expts/{conf["job_name"]}')

    exp_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=exp_dir)

    executor.update_parameters(
        name=conf["job_name"],
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

    job = executor.submit(protoclr_obow.slurm_main, conf_path)
    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
