import shutil
import tempfile
import uuid
from pathlib import Path

import submitit
import sys
from omegaconf import OmegaConf

import protoclr_obow

UUID = uuid.uuid4()
OmegaConf.register_new_resolver("uuid", lambda: str(UUID))


def make_conf_file_copy(path):
    fd, tmp_path = tempfile.mkstemp()
    shutil.copy2(path, tmp_path)
    return tmp_path


def parse_args():
    # TODO make this nicer somehow
    path = sys.argv[2]
    tmp_path = make_conf_file_copy(path)

    conf = OmegaConf.load(tmp_path)
    return conf, tmp_path


def main():
    cfg, cfg_path = parse_args()
    args = cfg["slurm"]
    ngpus = int(args["slurm_additional_parameters"]["gres"][-1])
    exp_dir = Path(f'./expts/{cfg["job_name"]}')

    exp_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=exp_dir)

    executor.update_parameters(
        name=cfg["job_name"],
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

    job = executor.submit(protoclr_obow.slurm_main, cfg_path, UUID)
    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
