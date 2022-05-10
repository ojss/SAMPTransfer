#!/bin/bash

#SBATCH --gres=gpu:a40:1
# You can control the resources and scheduling with '#SBATCH' settings
# (see 'man sbatch' for more information on setting these parameters)

# The default partition is the 'general' partition
#SBATCH --partition=general

# The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
#SBATCH --qos=medium

# The default run (wall-clock) time is 1 minute
#SBATCH --time=00-07:00:00

# The default number of parallel tasks per job is 1
#SBATCH --ntasks=1

# Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
# The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)
#SBATCH --cpus-per-task=8

# The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
#SBATCH --mem=12000

# Set mail type to 'END' to receive a mail when the job finishes
# Do not enable mails when submitting large numbers (>20) of jobs at once
#SBATCH --mail-type=END

# 90 seconds before training ends, to help create a checkpoint and requeue the job
#SBATCH --signal=SIGUSR1@90

module use /opt/insy/modulefiles

module load cuda/11.3
module load miniconda/3.9

echo "Running Algo:"
echo "$1"
echo "With config:"
echo "$2"
source activate /home/nfs/oshirekar/unsupervised_ml/ai2
python "$1" --config "$2"
