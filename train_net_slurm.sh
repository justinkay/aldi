#!/bin/bash
#SBATCH --job-name=daod-strong-baseline  # short name for the job
#SBATCH --partition=xeon-g6-volta        # partition to run on
#SBATCH --exclusive
#SBATCH --ntasks=4                       # total task count
#SBATCH --nodes=4                        # node count
#SBATCH --ntasks-per-node=1              # total number of tasks per node
#SBATCH --cpus-per-task=40               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G
#SBATCH --gres=gpu:volta:2               # number of gpus per node
#SBATCH --mail-type=begin                # send email when job begins
#SBATCH --mail-type=end                  # send email when job ends
#SBATCH --mail-user=haucke@mit.edu

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

srun -N4 ./train_net_slurm_instance.sh