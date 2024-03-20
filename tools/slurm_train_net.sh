#!/bin/bash

#SBATCH --job-name=aldi                       # short name for the job
#SBATCH --partition=xeon-g6-volta             # partition to run on
#SBATCH --exclusive                           # use entire node
#SBATCH --ntasks=4                            # total task count
#SBATCH --nodes=4                             # node count
#SBATCH --ntasks-per-node=1                   # total number of tasks per node
#SBATCH --cpus-per-task=40                    # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=40G                             # memory
#SBATCH --gres=gpu:volta:2                    # type and number of GPUs per node
#SBATCH --output=slurm/slurm-%j.out           # slurm output files

# setup distributed training
export LEADER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
export LEADER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "WORLD_SIZE="$WORLD_SIZE
echo "LEADER_ADDR="$LEADER_ADDR

# use both GPUs
export CUDA_VISIBLE_DEVICES="0,1"

# configuration values
CFG="${1:-configs/cityscapes/Base-RCNN-FPN-Cityscapes_strongaug_ema.yaml}"

# setup SuperCloud module system
module purge
module load anaconda/2023a

# run training script inside anaconda environment
srun -N$SLURM_JOB_NUM_NODES bash -c "\
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    conda run --no-capture-output -n aldi \
        python tools/train_net.py \
            --machine-rank \$SLURM_PROCID \
            --num-machines $SLURM_JOB_NUM_NODES \
            --num-gpus 2 \
            --dist-url tcp://$LEADER_ADDR:$LEADER_PORT \
            --config-file $CFG ${@:2}"
