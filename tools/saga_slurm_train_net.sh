#!/bin/bash

#SBATCH --account=nn10058k --job-name=benthic_daod  # create a short name for your job
#SBATCH --partition=a100 --gpus=2
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=16       # no of threads
#SBATCH --mem=20G                 # memory per node
#SBATCH --output=slurm/slurm-%j.out           # slurm output files
#SBATCH --error=slurm/slurm-%j.err

# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

# setup distributed training
echo
echo -n $SLURM_JOB_ID

TRUNC_JOB_ID=$(echo -n $SLURM_JOB_ID | tail -c 4)
echo "Truncated job id with 0 is ${TRUNC_JOB_ID}"
chars="0"
# Use sed to remove the specified characters
PORT=$(echo $TRUNC_JOB_ID | sed "s/[$chars]//g")
echo
echo " and job id without 0 is ${PORT}"

export LEADER_PORT=$((10000 + $PORT))
export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
export LEADER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo $LEADER_PORT $WORLD_SIZE $LEADER_ADDR


# use both GPUs
export CUDA_VISIBLE_DEVICES="0,1"


# setup module system
module --quiet purge
module load Anaconda3/2022.10
CONDA_ENV="../conda/py310_torch"

# configuration values
CFG_DIR="configs/urchininf/"
CFG_FILE="$1" #"Base-RCNN-FPN-urchininf_weakaug.yaml"
CLI=${@:2}
echo "${CFG_FILE} ${CLI}"

# run training script inside anaconda environment
srun -N${SLURM_JOB_NUM_NODES} bash -c "\
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    conda run --no-capture-output --prefix ${CONDA_ENV} \
        python tools/train_net.py \
            --machine-rank \$SLURM_PROCID \
            --num-machines $SLURM_JOB_NUM_NODES \
            --num-gpus 2 \
            --dist-url tcp://$LEADER_ADDR:$LEADER_PORT \
            --config-file ${CFG_DIR}${CFG_FILE} ${CLI}"
