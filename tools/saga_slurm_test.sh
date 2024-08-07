#!/bin/bash

#SBATCH --account=nn10058k --job-name=benthic_daod  # create a short name for your job
#SBATCH --partition=accel --gpus=1
#SBATCH --time=2:0:0
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=24       # no of threads
#SBATCH --mem=3G                 # memory per node
#SBATCH --output=slurm/slurm-%j.out           # slurm output files
#SBATCH --error=slurm/slurm-%j.err
#SBATCH --qos=devel


# it is good to have the following lines in any bash script
set -o errexit  # make bash exit on any error
set -o nounset  # treat unset variables as errors

# setup distributed training
echo -n $SLURM_JOB_ID
echo

PORT=$(echo -n $SLURM_JOB_ID | tail -c 4)
echo $PORT
echo ""

export LEADER_PORT=$((10000 + $PORT))
echo $LEADER_PORT

export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))
export LEADER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo -n "WORLD_SIZE="$WORLD_SIZE
echo -n "LEADER_ADDR="$LEADER_ADDR

# use both GPUs
export CUDA_VISIBLE_DEVICES="0"

# configuration values

# setup module system
module --quiet purge
module load Anaconda3/2022.10
source ${EBROOTANACONDA3}/bin/activate

export CONDA_PKGS_DIRS=/cluster/projects/nn10058k/hdoi5324_daod/conda/package-cache
echo -n ${CONDA_PKGS_DIRS}

CONDA_ENV="/cluster/projects/nn10058k/hdoi5324_daod/conda/py310_torch"

#conda create --prefix ${CONDA_ENV}  python=3.10 -y

conda activate ${CONDA_ENV}
echo -n "Activated env"
echo -n "***"

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y 
echo "***"
echo -n "Installed torch"

pip install neptune
echo -n "Installed neptune"

#conda install conda-forge::detectron2 -y 
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
echo "****"
echo -n "Installed detectron2"

conda install -c conda-forge neptune-detectron2 -y
echo -n "Finished"