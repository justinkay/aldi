#!/bin/bash

# configuration values
CFG="${1:-configs/cityscapes/cityscapes_baseline/Base-RCNN-FPN-Cityscapes.yaml}"

export CUDA_VISIBLE_DEVICES=0,1

conda run --no-capture-output -n daod-strong-baseline python train_net.py --machine-rank $SLURM_PROCID --num-machines $SLURM_JOB_NUM_NODES --num-gpus 2 --dist-url tcp://$MASTER_ADDR:$MASTER_PORT --config-file $CFG
