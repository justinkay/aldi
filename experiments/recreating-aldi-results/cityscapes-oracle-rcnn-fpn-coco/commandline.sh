#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here.
# Link name MUST be 'datasets' to be compatible with ALDI
ln -sfT ~/DATA datasets

# Run experiment
python ../../../aldi/tools/train_net.py --config input/OracleT-RCNN-FPN-Cityscapes_strongaug_ema_corrected.yaml --num-gpus 4