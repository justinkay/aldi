#!/bin/sh

# Point to your dataset folder containing both images and COCO formatted annotations here
ln -sfT ~/DATA datasets

# Run experiment
python ../../../aldi/tools/train_net.py --config input/OracleT-RCNN-FPN-Cityscapes_strongaug_ema_corrected.yaml --num-gpus 4