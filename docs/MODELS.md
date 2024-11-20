# ALDI model zoo

This file contains models trained for the experiments in [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029). All models were trained on 8 NVIDIA Tesla V100s with PyTorch 1.13.1 and CUDA 11.6. 

For compatibility with the config files we provide, download any models here to the `models` directory in this repo.

### Detectron2 Pretrained models

Here we provide links to models from the [Detectron2 model zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md) that we use for pre-training. All baseline and oracle models in our experiments start with these weights. Note we did not train these models, but provide links here for convenience.

**COCO pretrained Mask R-CNN w/ Res50-FPN backbone and 3x schedule:** [Config](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) [Model](https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl)

### Baselines / Burn-in checkpoints

Here we provide checkpoints for baselines trained on source-only data for each benchmark (Cityscapes &rarr; Foggy Cityscapes, Sim10k &rarr; Cityscapes, CFC Kenai &rarr; Channel).

We also use these checkpoints to initialize domain adaptive training, i.e. they also represent the end of the "burn-in" period.

#### Cityscapes &rarr; Foggy Cityscapes

| Backbone      | Download links |
| ----------- | ----------- |
| Resnet50 w/ FPN | [Config](https://github.com/justinkay/aldi/blob/main/configs/cityscapes/Base-RCNN-FPN-Cityscapes_strongaug_ema.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_baseline_strongaug_ema_foggy_val_model_best_591_ema2model.pth) |
| VitDet-B | [Config](https://github.com/justinkay/aldi/blob/main/configs/cityscapes/Base-RCNN-VitDetB-Cityscapes_strongaug_ema.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_vitdetb_baseline_strongaug_ema_foggy_val_model_best_4799_ema2model.pth) |

#### Sim10k &rarr; Cityscapes

| Backbone      | Download links |
| ----------- | ----------- |
| Resnet50 w/ FPN | [Config](https://github.com/justinkay/aldi/blob/main/configs/sim10k/Base-RCNN-FPN-Sim10k_strongaug_ema.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/sim10k_baseline_strongaug_ema_cityscapes_cars_val_model_best_768_ema2model.pth) |
| VitDet-B | [Config](https://github.com/justinkay/aldi/blob/main/configs/sim10k/Base-RCNN-VitDetB-Sim10k_strongaug_ema.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/sim10k_vitdetb_baseline_strongaug_ema_cityscapes_cars_val_model_best_817_ema2model.pth) |

#### CFC Kenai &rarr; Channel

| Backbone      | Download links |
| ----------- | ----------- |
| Resnet50 w/ FPN | [Config](https://github.com/justinkay/aldi/blob/main/configs/cfc/Base-RCNN-FPN-CFC_strongaug_ema.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/cfc_channel_test_model_best_strongaug_ema_667_ema2model.pth) |
| VitDet-B | [Config](https://github.com/justinkay/aldi/blob/main/configs/cfc/Base-RCNN-VitDetB-CFC_strongaug_ema.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/cfc_vitdetb_channel_test_model_best_strongaug_ema_690_ema2model.pth) |


### Final models

Here we provide the models trained using ALDI++.

#### Cityscapes &rarr; Foggy Cityscapes

| Backbone      | Download links |
| ----------- | ----------- |
| Resnet50 w/ FPN | [Config](https://github.com/justinkay/aldi/blob/main/configs/cityscapes/ALDI-Best-Cityscapes.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/aldi++_cityscapes_foggy_val_model_best.pth) [Log](https://github.com/justinkay/aldi/releases/download/v0.0.1/aldi++_cs_to_fcs_log.txt) |
| VitDet-B | [Config](https://github.com/justinkay/aldi/blob/main/configs/cityscapes/ALDI-Best-ViT-Cityscapes.yaml) [Model--TODO]() |

#### Sim10k &rarr; Cityscapes

| Backbone      | Download links |
| ----------- | ----------- |
| Resnet50 w/ FPN | [Config](https://github.com/justinkay/aldi/blob/main/configs/sim10k/ALDI-Best-Sim10k.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/aldi++_sim10k_cityscapes_cars_val_model_best.pth) [Log](https://github.com/justinkay/aldi/releases/download/v0.0.1/aldi++_sim10k_log.txt) |

#### CFC Kenai &rarr; Channel

| Backbone      | Download links |
| ----------- | ----------- |
| Resnet50 w/ FPN | [Config](https://github.com/justinkay/aldi/blob/main/configs/cfc/ALDI-Best-CFC.yaml) [Model](https://github.com/justinkay/aldi/releases/download/v0.0.1/aldi++_cfc_channel_test_model_best.pth) [Log](https://github.com/justinkay/aldi/releases/download/v0.0.1/aldi++_cfc_log.txt) |
