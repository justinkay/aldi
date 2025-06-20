# <div align="center"> Align and Distill: Unifying and Improving <br>Domain Adaptive Object Detection</div>

<div align="center">
 
Official codebase for [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029) (TMLR 2025).

\[[Project Page](https://aldi-daod.github.io/)\] \[[Arxiv](https://arxiv.org/abs/2403.12029)\] \[[PDF](https://arxiv.org/pdf/2403.12029.pdf)\] 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/align-and-distill-unifying-and-improving/unsupervised-domain-adaptation-on-cityscapes-1)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-cityscapes-1?p=align-and-distill-unifying-and-improving) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/align-and-distill-unifying-and-improving/unsupervised-domain-adaptation-on-sim10k-to-3)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-sim10k-to-3?p=align-and-distill-unifying-and-improving) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/align-and-distill-unifying-and-improving/unsupervised-domain-adaptation-on-cfc-daod)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-cfc-daod?p=align-and-distill-unifying-and-improving)

![](docs/aldi_banner_4.png)

</div>

## Updates

- **03/17/2025:** Accepted to [Transactions on Machine Learning Research](https://jmlr.org/tmlr/papers/) with a Featured Certification (Spotlight)! New additions include:

    - ALDI++ implementations for [DETR](https://github.com/justinkay/aldi/tree/main/aldi/detr) and [YOLO](https://github.com/justinkay/aldi/tree/main/aldi/yolo), which are SOTA for DETR and YOLO-based DAOD without additional hyperparameter tuning.
    - Larger [backbones](https://github.com/justinkay/aldi/blob/main/aldi/backbone.py) (Vit-L, ConvNeXt).
    <!-- - Easier access to [prior work]() on the `main` branch. -->

## Documentation

Align and Distill (ALDI) is a state-of-the-art framework for domain adaptive object detection (DAOD). 

ALDI is built on top of the [Detectron2](https://github.com/facebookresearch/detectron2/) object detection library and follows the same design patterns where possible. In particular, training settings are managed by [config files](configs), datasets are managed by a [dataset registry](aldi/datasets.py), training is handled by a custom [`Trainer`](aldi/trainer.py) class that extends [`DefaultTrainer`](https://github.com/facebookresearch/detectron2/blob/0ae803b1449cd2d3f8fa1b7c0f59356db10b3083/detectron2/engine/defaults.py#L323), and we provide a training script in [tools/train_net.py](tools/train_net.py) that comes with [all the same functionality](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) as the [Detectron2 script](https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py) by the same name.


<!-- This codebase provides implementations of the following DAOD methods:

| Method | Architecture | Code | Config(s) |
|---|---|---|---|
ALDI++ (Ours) | Faster R-CNN, DETR, YOLO, VitDet | |
Adaptive Teacher | Faster R-CNN | | 
MIC | Faster R-CNN | |
Probabilistic teacher | Faster R-CNN | |
SADA | Faster R-CNN | |  -->

<summary><h3>Install</h3></summary>

**1. Install PyTorch and torchvision.** Follow the [official install guide](https://pytorch.org/get-started/locally/) to install the correct versions for your CUDA version.

**2. Install ALDI.** Clone this repository and run:

```bash
pip install -e .
```

Optionally include the `--no-cache-dir` flag if you run into OOM issues.

<details closed>
<summary><b>3. (Optional) Install YOLO and DETR dependencies</b></summary>

Within the ALDI directory, run

```
git submodule update --init --recursive
```

to clone the relevant submodules for YOLO and DETR. 

For YOLO, some additional libraries are needed:

```
pip install pandas requests ipython psutil seaborn
```

For DETR, you must (1) perform an additional step to build the custom `MultiScaleDeformableAttention` operation. Note that CUDA/GPU access is required to run this script. **This has been successfully tested with PyTorch 2.5.1 but has known issues with 2.6.0.**

```
cd aldi/detr/libs/DeformableDETRDetectron2/deformable_detr/models/ops
bash make.sh
```

And (2), uncomment the following lines in `tools/train_net.py`

```
from aldi.detr.helpers import add_deformable_detr_config
import aldi.detr.align # register align mixins with Detectron2
import aldi.detr.distill # register distillers and distill mixins with Detectron2
add_deformable_detr_config(cfg)
```
</details>


<summary><h3>Data setup</h3></summary>

There are three kinds of "datasets" in domain adaptive object detection:

| Train (source) | Unlabeled (target) | Test (source or target) |
| -------- | -------- | -------- |
| Labeled source-domain images. Used for source-only baseline training, supervised burn-in, and domain-adaptive training. | Target-domain images that are optionally labeled. If unlabeled, used for domain-adaptive training only. If labeled, can be used to train "oracle" methods. Any labels are ignored during domain-adaptive training. | A labeled source- or target-domain validation set. In most DAOD papers this comes from the target domain, even though this breaks the constraints of unsupervised domain adaptation. |




**Set up DAOD benchmarks (Cityscapes, Sim10k, CFC)**


 
Follow [these instructions](docs/DATASETS.md) to set up data and reproduce benchmark results on the datasets in [our paper](https://arxiv.org/abs/2403.12029): Cityscapes &rarr; Foggy Cityscapes, Sim10k &rarr; Cityscapes, and CFC Kenai &rarr; Channel.


<details closed>

 <summary><b>Custom datasets (use your own data)</b></summary>

The easiest way to use your own dataset is to create a [COCO-formatted JSON files](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html) and [register your datasets with Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#register-a-coco-format-dataset). You will register each separately:

```python
# add this to the top of tools/train_net.py or aldi/datasets.py
from detectron2.data.datasets import register_coco_instances
register_coco_instances("your_train_dataset_name", {}, "path/to/your_train_coco_labels.json", "path/to/your/train/images/")
register_coco_instances("your_unlabeled_dataset_name", {}, "path/to/your_unlabeled_coco_labels.json", "path/to/your/unlabeled/images/")
register_coco_instances("your_test_dataset_name", {}, "path/to/your_test_coco_labels.json", "path/to/your/test/images/")
```

Note that by default Detectron2 assumes all paths are relative to `./datasets` relative to your current working directory. You can change this location if desired using the `DETECTRON2_DATASETS` environment variable, e.g.: `export DETECTRON2_DATASETS=/path/to/datasets`.

</details>


<summary><h3>Training</h3></summary>

See our [detailed training instructions](docs/TRAINING.md). The TL;DR is:
 
**Config setup**

Training is managed through [config files](configs/). We provide example configs for burn-in/baseline models, oracle models, and ALDI++.

You will need to modify (at least) the following values for any custom data:

```
DATASETS:
  TRAIN: ("your_training_dataset_name",) # needs to be a tuple, and can contain multiple datasets if you want
  UNLABELED: ("your_unlabeled_dataset_name",) # needs to be a tuple, and can contain multiple datasets if you want
  TEST: ("your_test_dataset_name",)  # needs to be a tuple, and can contain multiple datasets if you want
```

```
MODEL:
  ROI_HEADS:
    NUM_CLASSES: 9 # change to match your number of classes
```

**Run training**

ALDI involves two training phases: (1) burn-in, (2) domain adaptation. Again, please reference the [detailed training instructions](docs/TRAINING.md). Training involves running [tools/train_net.py](../tools/train_net.py) for each training phase:

```
python tools/train_net.py --config path/to/your/config.yaml
```

The script is compatible with all [Detectron2 training options]((https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html#training-evaluation-in-command-line)) (`--num-gpus`, in-line config modifications, etc.).


<summary><h3>Evaluation</h3></summary>

After training, to run evaluation with your model:

```
python tools/train_net.py --eval-only --config-file path/to/your/aldi_config.yaml MODEL.WEIGHTS path/to/your/model_best.pth
```


## Model zoo

We provide burn-in checkpoints and final models for DAOD benchmarks (Cityscapes &rarr; Foggy Cityscapes, Sim10k &rarr; Cityscapes, and CFC Kenai &rarr; Channel) in [the model zoo](docs/MODELS.md).

You can download the required model weights for any config file we provide using 

```bash
python tools/download_model_for_config.py --config-file path/to/config.yaml
```

## Extras and prior work

The [main](/justinkay/aldi/tree/main) branch contains all you need to run ALDI, and is a good starting point for most.

Additional code and configuration files to reproduce all experiments in [our paper](https://arxiv.org/abs/2403.12029) can be found on the [extras](/justinkay/aldi/tree/extras) branch.

## Reference

#### [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029)

[Justin Kay](https://justinkay.github.io), [Timm Haucke](https://timm.haucke.xyz/), [Suzanne Stathatos](https://suzanne-stathatos.github.io/), [Siqi Deng](https://www.amazon.science/author/siqi-deng), [Erik Young](https://home.tu.org/users/erikyoung), [Pietro Perona](https://scholar.google.com/citations?user=j29kMCwAAAAJ), [Sara Beery](https://beerys.github.io/), and [Grant Van Horn](https://gvanhorn38.github.io/).

Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real-world data, and (4) A new method, ALDI++, that achieves state-of-the-art results by a large margin. ALDI++ outperforms the previous state-of-the-art by +3.5 AP50 on Cityscapes to Foggy Cityscapes, +5.7 AP50 on Sim10k to Cityscapes (where ours is the only method to outperform a fair baseline), and +0.6 AP50 on CFC Kenai to Channel. ALDI and ALDI++ are architecture-agnostic, setting a new state-of-the-art for YOLO and DETR-based DAOD as well without additional hyperparameter tuning. Our framework, dataset, and state-of-the-art method offer a critical reset for DAOD and provide a strong foundation for future research.

If you find our work useful in your research please consider citing our paper:

```
@article{
kay2025align,
title={Align and Distill: Unifying and Improving Domain Adaptive Object Detection},
author={Justin Kay and Timm Haucke and Suzanne Stathatos and Siqi Deng and Erik Young and Pietro Perona and Sara Beery and Grant Van Horn},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=ssXSrZ94sR},
note={Featured Certification}
}
```
