# Benchmark dataset setup

In order to work our [default config files](../configs), all data is expected to be in the `datasets` directory of this repository in the following directory structure:
```
aldi/
    datasets/
        cityscapes/
            leftImg8bit/
            leftImg8bit_foggy/
            annotations/
                cityscapes_train_instances.json
                ...
        sim10k/
            images/
            coco_car_annotations.json
        cfc/
            images/
                cfc_train/
                cfc_val/
                cfc_channel_train/
                cfc_channel_test/
            coco_labels/
                cfc_train.json
                ...
```

## Cityscapes &rarr; Foggy Cityscapes

**Images:** Downloading Cityscapes and Foggy Cityscapes requires creating an account on the [Cityscapes website](https://www.cityscapes-dataset.com/). This is a multi-step process...

1. Create an  account on the [Cityscapes website](https://www.cityscapes-dataset.com/) and wait to be approved. This will be easier if you have an academic email address.
2. Install the [CityScapesScripts command line utilities](https://github.com/mcordts/cityscapesScripts).
3. Use `csDownload` on the command line to download the `leftImg8bit` and `leftImg8bit_foggy` images, and place them in `aldi/datasets/cityscapes/` as shown above.

**Labels:** For reproducibility, we provide JSON files containing the ground-truth bounding boxes we used for training and evaluation. As noted in our paper, these have differed in past codebases. Our files match the on-the-fly conversion done by Detectron2. See our code [here](../tools/convert_cityscapes_to_coco.py) for reference.

[Cityscapes train labels](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_train_instances.json)

[Cityscapes val labels](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_val_instances.json)

[Foggy Cityscapes train labels](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_train_instances_foggyALL.json)

[Foggy Cityscapes val labels](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_val_instances_foggyALL.json)

## Sim10k &rarr; Cityscapes

**Sim10k images:** Download the Sim10k images [here](https://deepblue.lib.umich.edu/data/downloads/ks65hc58r), and place them in `aldi/datasets/sim10k/images/` as shown above.

**Cityscapes images:** Follow instructions for setting up Cityscapes above. Note you will only need the `leftImg8bit` images, not the foggy ones

**Labels:** For DAOD, Sim10k &rarr; Cityscapes is typically a single-class challenge consisting only of the "car" class. We provide labels postprocessed for this task here:

[Sim10k cars train labels](https://github.com/justinkay/aldi/releases/download/v0.0.1/coco_car_annotations.json)

[Cityscapes cars train labels](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_train_instances_cars.json)

[Cityscapes cars val labels](https://github.com/justinkay/aldi/releases/download/v0.0.1/cityscapes_val_instances_cars.json)

## CFC Kenai &rarr; Channel

**Images:**

[CFC Kenai (source) train images](https://data.caltech.edu/records/bseww-80110/files/cfc_train.zip?download=1)

[CFC Kenai (source) val images](https://data.caltech.edu/records/bseww-80110/files/cfc_val.zip?download=1)

[CFC Channel (target) train images](https://data.caltech.edu/records/bseww-80110/files/cfc_channel_train.zip?download=1)

[CFC Channel (target) test images](https://data.caltech.edu/records/bseww-80110/files/cfc_channel_test.zip?download=1)

**Labels:**

[CFC Kenai (source) train labels](https://data.caltech.edu/records/bseww-80110/files/cfc_train.json?download=1)

[CFC Kenai (source) val labels](https://data.caltech.edu/records/bseww-80110/files/cfc_val.json?download=1)

[CFC Channel (target) train labels](https://data.caltech.edu/records/bseww-80110/files/cfc_channel_train.json?download=1)

[CFC Channel (target) test labels](https://data.caltech.edu/records/bseww-80110/files/cfc_channel_test.json?download=1)

