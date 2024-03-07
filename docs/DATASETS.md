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

[Cityscapes train labels](bit.ly/49SFIzU)

[Cityscapes val labels](bit.ly/3V757Bx)

[Foggy Cityscapes train labels](bit.ly/3V5qD9S)

[Foggy Cityscapes val labels](bit.ly/49WOCfE)

## Sim10k &rarr; Cityscapes

**Sim10k images:** Download the Sim10k images [here](https://deepblue.lib.umich.edu/data/downloads/ks65hc58r), and place them in `aldi/datasets/sim10k/images/` as shown above.

**Cityscapes images:** Follow instructions for setting up Cityscapes above. Note you will only need the `leftImg8bit` images, not the foggy ones

**Labels:** For DAOD, Sim10k &rarr; Cityscapes is typically a single-class challenge consisting only of the "car" class. We provide labels postprocessed for this task here:

[Sim10k cars train labels](bit.ly/4a15iCL)

[Cityscapes cars train labels](bit.ly/3wKORfH)

[Cityscapes cars val labels](bit.ly/3T4v9Tr)

## CFC Kenai &rarr; Channel

**Images:**

[CFC Kenai (source) train images](bit.ly/3wKOCBj)

[CFC Kenai (source) val images](bit.ly/3v7jz1R)

[CFC Channel (target) train images](bit.ly/3TlXqGt)

[CFC Channel (target) test images](bit.ly/3uQr8u1)

**Labels:**

[CFC Kenai (source) train labels](bit.ly/48EMZ5b)

[CFC Kenai (source) val labels](bit.ly/3V2v8Cc)

[CFC Channel (target) train labels](bit.ly/4c8jkVd)

[CFC Channel (target) test labels](bit.ly/49GF74E)

