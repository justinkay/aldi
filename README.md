# Align and Distill (ALDI): Unifying and Improving Domain Adaptive Object Detection

This is the official codebase for [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029).

\[[Project Page](https://aldi-daod.github.io/)\] \[[Arxiv](https://arxiv.org/abs/2403.12029)\] \[[PDF](https://arxiv.org/pdf/2403.12029.pdf)\] 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/align-and-distill-unifying-and-improving/unsupervised-domain-adaptation-on-cityscapes-1)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-cityscapes-1?p=align-and-distill-unifying-and-improving) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/align-and-distill-unifying-and-improving/unsupervised-domain-adaptation-on-sim10k-to-3)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-sim10k-to-3?p=align-and-distill-unifying-and-improving) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/align-and-distill-unifying-and-improving/unsupervised-domain-adaptation-on-cfc-daod)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-cfc-daod?p=align-and-distill-unifying-and-improving)

![](docs/aldi_banner_4.png)

Align and Distill (ALDI) is a state-of-the-art framework for domain adaptive object detection (DAOD), built on top of the [Detectron2](https://github.com/facebookresearch/detectron2/) object detection library. ALDI is:

**Accurate:** ALDI's default settings achieve <ins>state-of-the-art results</ins> on DAOD benchmarks including Cityscapes &rarr; Foggy Cityscapes, Sim10k &rarr; Cityscapes, and CFC Kenai &rarr; Channel.

**Fast to train:** Thanks to efficient dataloading and optimized burn-in settings, ALDI <ins>trains upwards of 20x faster</ins> than other DAOD methods.
 
**Easy to use:** Training DAOD models on your own data requires just a couple steps; see [setting up your own data](docs/CUSTOM_DATA.md) and [training ALDI](docs/TRAINING.md).

**Extensible:** The framework is lightweight, reusing default components from Detectron2 wherever possible. See [code documentation](docs/CODEBASE.md) for an overview of the code structure and design decisions.

## Installation

See [installation instructions](docs/INSTALL.md).

## Model zoo

We provide burn-in checkpoints and final models for DAOD benchmarks (Cityscapes &rarr; Foggy Cityscapes, Sim10k &rarr; Cityscapes, and CFC Kenai &rarr; Channel) in [the model zoo](docs/MODELS.md).

For compatibility with existing config files, download models to the `models/` directory in this repo.

You can download the required model weights for any config file we provide using `python tools/download_model_for_config.py --config-file path/to/config.yaml`

## Benchmark dataset setup

Follow [these instructions](docs/DATASETS.md) to set up data and reproduce benchmark results on Cityscapes &rarr; Foggy Cityscapes, Sim10k &rarr; Cityscapes, and CFC Kenai &rarr; Channel.

## <a id="own-data"></a>Using your own data 

To use ALDI on your own data, see [instructions for custom datasets](docs/CUSTOM_DATA.md).

## Training ALDI

See [training instructions](docs/TRAINING.md).

## Extras

The [main](/justinkay/aldi/tree/main) branch contains all you need to run ALDI, and is a good starting point for most.

Additional code and configuration files to reproduce all experiments in [our paper](https://arxiv.org/abs/2403.12029) can be found on the [extras](/justinkay/aldi/tree/extras) branch.

## Reference

#### [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029)

[Justin Kay](https://justinkay.github.io), [Timm Haucke](https://timm.haucke.xyz/), [Suzanne Stathatos](https://suzanne-stathatos.github.io/), [Siqi Deng](https://www.amazon.science/author/siqi-deng), [Erik Young](https://home.tu.org/users/erikyoung), [Pietro Perona](https://scholar.google.com/citations?user=j29kMCwAAAAJ), [Sara Beery](https://beerys.github.io/), and [Grant Van Horn](https://gvanhorn38.github.io/).

Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real-world data, and (4) A new method, ALDI++, that achieves state-of-the-art results by a large margin. ALDI++ outperforms the previous state-of-the-art by +3.5 AP50 on Cityscapes → Foggy Cityscapes, +5.7 AP50 on Sim10k → Cityscapes (where ours is the only method to outperform a fair baseline), and +2.0 AP50 on CFC Kenai → Channel. Our framework, dataset, and state-of-the-art method offer a critical reset for DAOD and provide a strong foundation for future research. 

If you find our work useful in your research please consider citing our paper:

```
@misc{kay2024align,
      title={Align and Distill: Unifying and Improving Domain Adaptive Object Detection}, 
      author={Justin Kay and Timm Haucke and Suzanne Stathatos and Siqi Deng and Erik Young and Pietro Perona and Sara Beery and Grant Van Horn},
      year={2024},
      eprint={2403.12029},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
