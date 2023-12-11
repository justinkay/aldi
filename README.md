# ALDI
Align and distill (ALDI): A unified framework for domain adaptive object detection.

## Installation

#### 1. Prerequisites (start here!)

i. Install the appropriate versions of PyTorch and torchvision for your machine. Follow the instructions [here](https://pytorch.org/get-started/locally/). *You must do this before installing ALDI!*

#### 2. Installing ALDI and its dependencies

We recommend developing in a new Conda environment (e.g. using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)):

```
conda create -n aldi python=3.9
conda activate aldi
```

i. Clone this repository with submodules:

```
git clone --recurse-submodules git@github.com:justinkay/aldi.git
cd aldi
```

(If you forgot to `--recurse-submodules`, don't worry, you can do this later)

ii. Install Detectron2:

```
git submodule update --init --recursive
cd libs/detectron2
pip install -e .
```

iii. Install ALDI:

```
# if working directory is libs/detectron2
cd ../..

pip install -e .
```

### Downloading data

TODO


### Model zoo

TODO

You can download all models using `models/download_models.sh`

You will need the Github CLI installed to do so. You can install with conda: `conda install gh --channel conda-forge`
