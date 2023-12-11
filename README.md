# ALDI
Align and distill (ALDI): A unified framework for domain adaptive object detection.

## Installation

#### 1. Environment setup (recommended)

We recommend developing in a new Conda environment (e.g. using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)):

```
conda create -n aldi python=3.9
conda activate aldi
```

#### 2. Prerequisites (required)

i. Install the appropriate versions of PyTorch and torchvision for your machine. Follow the instructions [here](https://pytorch.org/get-started/locally/). *You must do this before installing ALDI!*

#### 3. Installing ALDI and its dependencies

i. Clone this repository with submodules:

```
git clone git@github.com:justinkay/aldi.git
cd aldi
```

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
