# Installing ALDI

#### 1. Environment setup (Recommended)

We recommend developing in a new Conda environment (e.g. using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)):

```
conda create -n aldi python=3.9
conda activate aldi
```

#### 2. Prerequisites (Required)

1. Install the appropriate versions of PyTorch and torchvision for your machine. Follow the instructions [here](https://pytorch.org/get-started/locally/). **You must do this before installing ALDI!**

#### 3. Installing ALDI and its dependencies

1. Clone this repository with submodules:

```
git clone git@github.com:justinkay/aldi.git
cd aldi
```

2. Install Detectron2:

```
git submodule update --init --recursive
cd libs/detectron2
pip install -e .
```

3. Install ALDI:

```
# if working directory is libs/detectron2
cd ../..

pip install -e .
```
