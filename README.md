# doad-strong-baseline
Strong unified baselines for domain adaptive object detection.

### Dependencies
Install dependencies using anaconda:
```
conda env create -f environment.yml
conda activate daod-strong-baseline
```

### Detectron v0.7 Setup
For now use the unofficial 0.7 release.
```
git submodule update --init --recursive
cd libs/detectron2
pip install -e .
```
Once this is an official release, replace with `pip install`.

### Downloading data

TODO


### Model zoo

TODO

You can download all models using `models/download_models.sh`

You will need the Github CLI installed to do so. You can install with conda: `conda install gh --channel conda-forge`
