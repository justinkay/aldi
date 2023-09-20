# doad-strong-baseline
Strong unified baselines for domain adaptive object detection.


### Detectron v0.7 Setup
For now use the unofficial 0.7 release.
```
cd libs
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git fetch origin pull/4868/head
git checkout -b pullrequest FETCH_HEAD
pip install -e .
```
Once this is an official release, replace with `pip install`.

### Downloading data

TODO


### Model zoo

TODO

You can download all models using `models/download_models.sh`

You will need the Github CLI installed to do so. You can install with conda: `conda install gh --channel conda-forge`
