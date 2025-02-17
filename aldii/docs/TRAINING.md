# Training ALDI

Training ALDI involves two steps:
1. Baseline training / burn-in
2. Domain adaptive training

This assumes you have:
- A working installation; see the [installation guide](INSTALL.md) if you have not done this yet.
- Datasets set up and registered with Detectron2; see [setting up benchmark datasets](DATASETS.md) or [setting up your own data](CUSTOM_DATA.md) if you have not done this yet.

## 1. Baseline training / burn-in

The first step is to train a standard object detector on your source dataset (AKA "source-only training"). This has two purposes:

- **Baseline.** For evaluation, this model tells you the best you can do *before domain adaptation*, using only source-domain data.
- **Burn-in.** Source-only training is also an effective method for network initialization, AKA "burn-in", before performing domain adaptation. See [our paper](https://arxiv.org/abs/2403.12029) for more details.

### Set up configuration file for baseline training / burn-in

You will need a custom Detectron2 configuration file for every training run. We recommend you start with [configs/cityscapes/Base-RCNN-FPN-Cityscapes_strongaug_ema.yaml](../configs/cityscapes/Base-RCNN-FPN-Cityscapes_strongaug_ema.yaml) as a guide for performing baseline training / burn-in. Note that compared to standard Detectron2 training configurations, this config file uses:

- Stronger augmentations
- Exponential moving average (EMA)

We have found these settings to both increase overall performance (in terms of mAP) as well as reduce training time during the domain adaptation step. See our [paper](https://arxiv.org/abs/2403.12029) for more details.

**Modifications you must make** To use with your dataset, make a copy of [configs/cityscapes/Base-RCNN-FPN-Cityscapes_strongaug_ema.yaml](../configs/cityscapes/Base-RCNN-FPN-Cityscapes_strongaug_ema.yaml) and change the following values:

```
DATASETS:
  TRAIN: ("your_training_dataset_name",) # needs to be a tuple, and can contain multiple datasets if you want
  TEST: ("your_test_dataset_name",)  # needs to be a tuple, and can contain multiple datasets if you want
```

### Run training

To train, run:

```
python tools/train_net.py --config path/to/your/burn_in_config.yaml
```

Other training options include `--num-gpus` to run distributed training; see [tools/train_net.py](../tools/train_net.py) and the [Detectron2 training docs](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html#training-evaluation-in-command-line) for more details.

### Where is your final model? 

A `BestCheckpointer` will be used by default to save the best model checkpoint based on validation performance on each `DATASETS.TEST`; this model will be saved according to the `OUTPUT_DIR` in your config file, and will end in `_best.pth`.

## 2. Domain adaptive training

Now you're ready to use ALDI for domain adaptation. Again this involves creating a configuration file and running `tools/train_net.py`.

### Configuration file setup 

See [configs/cityscapes/ALDI-Best-Cityscapes.yaml](../configs/cityscapes/ALDI-Best-Cityscapes.yaml) for an example. You must change two things: the **dataset** and the **path to initial model weights.**

**Dataset** To use with a different dataset, simply change `DATASETS.UNLABELED` to point to your unsupervised target-domain data (like `DATASETS.TRAIN` AND `DATASETS.TEST`, this must also be registered with Detectron2 as described in the [dataset setup guide](CUSTOM_DATA.md)):

```
DATASETS:
  UNLABELED: ("your_unlabeled_dataset_name",)
```

**Model weights** To use your burned-in model from step 1, change:

```
MODEL:
  WEIGHTS: "path/to/your/burned_in_model.pth"
```

### Training

To train, run:

```
python tools/train_net.py --config path/to/your/aldi_config.yaml
```

Domain adaptive training is also compatible with standard Detectron2 training options such as `--num-gpus`.

### Evaluation

After training, to run evaluation with your model:

```
python tools/train_net.py --eval-only --config-file path/to/your/aldi_config.yaml MODEL.WEIGHTS path/to/your/model_best.pth
```

### Feature visualization

To visualize a trained model's feature space, run:

```
python tools/visualize_featurespace.py --config-file path/to/your/aldi_config.yaml MODEL.WEIGHTS path/to/your/model.pth
```

The plots will be written into that config's `OUTPUT_DIR`.