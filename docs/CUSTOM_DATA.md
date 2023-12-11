# Running on your own data

To run on your own data, you will need:

1. A directory containing your images
2. [COCO](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html)-formatted annotations for your images

You will need a separate COCO file for each dataset. Typically this will include:
- Train: Labeled source-domain images for supervised burn-in and training
- Unlabeled: Target-domain images for DAOD. These can be optionally labeled if you also want to use them to train a supervised "oracle" for comparison. If unlabeled, the `"annotations"` entry of your COCO file should be an empty list.
- Test: A labeled validation set. In most DAOD papers this comes from your target domain, even though this breaks the constraints of UDA.

To use your dataset with $ALDI$, 

1. Make your data (images and annotations) accessible from the `datasets` directory in this repository. You can either place your data here, or add symlinks to it.

2. Register each of your dataset(s) by adding a new line to [dataset.py](). Paths should be relative to the base directory of this repo. Note that the images directories can be the same if you'd like, since images will be chosen according to their paths in the COCO JSON files.

```
register_coco_instances("your_training_dataset_name", {}, "path/to/your_training_coco_labels.json", "path/to/your/train/images/")
register_coco_instances("your_unlabeled_dataset_name", {}, "path/to/your_unlabeled_coco_labels.json", "path/to/your/unlabeled/images/")
register_coco_instances("your_test_dataset_name", {}, "path/to/your_test_coco_labels.json", "path/to/your/test/images/")
```

3. Now your datasets can be referenced by name in config.yaml files, e.g.:

```
DATASETS:
  TRAIN: ("your_training_dataset_name",) # needs to be a tuple, and can contain multiple datasets if you want
  UNLABELED: ("your_unlabeled_dataset_name",) # needs to be a tuple, and can contain multiple datasets if you want
  TEST: ("your_test_dataset_name",)  # needs to be a tuple, and can contain multiple datasets if you want
```
