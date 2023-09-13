from detectron2.data.datasets import register_coco_instances

# Cityscapes 
register_coco_instances("cityscapes_train", {},         "datasets/cityscapes/annotations/cityscapes_train_instances.json",                  "datasets/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances.json",                    "datasets/cityscapes/leftImg8bit/val/")

# Foggy Cityscapes
register_coco_instances("cityscapes_foggy_train", {},   "datasets/cityscapes_foggy/annotations/cityscapes_train_instances_foggyALL.json",   "datasets/cityscapes_foggy/leftImg8bit/train/")
register_coco_instances("cityscapes_foggy_val", {},     "datasets/cityscapes_foggy/annotations/cityscapes_val_instances_foggyALL.json",     "datasets/cityscapes_foggy/leftImg8bit/val/")
# for evaluating COCO-pretrained models: category IDs are remapped to match
register_coco_instances("cityscapes_foggy_val_coco_ids", {},     "datasets/cityscapes_foggy/annotations/cityscapes_val_instances_foggyALL_coco.json",     "datasets/cityscapes_foggy/leftImg8bit/val/")

# Sim10k
# TODO

# CFC
register_coco_instances("cfc_train", {},         "datasets/cfc/coco_labels/cfc_train.json",                  "datasets/cfc/images/cfc_train/")
register_coco_instances("cfc_val",   {},         "datasets/cfc/coco_labels/cfc_val.json",                    "datasets/cfc/images/cfc_val/")
# register_coco_instances("cfc_channel_train", {},         "datasets/cfc/coco_labels/cfc_channel_train.json",                  "datasets/cfc/images/cfc_channel_train/a")
register_coco_instances("cfc_channel_test",   {},         "datasets/cfc/coco_labels/cfc_channel_test.json",                    "datasets/cfc/images/cfc_channel_test/")