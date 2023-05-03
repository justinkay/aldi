from detectron2.data.datasets import register_coco_instances

# Cityscapes -> Foggy Cityscapes
register_coco_instances("cityscapes_train", {},         "datasets/cityscapes/annotations/cityscapes_train_instances.json",                  "cityscapes/")
register_coco_instances("cityscapes_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances.json",                    "cityscapes/")
register_coco_instances("cityscapes_foggy_train", {},   "datasets/cityscapes_foggy/annotations/cityscapes_train_instances_foggyALL.json",   "cityscapes_foggy/")
register_coco_instances("cityscapes_foggy_val", {},     "datasets/cityscapes_foggy/annotations/cityscapes_val_instances_foggyALL.json",     "cityscapes_foggy/")