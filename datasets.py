from detectron2.data.datasets import register_coco_instances

# Cityscapes -> Foggy Cityscapes
register_coco_instances("cityscapes_train", {}, "datasets/cityscapes/annotations/instancesonly_filtered_gtFine_train.json", "datasets/cityscapes/")
register_coco_instances("cityscapes_val", {}, "datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json", "datasets/cityscapes/")
register_coco_instances("cityscapes_foggy_train", {}, "datasets/cityscapes_foggy/annotations/instancesonly_filtered_gtFine_train_foggyALL.json", "datasets/cityscapes_foggy/")
register_coco_instances("cityscapes_foggy_val", {}, "datasets/cityscapes_foggy/annotations/instancesonly_filtered_gtFine_val_foggyALL.json", "datasets/cityscapes_foggy/")