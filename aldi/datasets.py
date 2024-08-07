from detectron2.data.datasets import register_coco_instances

cyclegan_results_name_cityscapes2foggy = "cityscapes2foggy_002"
cyclegan_results_epoch_cityscapes2foggy = 50

# Cityscapes 
register_coco_instances("cityscapes_train", {"image_dir_prefix": "datasets/cityscapes/leftImg8bit/train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/{cyclegan_results_name_cityscapes2foggy}/test_{cyclegan_results_epoch_cityscapes2foggy}/images/fake_A"},         "datasets/cityscapes/annotations/cityscapes_train_instances.json",                  "datasets/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances.json",                    "datasets/cityscapes/leftImg8bit/val/")

# Foggy Cityscapes
register_coco_instances("cityscapes_foggy_train", {"image_dir_prefix": "datasets/cityscapes/leftImg8bit_foggy/train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/{cyclegan_results_name_cityscapes2foggy}/test_{cyclegan_results_epoch_cityscapes2foggy}/images/fake_B"},   "datasets/cityscapes/annotations/cityscapes_train_instances_foggyALL.json",   "datasets/cityscapes/leftImg8bit_foggy/train/")
register_coco_instances("cityscapes_foggy_val", {},     "datasets/cityscapes/annotations/cityscapes_val_instances_foggyALL.json",     "datasets/cityscapes/leftImg8bit_foggy/val/")
# for evaluating COCO-pretrained models: category IDs are remapped to match
register_coco_instances("cityscapes_foggy_val_coco_ids", {},     "datasets/cityscapes/annotations/cityscapes_val_instances_foggyALL_coco.json",     "datasets/cityscapes/leftImg8bit_foggy/val/")

# Sim10k
register_coco_instances("sim10k_cars_train", {"image_dir_prefix": "datasets/sim10k/images", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/sim10k2cityscapes/test_20/images/fake_A"},             "datasets/sim10k/coco_car_annotations.json",                  "datasets/sim10k/images/")
register_coco_instances("cityscapes_cars_train", {"image_dir_prefix": "datasets/cityscapes/leftImg8bit/train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/sim10k2cityscapes/test_20/images/fake_B"},         "datasets/cityscapes/annotations/cityscapes_train_instances_cars.json",                  "datasets/cityscapes/leftImg8bit/train/")
register_coco_instances("cityscapes_cars_val",   {},         "datasets/cityscapes/annotations/cityscapes_val_instances_cars.json",                    "datasets/cityscapes/leftImg8bit/val/")

# CFC
register_coco_instances("cfc_train", {"image_dir_prefix": "datasets/cfc/images/cfc_train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/cfc_002/test_1/images/fake_A"},         "datasets/cfc/coco_labels/cfc_train.json",                  "datasets/cfc/images/cfc_train/")
register_coco_instances("cfc_val",   {},         "datasets/cfc/coco_labels/cfc_val.json",                    "datasets/cfc/images/cfc_val/")
register_coco_instances("cfc_channel_train", {"image_dir_prefix": "datasets/cfc/images/cfc_channel_train", "translated_image_dir": f"datasets/daod-strong-baseline-cyclegan-results/cfc_002/test_1/images/fake_B"},         "datasets/cfc/coco_labels/cfc_channel_train.json",                  "datasets/cfc/images/cfc_channel_train/")
register_coco_instances("cfc_channel_test",   {},         "datasets/cfc/coco_labels/cfc_channel_test.json",                    "datasets/cfc/images/cfc_channel_test/")

# Urchin synthetic nudi_urchin3
register_coco_instances("nudi_urchin3_train", {}, "datasets/collated_outputs/nudi_urchin3/annotations/instances_train2023.json", "datasets/collated_outputs/nudi_urchin3/train2023")
register_coco_instances("nudi_urchin3_test", {}, "datasets/collated_outputs/nudi_urchin3/annotations/instances_test2023.json", "datasets/collated_outputs/nudi_urchin3/test2023")

# Urchin synthetic urchininf_v0
register_coco_instances("urchininf_v0_train", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_train2023.json", "datasets/collated_outputs/urchininf_v0/train2023")
register_coco_instances("urchininf_v0_test", {}, "datasets/collated_outputs/urchininf_v0/annotations/instances_test2023.json", "datasets/collated_outputs/urchininf_v0/test2023")


# UDD
register_coco_instances("UDD_test", {}, "datasets/UDD/annotations/instances_test2023_remap.json", "datasets/UDD/test2023")
register_coco_instances("UDD_train", {}, "datasets/UDD/annotations/instances_train2023_remap.json", "datasets/UDD/train2023")

# Squidle
register_coco_instances("squidle_urchin_2011_test", {}, "datasets/squidle_coco/squidle_urchin_2011/annotations/instances_test2023.json", "datasets/squidle_coco/squidle_urchin_2011/test2023")
register_coco_instances("squidle_urchin_2009_train", {}, "datasets/squidle_coco/squidle_urchin_2009/annotations/instances_train2023.json", "datasets/squidle_coco/squidle_urchin_2009/train2023")

