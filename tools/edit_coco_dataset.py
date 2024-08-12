import json
import os

from pycocotools.coco import COCO


def trim_coco_dataset(dataset_file, categories_to_keep=[1]):
    # Data loading code
    print("Loading data")

    coco_dataset = COCO(dataset_file).dataset
    print("Creating data loaders")

    coco_dataset['annotations'] = [ann for ann in coco_dataset['annotations'] if
                                   ann['category_id'] in categories_to_keep]
    coco_dataset['categories'] = [cat for cat in coco_dataset['categories'] if cat['id'] in categories_to_keep]

    # Save dataset
    dataset_dir = os.path.split(dataset_file)[0]
    trim_dataset_file = os.path.join(dataset_dir,
                                     f"instances_trim.json")

    with open(trim_dataset_file, "w") as fp:
        json.dump(coco_dataset, fp)


if __name__ == "__main__":
    annotation_path = "datasets/S-UODAC2020/COCO_Annotations/instances_target.json"
    trim_coco_dataset(annotation_path)
