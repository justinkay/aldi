import os 
import shutil

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

root = os.getenv("DETECTRON2_DATASETS", "../datasets")

_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
}

for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
    print("registering", key)
    meta = _get_builtin_metadata("cityscapes")
    image_dir = os.path.join(root, image_dir)
    gt_dir = os.path.join(root, gt_dir)
    inst_key = key
    DatasetCatalog.register(
        inst_key,
        lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
            x, y, from_json=False, to_polygons=False
        ),
    )
    MetadataCatalog.get(inst_key).set(
        image_dir=image_dir, gt_dir=gt_dir, evaluator_type="coco", **meta
    )

    # convert to COCO format
    coco_path = os.path.join(root, "cityscapes/annotations/")
    os.makedirs(coco_path, exist_ok=True)
    output_filename = os.path.join(coco_path, key+"_instances.json")
    print("converting", key, "to", output_filename)
    convert_to_coco_json(key, output_filename, allow_cached=True)

    # add to foggy cityscapes as well
    coco_foggy_path = os.path.join(root, "cityscapes_foggy/annotations/")
    os.makedirs(coco_foggy_path, exist_ok=True)
    print("copying", output_filename, "to", coco_foggy_path)
    shutil.copy(output_filename, coco_foggy_path)