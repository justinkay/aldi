import os 
import shutil
import json
import copy

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
    # (conversion will be finished below)
    coco_foggy_path = os.path.join(root, "cityscapes_foggy/annotations/")
    os.makedirs(coco_foggy_path, exist_ok=True)
    print("copying", output_filename, "to", coco_foggy_path)
    shutil.copy(output_filename, coco_foggy_path)

# add all fog levels
for s in ('train', 'val'):
    js_loc = f'{root}/cityscapes_foggy/annotations/cityscapes_{s}_instances.json'
    with open(js_loc) as f:
        js = json.load(f)
    print(js.keys())

    # img_id_start = max([i['id'] for i in js['images']]) + 1
    anno_id_start = max([i['id'] for i in js['annotations']]) + 1

    new_coco = copy.deepcopy(js)

    new_images = []
    for img in js['images']:
        im0 = copy.deepcopy(img)
        # im0['id'] = img['id'] 
        im0['file_name'] = img['file_name'].replace("/cityscapes/", "/cityscapes_foggy/").replace('.png', '_foggy_beta_0.01.png')
        im0['id'] = os.path.basename(im0['file_name'])

        im1 = copy.deepcopy(img)
        # im1['id'] = img['id'] + img_id_start
        im1['file_name'] = img['file_name'].replace("/cityscapes/", "/cityscapes_foggy/").replace('.png', '_foggy_beta_0.02.png')
        im1['id'] = os.path.basename(im1['file_name'])

        im2 = copy.deepcopy(img)
        # im2['id'] = img['id'] + 2*img_id_start
        im2['file_name'] = img['file_name'].replace("/cityscapes/", "/cityscapes_foggy/").replace('.png', '_foggy_beta_0.005.png')
        im2['id'] = os.path.basename(im2['file_name'])

        new_images.append(im0)
        new_images.append(im1)
        new_images.append(im2)

    new_annotations = []
    for ann in js['annotations']:
        ann0 = copy.deepcopy(ann)
        ann0['id'] = ann['id']
        ann0['image_id'] = ann['image_id'].replace('.png', '_foggy_beta_0.01.png')

        ann1 = copy.deepcopy(ann)
        ann1['id'] = ann['id'] + anno_id_start
        ann1['image_id'] = ann['image_id'].replace('.png', '_foggy_beta_0.02.png')

        ann2 = copy.deepcopy(ann)
        ann2['id'] = ann['id'] + 2*anno_id_start
        ann2['image_id'] = ann['image_id'].replace('.png', '_foggy_beta_0.005.png')

        new_annotations.append(ann0)
        new_annotations.append(ann1)
        new_annotations.append(ann2)

    new_coco['images'] = new_images
    new_coco['annotations'] = new_annotations

    print(len(js['images']), len(js['annotations']))
    print(len(new_coco['images']), len(new_coco['annotations']))

    json.dump(new_coco, open(f'{root}/cityscapes_foggy/annotations/cityscapes_{s}_instances_foggyALL.json', 'w'))