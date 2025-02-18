import numpy as np
import json
import pandas as pd
import sys

def load_json(file):
    o = {}
    with open(file, 'r') as f:
        o = json.load(f)
    return o

def image(row):
    image = {}
    image["id"] = row.fileid
    image["height"] = 3420 # FIXME currently hardcoded - update after cropping?
    image["width"] = 6080 # FIXME
    image["file_name"] = row.filename
    return image

def annotation(row, category_id):
    annotation = {}
    annotation["id"] = row.fileid + str(row.region_id)
    annotation["image_id"] = row.fileid
    annotation["category_id"] = category_id
    annotation["segmentation"] = [] # NOTE blank? We don't have this. Wondering if this needs to be disabled if we use flat-bug data 
    shape = json.loads(row.region_shape_attributes)
    annotation["area"] = shape["width"]*shape["height"]
    annotation["bbox"] = [shape["x"], shape["y"], shape["width"], shape["height"]]
    annotation["iscrowd"] = 0
    return annotation

def get_category_id_from_name(categories, name):
    for c in categories:
        if c["name"] == name:
            return c["id"]
    KeyError("Category didn't exist")

def convert(identifier, info_file, categories_file, csv_file, coco_file_destination):
    data = pd.read_csv(csv_file, on_bad_lines='skip')
    data['fileid'] = pd.Categorical(data['filename'], ordered=True).codes # create a unique file id for each unique filename-row
    data['fileid'] = identifier + data['fileid'].astype(str)
    
    info_json = load_json(info_file)[identifier]["info"]

    # Create images entries, one for each image
    images = []
    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))
    
    # Create categories entries 
    categories = load_json(categories_file)["categories"]

    # Create annotations entries
    annotations = []
    for row in data.itertuples():
        if row.region_count == 0:
            continue
        # if there is a detection, append the annotation
        category_name = list((json.loads(row.region_attributes))["Insect"].keys())[0] 
        category_id = get_category_id_from_name(categories, category_name)
        annotations.append(annotation(row,category_id))
    
    # Create final coco-json file 
    data_coco = {}
    data_coco["info"] = info_json
    data_coco["license"] = None
    data_coco["categories"] = categories
    data_coco["images"] = images
    data_coco["annotations"] = annotations
    with open(coco_file_destination, "w") as f:
        json.dump(data_coco, f, indent=4)

if __name__ == "__main__":
    title = "200601-HF2G-f" # sys.argv[0]
    info_file = "../ERDA/bugmaster/datasets/pitfall-cameras/info.json"
    categories_file = "../ERDA/bugmaster/datasets/pitfall-cameras/categories.json"
    csv_file = "../ERDA/bugmaster/datasets/pitfall-cameras/annotations/010620 HF2G Flash on_csv.csv"
    save_coco_path = "../ERDA/bugmaster/datasets/pitfall-cameras/annotations/200601-HF2G-f.json"
    convert(title, info_file, categories_file, csv_file, save_coco_path)
