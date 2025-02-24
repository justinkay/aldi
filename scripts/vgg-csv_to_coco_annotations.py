import numpy as np
import json
import pandas as pd
import sys
import argparse

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
    image["file_name"] = + row.filename
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

def gen_info(img_folder_name):
    info_json = load_json(info_file)
    specs = load_json(info_file)[img_folder_name]["specs"]
    info = {}
    description = f"{info_json["fields"][specs["field"]]} {info_json["crops"][specs["crop"]]}, camera {specs["camera"]}."
    info["description"] = description
    info["date_created"] = specs["dates"][0]
    return info


def convert(file_prefix, img_folder_name, csv_file, coco_file_destination):
    data = pd.read_csv(csv_file, on_bad_lines='skip')
    #data['fileid'] = pd.Categorical(data['filename'], ordered=True).codes # create a unique file id for each unique filename-row
    data['fileid'] = file_prefix + data['filename'].astype(str)
    data['filename'] = f"{img_folder_name}/{file_prefix}{data['filename'].astype(str)}"

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
    data_coco["info"] = gen_info(img_folder_name)
    data_coco["license"] = None
    data_coco["categories"] = categories
    data_coco["images"] = images
    data_coco["annotations"] = annotations
    with open(coco_file_destination, "w") as f:
        json.dump(data_coco, f, indent=4)

def create_title(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}.json"

def get_src_csv_name(date:str, camera, flash=False):
    date_list = date.split("-").reverse()
    src_date = "".join(date_list)
    flashstr = " Flash on" if flash else ""
    return f"{src_date} {camera}{flashstr}_csv.csv"

def get_prefix(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}_"

def get_img_folder_name(field, crop, camera):
    return f"{field}-{crop}-{camera}"

info_file = "../ERDA/bugmaster/datasets/pitfall-cameras/info.json"
categories_file = "../ERDA/bugmaster/datasets/pitfall-cameras/categories.json"

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Convert VGG CSV annotations to COCO JSON for given location and date.")
    parser.add_argument("field", help="Field, e.g. GH for Geescroft and Highfield.")
    parser.add_argument("crop", help="OSR or WWH.")
    parser.add_argument("camera", help="Camera name.")
    parser.add_argument('-f', action='store_true', help="Add flag if flash on")
    parser.add_argument("date", help="date of capture.")
    
    # Parse the arguments
    args = parser.parse_args()

    src_csv = f"../ERDA/bugmaster/datasets/pitfall-cameras/annotations/{get_src_csv_name(args.date, args.camera, flash=args.f)}"
    dest = f"../ERDA/bugmaster/datasets/pitfall-cameras/annotations-converted/{create_title(args.field, args.crop, args.camera, args.date, flash=args.f)}" # sys.argv[0]
    
    file_prefix = get_prefix(args.field, args.crop, args.camera, args.date, flash=args.f)
    img_folder_name = get_img_folder_name(args.field, args.crop, args.camera)
    convert(file_prefix, img_folder_name, src_csv, dest)

if __name__ == "__main__":
    main()
    
