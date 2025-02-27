import numpy as np
import json
import pandas as pd
import sys
import argparse
import COCO_util as ccu
import os


ignored_images = {} # "csv_file_name": [{"img_name": "img_name", "explanation": "...", "img_name", ...]

def create_title(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}.json"

def get_filename_for_csv_annotations(date:str, camera, flash=False):
    date_list = date.split("-")
    date_list.reverse()
    src_date = "".join(date_list)
    flash = "on" if flash else "off"
    return f"{src_date} {camera} Flash {flash}_csv.csv"

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
    annotation["id"] = row.fileid + "_" + str(row.region_id) 
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
    info_json = load_json(ccu.INFO_FILE_PATH)
    specs = info_json["folders"][img_folder_name]["specs"]
    field = specs["field"]
    crop = info_json["crops"][specs["crop"]]["name"]
    camera = specs["camera"]
    date_first = specs["dates"][0]
    date_last = specs["dates"][-1]

    info = {}
    description = f"{field} {crop} field, camera {camera} - recorded from {date_first} to {date_last} (specific date in filename)."
    info["description"] = description
    info["date_created"] = date_first
    return info


def convert(file_prefix, img_folder_name, csv_file_path, coco_file_destination):
    data = pd.read_csv(csv_file_path, on_bad_lines='skip')


    data['fileid'] = file_prefix + data['filename'].astype(str) # create fileid column
                                                                # id is the file prefix prepended to the original filename (this corresponds to the correct filename in OUR system)
    data['filename'] = img_folder_name + "/" + data['fileid'].astype(str) # update the filename to have the entire path (and of course the correct filename, which we just saved in the fileid-column)

    # Create categories entries 
    categories = load_json(categories_file)["categories"]
    
    # Create images entries, one for each image
    images = []
    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))
    
    # Create annotations entries
    annotations = []
    name_mappings = load_json(categories_file)["name_mappings"]
    category_name_to_id = {cat["name"]: cat["id"] for cat in categories}
    for row in data.itertuples():
        if row.region_count <= 0:
            continue
        # if there is a detection, append the annotation
        category_name = ccu.extract_category_name_from_region_attributes(row.region_attributes)

        no_insect_label_but_was_annotated = not bool(category_name)
        if no_insect_label_but_was_annotated: 
            csv_name = csv_file_path.split("/")[-1]
            og_filename = row.fileid.split("_")[-1]
            if img_folder_name not in ignored_images.keys():
                ignored_images[img_folder_name] = [] 
            ignored_images[img_folder_name].append(ccu.ignored_img(filename=row.fileid, explanation="Incomplete annotation: No insect class in annotation (region_attributes).", og_csv_name=csv_name, og_filename=og_filename))
            continue

        category_name = ccu.normalise_category_name(category_name)
        if category_name in name_mappings.keys():
            category_name = name_mappings[category_name] # make correction, if needed
        category_id = category_name_to_id[category_name]
        ann = annotation(row,category_id)
        annotations.append(ann)
    
    # Create final coco-json file 
    data_coco = {}
    data_coco["info"] = gen_info(img_folder_name)
    data_coco["license"] = None
    data_coco["images"] = images
    data_coco["annotations"] = annotations
    data_coco["categories"] = categories
    with open(coco_file_destination, "w") as f:
        json.dump(data_coco, f, indent=4)

def record_ignored_images(ignored_images, dest_dir):
    already_ignored = load_json(ccu.IGNORED_IMAGES_PATH)
    if not bool(already_ignored): already_ignored = {}
    with open(ccu.IGNORED_IMAGES_PATH, 'w') as f:
        for key in ignored_images.keys():
            if key in already_ignored:
                already_ignored[key].append(ignored_images[key])
            else:
                already_ignored[key] = ignored_images[key]
        json.dump(already_ignored, f, indent=4)

    
categories_file = "data-annotations/pitfall-cameras/info/categories.json"

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

    src_csv = f"data-annotations/pitfall-cameras/originals/{get_filename_for_csv_annotations(args.date, args.camera, flash=args.f)}"
    dest = f"data-annotations/pitfall-cameras/originals-converted/{create_title(args.field, args.crop, args.camera, args.date, flash=args.f)}" # sys.argv[0]
    
    file_prefix = ccu.get_file_prefix_from_specs(args.field, args.crop, args.camera, args.date, flash=args.f)
    img_folder_name = ccu.get_img_folder_name_from_specs(args.field, args.crop, args.camera)
    convert(file_prefix, img_folder_name, src_csv, dest)
    print(f"Ignored {sum([len(ignored_images[key]) for key in ignored_images.keys()])} images due to errors - see the file for more details.")
    record_ignored_images(ignored_images=ignored_images, dest_dir="data-annotations/pitfall-cameras/info/")
    



if __name__ == "__main__":
    main()

    # Example usage:
    # python scripts/COCO_convert_vgg_csv_to_coco_annotations.py GH OSR HF2G -f 20-06-02
    
