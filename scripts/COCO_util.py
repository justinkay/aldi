import re
import json
from PIL import Image
import os

INFO_FILE_PATH =  "data-annotations/pitfall-cameras/info/info.json"
IGNORED_IMAGES_PATH =  "data-annotations/pitfall-cameras/info/ignored_images.json"
IMAGES_FOLDER = "../ERDA/bugmaster/datasets/pitfall-cameras/images/"

LOCATIONS = ['GH_OSR_HF2G', 'GH_OSR_LF1F', 'GH_OSR_LF2E', 'GH_OSR_NARS26', 'GH_OSR_NARS30', 'LG_OSR_HF2F', 'LG_OSR_LF1D', 'LG_OSR_LF1G', 'LG_OSR_LF2F', 'LG_OSR_LS3E', 'LG_WWH_NARS30', 'WW_OSR_HF2F', 'WW_OSR_LF1D', 'WW_OSR_LF1G', 'WW_OSR_LF2F', 'WW_OSR_LS3E']

def create_title(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}.json"

def get_specs_from_info(img_folder_name, info_file_path=INFO_FILE_PATH):
    info = {}
    with open(info_file_path, 'r') as f:
        info = json.load(f)
        specs = info["folders"][img_folder_name]["specs"]
    return specs

def get_img_folder_name_from_specs(field, crop, camera):
    return f"{field}-{crop}-{camera}"

def get_file_prefix_from_specs(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}_"

def normalise_category_name(name:str):
    """ Normalise the string with the category name st. it's lower case and uses space separation"""
    name = name.lower()  # Convert to lowercase
    name = re.sub(r"[._]", " ", name)  # Replace dots and underscores with spaces
    name = re.sub(r"\s+", " ", name).strip()  # Remove extra spaces
    name = reorder_unknown(name)
    return name

def reorder_unknown(name):
    """Moves 'unknown' to be after the first word, if present."""
    words = name.split()
    if "unknown" in words and words[-1] != "unknown":
        words.remove("unknown")
        words.append("unknown")  # Place "unknown" last
    return " ".join(words)

def read_image_size(image_path):
    if os.path.exists(image_path):
        with Image.open(image_path) as image:
            return image.size # width, height
    else:
        print(f"Warning: Image not found - {image_path}")    

def extract_category_name_from_region_attributes(attr):
    try:
        attr_dict = json.loads(attr)
        #keys = list(attr_dict.keys())
        #if ("Insect ID" not in keys) and ("Insect" not in keys) and ("Insects" not in keys): print(attr_dict)
        for key in ["Insect ID"]:
            # retrieve label from this format: {"Insect": "Carabid"}
            if key in attr_dict:
                return attr_dict[key]
        for key in ["Insect", "Insects"]:
            # retrieve label from this format: {"Insect": {"Carabid": True}}
            if key in attr_dict:
                keys = list(attr_dict[key].keys())
                if len(keys)==0:
                    return None # no label
                return keys[0] 
    except json.JSONDecodeError:
        return None # ignore malformed json
    return None 

def ignored_img(filename, explanation, og_filename, og_csv_name):
    return {"filename": filename, "explanation": explanation, "original_filename": og_filename, "orignal_csv_file": og_csv_name}