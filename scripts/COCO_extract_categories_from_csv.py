import numpy as np
import json
import pandas as pd
import sys
import argparse
import os
import COCO_util as ccu
import re

name_mappings = { # corrections for typos and redundancies that weren't caught by stanadardizing the name
    "arachnid": "arachnida",
    "amara sp": "amara",
    "carabid": "carabidae", 
    "carabid unknown": "carabidae",  
    "carabids": "carabidae",  
    "carabid lar": "beetle",  
    "linyphiiidae": "linyphiidae",
    "molllusca": "mollusca",
    "phyllotreta sp": "phyllotreta",
    "psyllidoes chrysocephalus": "psylliodes chrysocephalus",
    "spider": "arachnida",
    "tachyprous hypnorum": "tachyporus hypnorum"
    }

def clean_categories(cats):
    """Normalises all category names and merges according to typos/redundancies (manually defined in the dict above)"""
    cleaned_set = set()

    for category in cats:
        normalized = ccu.normalise_category_name(category) # lower case, space separation, "unknown" comes last
        
        if normalized in name_mappings: # overwrite with the correction if it's there!
            category_name = name_mappings[normalized]
            print("True: "+ category_name)
            print("Wrong: "+ normalized)
        else:
            category_name = normalized  

        cleaned_set.add(category_name) # insert the (corrected) category 

    return cleaned_set

def extract_category_name_from_region_attributes(attr):
    cat = set()
    try:
        attr_dict = json.loads(attr)
        for key in ["Insect", "Insects"]:
            if key in attr_dict:
                return  attr_dict[key].keys() # NOTE - we assume that the supercategory is always insect (or insects, lol)
    except json.JSONDecodeError:
        cat = set() # ignore malformed json
    return cat

def extract_categories_from_vgg_csv(src):
    df = pd.read_csv(src, on_bad_lines='skip')
    unique_categories = set()  # Store unique category names
    for row in df.itertuples():
        if row.region_count <= 0:
            continue
        # if there is a detection, append the annotation
        cat = extract_category_name_from_region_attributes(row.region_attributes)
        unique_categories.update(cat)
    return unique_categories

def extract_categories_from_vgg_csv_dir(src_dir):
    """Runs through a given directory of vgg-csv annotation files and extracts all unique categories (returned as a set)"""
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and os.path.splitext(f)[1].lower().endswith((".csv")))]
    total_files = len(files)
    
    categories = set()
    
    for index, filename in enumerate(files, start=1): 
        # Build the full path to the file
        src_file_path = os.path.join(src_dir, filename)
        cats = extract_categories_from_vgg_csv(src_file_path)
        categories.update(cats)
        # Print progress every 5 files
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files")
    return categories

def create_coco_categories_from_set(cats:set):
    """Create category dictionary with unique, sorted category names"""
    return [
        {"id": idx + 1, "name": cat, "supercategory": "insect"}
        for idx, cat in enumerate(sorted(cats))  # Sorted to ensure consistency
    ]

def save_categories_to_file(cats, mappings, dest_dir, filename):
    path = os.path.join(dest_dir, filename+".json")
    categories = {}
    categories["categories"] = cats
    categories["name_mappings"] = mappings # also save the mappings to use when generating the annotations from the csv-files that have those typos
    with open(path, "w") as f:
        json.dump(categories, f, indent=4)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract categories from all VGG-CSV annotations into a single JSON-object (\"categories\") compatible with the COCO-format.")
    parser.add_argument("src_dir", nargs="?", help="Source directory containing the csv files.", default="../ERDA/bugmaster/datasets/pitfall-cameras/annotations/Annotations and other files/CSV files")
    parser.add_argument("dest_dir", nargs="?", help="Directory at which to save the generated categories.json.", default="../ERDA/bugmaster/datasets/pitfall-cameras/annotations/")
    parser.add_argument("filename", nargs="?", default="categories", help="Optional name of the generated file (default is \"categories.json\").")
    
    # Parse the arguments
    args = parser.parse_args()

    # do the thing :)
    categories_set = extract_categories_from_vgg_csv_dir(args.src_dir)
    categories_set_clean = clean_categories(categories_set)
    coco_categories = create_coco_categories_from_set(categories_set_clean)
    print(f"Extracted {len(coco_categories)} categories from the annotations.")
    save_categories_to_file(cats=coco_categories, mappings=name_mappings, dest_dir=args.dest_dir, filename=args.filename)
    


if __name__ == "__main__":
    main()