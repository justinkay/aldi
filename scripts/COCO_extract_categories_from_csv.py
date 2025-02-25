import numpy as np
import json
import pandas as pd
import sys
import argparse
import os
import COCO_util as ccu
import re

def extract_category_name_from_region_attributes(attr):
    cat = set()
    try:
        attr_dict = json.loads(attr)
        raw_cat = attr_dict["Insect"].keys() # NOTE - we assume that the supercategory is always insect
        return ccu.normalise_category_name(raw_cat)
    except json.JSONDecodeError:
        cat = set() # ignore malformed json
    return cat

def extract_categories_from_vgg_csv(src):
    df = pd.read_csv(src, on_bad_lines='skip')
    unique_categories = set()  # Store unique category names
    for attributes in df['region_attributes']:
        cat = extract_category_name_from_region_attributes(attributes)
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
        # Print progress every 50 files
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files")
    return categories

def create_coco_categories_from_set(cats:set):
    """Create category dictionary with unique, sorted category names"""
    return [
        {"id": idx + 1, "name": cat, "supercategory": "Insect"}
        for idx, cat in enumerate(sorted(cats))  # Sorted to ensure consistency
    ]

def save_categories_to_file(cats, dest_dir, filename):
    path = os.path.join(dest_dir, filename)
    with open(path, "w") as f:
        json.dump(cats, f, indent=4)


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Extract categories from all VGG-CSV annotations into a single JSON-object (\"categories\") compatible with the COCO-format.")
    parser.add_argument("src_dir", help="Source directory containing the csv files.", default="../ERDA/bugmaster/datasets/pitfall-cameras/annotations/Annotations and other files/CSV files")
    parser.add_argument("dest_dir", help="Directory at which to save the generated categories.json.")
    parser.add_argument("filename", default="categories", help="Optional name of the generated file (default is \"categories.json\").")
    
    # Parse the arguments
    args = parser.parse_args()

    # do the thing :)
    categories_set = extract_categories_from_vgg_csv_dir(args.src_dir)
    coco_categories = create_coco_categories_from_set(categories_set)
    save_categories_to_file(coco_categories, dest_dir=args.dest_dir, filename=args.filename)


if __name__ == "__main__":
    main()