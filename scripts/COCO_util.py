import re
import json

def create_title(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}.json"


def get_specs(filename):
    pass

def get_prefix_from_specs(field, crop, camera, date, flash=False):
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


def extract_category_name_from_region_attributes(attr):
    cat = set()
    try:
        attr_dict = json.loads(attr)
        keys = list(attr_dict.keys())
        if ("Insect ID" not in keys) and ("Insect" not in keys) and ("Insects" not in keys): print(keys)
        for key in ["Insect ID"]:
            if key in attr_dict:
                return set([attr_dict[key]])
        for key in ["Insect", "Insects"]:
            if key in attr_dict:
                return attr_dict[key].keys() # NOTE - we assume that the supercategory is always insect (or insects, lol)
    except json.JSONDecodeError:
        cat = set() # ignore malformed json
    return cat