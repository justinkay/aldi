import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

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

def standardise_category_name(name:str):
    normalized = normalise_category_name(name)
    
    stemmer = PorterStemmer() # handles pluralis
    lemmatized = " ".join([stemmer.stem(word) for word in normalized.split()])
    
    #standardized = " ".join(sorted(lemmatized.split()))  # Sort words alphabetically
    return lemmatized