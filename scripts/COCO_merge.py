import json

def merge_coco_json(coco1, coco2):
    """ 
    Merges the COCO JSON files in the provided list to a single JSON object.
    NOTE the function assumes that the categories match and that there are no 
    clashes in image or annotation ids (this is handled by our naming convention 
    when we generated the individual files)

    :param coco_list: List of file paths to the COCO-JSON files we want to merge.
    """

    with open(coco1, "r") as f:
        coco0 = json.load(f)
    merged_info = coco0["info"]
    merged_license = coco0["license"]
    merged_categories = coco0["categories"]

    merged_images = []
    merged_annotations = []
    for file in [coco1, coco2]:
        # Load datasets
        with open(file, "r") as f:
            coco = json.load(f)
        
        if merged_categories != coco["categories"]:
            raise ValueError(f"Categories do not match in file {file}")
        
        for img in coco["images"]:
            merged_images.append(img)
        
        for ann in coco["annotations"]:
            merged_annotations.append(ann)

    merged = {}
    merged["info"] = merged_info
    merged["license"] = merged_license
    merged["images"] = merged_images
    merged["annotations"] = merged_annotations
    merged["categories"] = merged_categories

    return merged
         

def save_merged_annotations(merged, destination_path):
    # Save the merged dataset
    with open(destination_path, "w") as f:
        json.dump(merged, f, indent=4)


def main():
    destination_dir = "data-annotations/pitfall-cameras/merged-by-location/"
    src_dir = "data-annotations/pitfall-cameras/originals-converted/"
    destination = destination_dir + "GH-OSR-HF2G"
    merged= merge_coco_json(coco1=src_dir+"GH_OSR_HF2G_20-06-01_fl_.json", coco2=src_dir+"GH_OSR_HF2G_20-06-02_fl_.json")
    save_merged_annotations(merged=merged, destination_path=destination)


if __name__ == "__main__":
    main()