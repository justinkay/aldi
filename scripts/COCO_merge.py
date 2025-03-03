import json
import os
import COCO_util as ccu 

def merge_coco_json(coco_list):
    """ 
    Merges the COCO JSON files in the provided list to a single JSON object.
    NOTE the function assumes that the categories match and that there are no 
    clashes in image or annotation ids (this is handled by our naming convention 
    when we generated the individual files)

    :param coco_list: List of file paths to the COCO-JSON files we want to merge.
    """

    with open(coco_list[0], "r") as f:
        coco0 = json.load(f)
    merged_info = coco0["info"]
    merged_license = coco0["license"]
    merged_categories = coco0["categories"]

    merged_images = []
    merged_annotations = []
    total_files = len(coco_list)
    for index, file in enumerate(coco_list):
        # Load datasets
        with open(file, "r") as f:
            coco = json.load(f)
        
        if merged_categories != coco["categories"]:
            raise ValueError(f"Categories do not match in file {file}")
        
        for img in coco["images"]:
            merged_images.append(img)
        
        for ann in coco["annotations"]:
            merged_annotations.append(ann)

        # Print progress every 2 files
        if index % 2 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files")

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


def merge_by_location(field, crop, camera, src_dir="data-annotations/pitfall-cameras/originals-converted/"):
    # Find the files pertaining to the given field, crop 
    prefix = "_".join([field, crop, camera])
    files_to_merge = []
    for f in os.listdir(src_dir):
        if (not os.path.isfile(os.path.join(src_dir, f))): continue
        if not f.endswith(".json"): continue
        is_desired_location = f.startswith(prefix)
        if is_desired_location:
            files_to_merge.append(os.path.join(src_dir, f))
    merged = merge_coco_json(files_to_merge)
    
    return merged


def main():
    destination_dir = "data-annotations/pitfall-cameras/merged-by-location/"
    src_dir = "data-annotations/pitfall-cameras/originals-converted/"
    destination = destination_dir + "GH-OSR-HF2G.json"
    merged = merge_by_location(field="GH", crop="OSR", camera="HF2G")
    save_merged_annotations(merged=merged, destination_path=destination)


if __name__ == "__main__":
    main()