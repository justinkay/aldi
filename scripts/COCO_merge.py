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
            print(f"--- Processed {index} out of {total_files} files.")

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

def merge_by_location(src_dir="data-annotations/pitfall-cameras/originals-converted/", dest_dir= "data-annotations/pitfall-cameras/merged-by-location/"):
    # Create lists of paths to all files to merge for each location
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
    location_to_file_list = {}
    for loc in ccu.LOCATIONS:
        location_to_file_list[loc] = []
    for f in files:
        # if file pertains to this location, add its path to the respective list
        file_prefix = "_".join(f.split("_")[0:3])
        if file_prefix not in location_to_file_list.keys(): raise ValueError(f"Location does not exist for file, {f}")
        location_to_file_list[file_prefix].append(os.path.join(src_dir, f))
    print(location_to_file_list)
        
    for loc in ccu.LOCATIONS:
        merged = merge_coco_json(location_to_file_list[loc])
        save_merged_annotations(merged=merged, destination_path=dest_dir+loc+".json")


def merge_all_in_dir(src_dir, dest_path):
    files_to_merge = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
    merged = merge_coco_json(coco_list=files_to_merge)
    save_merged_annotations(merged=merged, destination_path=dest_path)
    
def main():
    merge_by_location()





if __name__ == "__main__":
    main()