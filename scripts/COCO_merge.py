import json
import os
import argparse
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
    seen_images = set()
    seen_annotations = set()
    total_files = len(coco_list)

    no_of_images = 0
    no_of_annotations = 0
    for index, file in enumerate(coco_list):
        # Load datasets
        with open(file, "r") as f:
            coco = json.load(f)
        
        if merged_categories != coco["categories"]:
            raise ValueError(f"Categories do not match in file {file}")
        
        for img in coco["images"]:
            if img["id"] in seen_images: raise ValueError(f"Image {img["id"]} already seen!")
            merged_images.append(img)
            seen_annotations.add(img["id"])
        
        for ann in coco["annotations"]:
            if ann["id"] in seen_annotations: raise ValueError(f"Annotation {ann["id"]} already seen!")
            merged_annotations.append(ann)
            seen_annotations.add(ann["id"])

        # Print progress every 5 files
        if index % 5 == 0 or index == total_files:
            print(f"--- Processed {index} out of {total_files} files.")
        
        no_of_images += len(coco["images"])
        no_of_annotations += len(coco["annotations"])
   
    print(f"Sum of images: {no_of_images}\nMerged images: {len(merged_images)}")
    print(f"Sum of anns: {no_of_annotations}\nMerged anns: {len(merged_annotations)}")

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
        
    for loc in ccu.LOCATIONS:
        print(f"Merging for location, {loc}...")
        merged = merge_coco_json(location_to_file_list[loc])
        save_merged_annotations(merged=merged, destination_path=dest_dir+loc+".json")
        print(f"Finished merging for location, {loc}.")


def merge_all_in_dir(src_dir, dest_path):
    files_to_merge = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith(".json")]
    merged = merge_coco_json(coco_list=files_to_merge)
    merged["info"] = "Annotations for object detections in the images collection in the ECOSTACK-project's experiment with pitfall traps and wild cameras. The images were taken in June-July 2020 in oil rapeseed and winter wheat fields in the UK belonging to Geescroft & Highfield, Long Hoos & Great Knott, and Whitehorse & Webbs. Converted to COCO-format by Stinna Danger and Mikkel Berg for their thesis project at Aarhus University at the Department of Computer Science with biodiversity group at the Department of Ecoscience."
    save_merged_annotations(merged=merged, destination_path=dest_path)
    
def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Merge the json files for COCO-datasets in a directory.")
    parser.add_argument("src_dir", nargs="?", help="Source directory containing the individual COCO-files you want to merge.", default="data-annotations/pitfall-cameras/originals-converted/")
    parser.add_argument("dest_dir", nargs="?", help="Directory at which to save the merged .json-file.", default="data-annotations/pitfall-cameras/merged-by-location/")
    parser.add_argument('-loc', action='store_true', help="Add flag if you wish to merge by location.")
    
    # Parse the arguments
    args = parser.parse_args()
    if args.loc: merge_by_location(src_dir=args.src_dir, dest_dir=args.dest_dir)
    else: merge_all_in_dir(src_dir=args.src_dir, dest_path="data-annotations/pitfall-cameras/pitfall-cameras_all.json")

if __name__ == "__main__":
    main()