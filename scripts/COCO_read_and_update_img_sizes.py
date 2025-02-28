import json
import os
from PIL import Image

def update_image_sizes(coco_json_file_path, image_folder):
    """
    Updates image width and height in COCO JSON based on actual image dimensions.

    :param coco_data: Loaded COCO JSON as a Python dictionary.
    :param image_folder: Path to the folder containing the images.
    """

    with open(coco_json_file_path, "r") as f:
        coco = json.load(f)
    no_of_images = len(coco["images"])

    for index, img in enumerate(coco["images"], start=0):
        # Read the size
        image_path = os.path.join(image_folder, img["file_name"])
        if os.path.exists(image_path):
            with Image.open(image_path) as image:
                img["width"], img["height"] = image.size
        else:
            print(f"Warning: Image not found - {image_path}")
        
        if (index+1) % 100 == 0 or (index+1) == no_of_images:
            print(f"--- Processed {index+1} out of {no_of_images} images.")

    with open(coco_json_file_path, "w") as f:
        json.dump(coco, f, indent=4)

def update_image_sizes_for_all_annotations_in_dir(src_dir, image_folder):
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and os.path.splitext(f)[1].lower().endswith((".json")))]
    total_files = len(files)
       
    for index, filename in enumerate(files, start=1): 
        # Build the full path to the file
        src_file_path = os.path.join(src_dir, filename)
        update_image_sizes(src_file_path, image_folder=image_folder)
        # Print progress every 5 files
        #if index % 5 == 0 or index == total_files: 
        print(f"Processed {index} out of {total_files} annotation files.")


image_folder = "../ERDA/bugmaster/datasets/pitfall-cameras/images/" 
coco_json_dir = "data-annotations/pitfall-cameras/originals-converted/"
coco_json = coco_json_dir + "LG_OSR_HF2F_20-06-29_fl_.json"

def main():
    update_image_sizes_for_all_annotations_in_dir(src_dir=coco_json_dir, image_folder=image_folder)
    # update_image_sizes(coco_json_file_path=coco_json, image_folder=image_folder)
    
if __name__ == "__main__":
    main()