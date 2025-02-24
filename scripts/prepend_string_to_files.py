import os
import argparse
import shutil

def rename_files_in_directory(src_dir, dest_dir, prefix):
    """
    Renames all files in the given directory with the provided prefix.
    """
    os.makedirs(dest_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if (os.path.isfile(os.path.join(src_dir, f)) and os.path.splitext(f)[1].lower().endswith((".png", ".jpg", ".jpeg")))]
    total_files = len(files)

    for index, filename in enumerate(files, start=1):     
        # Build the full path to the file
        src_file_path = os.path.join(src_dir, filename)
        
        new_name = prefix + filename
        dest_file_path = os.path.join(dest_dir, new_name)
            
        # Move the file inkl. prefix
        shutil.move(src_file_path, dest_file_path)

        # Print progress every 50 files
        if index % 50 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files")

def create_prefix(field, crop, camera, date, flash=False):
    location = f"{field}_{crop}_{camera}"
    flashstr = "_fl" if flash else ""
    return f"{location}_{date}{flashstr}_"

def get_dir(field, crop, camera):
    return f"../ERDA/bugmaster/datasets/pitfall-cameras/images/{field}-{crop}-{camera}"

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Prepend prefix to image files for given location and date.")
    parser.add_argument("src_dir", help="source directory")
    parser.add_argument("field", help="Field, e.g. GH for Geescroft and Highfield.")
    parser.add_argument("crop", help="OSR or WWH.")
    parser.add_argument("camera", help="Camera name.")
    parser.add_argument('-f', action='store_true', help="Add flag if flash on")
    parser.add_argument("date", help="date of capture.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    prefix = create_prefix(args.field, args.crop, args.camera, args.date, flash=args.f)
    dir = get_dir(args.field, args.crop, args.camera)
    src_dir = f"../ERDA/bugmaster/datasets/pitfall-cameras/images/{args.src_dir}"
    rename_files_in_directory(src_dir, dir, prefix)

if __name__ == "__main__":
    main()
