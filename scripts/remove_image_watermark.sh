#!/bin/bash 

if [ "$#" -ne 2 ]; then
  echo "Error: Exactly two folders must be specified."
  echo "Usage: $0 <source-folder> <target-folder> (Overwrites images if same folder)"
  exit 1
fi

input_dir=$1    
output_dir=$2
mkdir -p "$output_dir"

for img in "$input_dir"/*; do
    filename=$(basename "$img")
    magick "$img" -crop "100x93%+0+0" "$output_dir/$filename"
    echo "Cropped $filename"
done

echo "Cropping complete. Images stored in '$output_dir'"