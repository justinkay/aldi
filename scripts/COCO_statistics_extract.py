import json
import os
from collections import Counter

def update_global_stats(global_stats, stats_to_add):
    updated = global_stats.copy()
    updated["total images"] += stats_to_add["total images"]
    updated["total annotations"] += stats_to_add["total annotations"]
    updated["positive samples"] +=  stats_to_add["positive samples"]
    updated["negative samples"] += stats_to_add["negative samples"]
    for cls, count in stats_to_add.get("class distribution", {}).items():
        if cls in updated["class distribution"]: updated["class distribution"][cls] += count
        else: updated["class distribution"][cls] = count
    return updated

def get_coco_statistics(coco):
    """
    Extracts statistics from a COCO JSON dataset.
    
    :param coco_data: Loaded COCO JSON as a dictionary.
    :return: A dictionary with dataset statistics.
    """
    image_count = len(coco["images"])  
    annotation_count = len(coco["annotations"]) 
    category_count = Counter() # occurrences of each category, Counter({id: count, id: count, id: count, ...})
    image_to_ann_count = Counter()  # image id -> annotation count

    for ann in coco["annotations"]:
        image_to_ann_count[ann["image_id"]] += 1
        category_count[ann["category_id"]] += 1
        
    positive_samples = sum(1 for count in image_to_ann_count.values() if count > 0)
    negative_samples = image_count - positive_samples

    category_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]} # category id -> category name
    class_distribution_unsorted = {category_id_to_name[cid]: count for cid, count in category_count.items()}
    class_distribution = dict(sorted(class_distribution_unsorted.items()))

    return {
        "total images": image_count,
        "total annotations": annotation_count,
        "positive samples": positive_samples,
        "negative samples": negative_samples,
        "class distribution": class_distribution
    }

def collect_statistics_from_directory(json_dir):
    """
    Collects statistics across multiple COCO JSON files.
    
    :param json_dir: Path to the directory containing COCO JSON files.
    """
    global_stats = {}
    per_file_stats = {} 
    
    files = [f for f in os.listdir(json_dir) if (os.path.isfile(os.path.join(json_dir, f)) and f.lower().endswith((".json")))]
    total_files = len(files)
    for index, filename in enumerate(files, start=1):
        json_path = os.path.join(json_dir, filename)
        with open(json_path, "r") as f:
            coco_data = json.load(f)
        this_files_stats = get_coco_statistics(coco_data)
        per_file_stats[filename] = this_files_stats

        if index == 1: global_stats = this_files_stats.copy()
        else: global_stats = update_global_stats(global_stats=global_stats, stats_to_add=this_files_stats)
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files.")

    print(json.dumps(global_stats, indent=4))
    return {"overall": global_stats, "per file": per_file_stats}

ann_dir = "data-annotations/pitfall-cameras/merged-by-location"  


def main():
    stats = collect_statistics_from_directory(ann_dir)
    with open("data-annotations/pitfall-cameras/info/statistics/statistics.json", 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
