import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def plot_and_save_class_distribution(class_distribution, filename="data-annotations/pitfall-cameras/info/statistics_class_distribution_zoomedmore.png"):
    """
    Plots and saves a bar chart of the class distribution.
    
    :param class_distribution: Dictionary {class_name: count}
    :param filename: Output file name
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()), palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Category")
    plt.ylabel("Number of Instances")
    plt.ylim(5,30)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved class distribution chart as {filename}")

def update_global_stats(global_stats, stats_to_add):
    updated = global_stats
    print(type(updated["total annotations"]))
    print(type(stats_to_add["total annotations"]))
    updated["total images"] += stats_to_add["total images"]
    updated["total annotations"] += stats_to_add["total annotations"]
    updated["positive samples"] +=  stats_to_add["positive samples"]
    updated["negative samples"] += stats_to_add["negative samples"]
    updated["positive samples (%)"] =  round(((updated["positive samples"]/updated["total images"])*100),2) # round((((stats_to_add["annotated images (%)"] / 100) * stats_to_add["total images"] + (stats_to_add["annotated images (%)"] / 100) * global_stats["total images"]) / (updated["total images"]) * 100), 2),
    updated["negative samples (%)"] = 100 - updated["positive samples"] # round(((updated["negative samples"]/updated["total images"])*100),2)
    updated["class distribution"].update(stats_to_add["class distribution"])
    return updated

def get_coco_statistics(coco):
    """
    Extracts statistics from a COCO JSON dataset.
    
    :param coco_data: Loaded COCO JSON as a dictionary.
    :return: A dictionary with dataset statistics.
    """
    image_count = len(coco["images"])  
    annotation_count = len(coco["annotations"]) 
    category_count = Counter() # occurrences of each category
    image_to_ann_count = {img["id"]: 0 for img in coco["images"]}  # image id -> annotation count

    for ann in coco["annotations"]:
        image_to_ann_count[ann["image_id"]] += 1
        category_count[ann["category_id"]] += 1
    
    positive_samples = sum(1 for count in image_to_ann_count.values() if count > 0)
    positive_percentage = round((positive_samples / image_count) * 100 if image_count > 0 else 0,2) # don't divide by zero lol 
    negative_samples = image_count - positive_samples
    negative_percentage = 100 - positive_percentage

    category_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]} # category id -> category name
    class_distribution = {category_id_to_name[cid]: count for cid, count in category_count.items()}

    return {
        "total images": image_count,
        "total annotations": annotation_count,
        "positive samples": positive_samples,
        "negative samples": negative_samples,
        "positive samples (%)": negative_percentage,
        "negative samples (%)": positive_percentage,
        "class distribution": class_distribution
    }

def collect_statistics_from_directory(json_dir):
    """
    Collects statistics across multiple COCO JSON files.
    
    :param json_dir: Path to the directory containing COCO JSON files.
    """
    global_stats = {}
    per_file_stats = {} 
    
    cwd = os.getcwd()
    files = [f for f in os.listdir(json_dir) if (os.path.isfile(os.path.join(json_dir, f)) and os.path.splitext(f)[1].lower().endswith((".csv")))]
    total_files = len(files)
    for index, filename in enumerate(os.listdir(json_dir), start=1):
        if filename.endswith(".json"):
            json_path = os.path.join(json_dir, filename)
            with open(json_path, "r") as f:
                coco_data = json.load(f)

            this_files_stats = get_coco_statistics(coco_data)
            per_file_stats[filename] = this_files_stats

            if index == 1: global_stats = this_files_stats
            global_stats = update_global_stats(global_stats=global_stats, stats_to_add=this_files_stats)
        if index % 5 == 0 or index == total_files:
            print(f"Processed {index} out of {total_files} files.")

    print("\n**Overall Statistics Across All Datasets**")
    print(json.dumps(global_stats, indent=4))
    plot_and_save_class_distribution(global_stats["class distribution"])
    return {"overall": global_stats, "per file": per_file_stats}

ann_dir = "data-annotations/pitfall-cameras/originals-converted"  

def main():
    stats = collect_statistics_from_directory(ann_dir)
    with open("data-annotations/pitfall-cameras/info/statistics.json", 'w') as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    main()
