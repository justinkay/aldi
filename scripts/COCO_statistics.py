import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np


def sort_by_value(keys, vals):
    sorted_indices = np.argsort(vals)[::-1]  # Get indices for sorting
    sorted_keys = [keys[i] for i in sorted_indices]
    sorted_vals = [vals[i] for i in sorted_indices]
    return sorted_keys, sorted_vals

def plot_category_prevalence_per_file(statistics_dict, output_path="data-annotations/pitfall-cameras/info/statistics/categories_per_file.png"):
    """
    Creates and saves bar charts showing the prevalence of each class across multiple datasets.

    :param statistics_dict: Dictionary with dataset statistics.
    :param output_path: Path to save the final image.
    """
    datasets_unsorted = list(statistics_dict.keys())
    num_datasets = len(datasets_unsorted)
    
    # Prepare for class distributions (collect all classes across datasets)
    class_names = set()
    for stats in statistics_dict.values():
        class_names.update(stats["class distribution"].keys())
    class_names = sorted(list(class_names))  # Sort for consistent ordering
    
    # Create a figure with a subplot for each class
    num_classes = len(class_names)
    fig, axs = plt.subplots(num_classes, 1, figsize=(max(10, 0.5 * num_datasets), 5 * num_classes))

    if num_classes == 1:
        axs = [axs]  # Make sure axs is iterable even if we have only one class

    for idx, class_name in enumerate(class_names):
        # Collect the class counts for each dataset
        class_counts = [stats["class distribution"].get(class_name, 0) for stats in statistics_dict.values()]
        datasets, class_counts = sort_by_value(keys=datasets_unsorted, vals=class_counts)
        # Plot the bar chart for this class
        axs[idx].bar(datasets, class_counts, color='skyblue')
        axs[idx].set_title(f"Prevalence of {class_name}")
        axs[idx].set_ylabel(f"Count of {class_name}")
        axs[idx].set_xlabel("Datasets")
        axs[idx].set_xticks(range(len(datasets)))
        axs[idx].set_xticklabels(datasets, rotation=45, ha="right")

    # Adjust layout for readability
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved category prevalence chart as {output_path}")

def plot_negative_sample_percentages(statistics_dict, output_path="data-annotations/pitfall-cameras/info/statistics/negative_percentage_per_file.png"):
    """
    Creates and saves visualizations comparing negative sample percentages
    across multiple datasets.
    
    :param statistics_dict: Dictionary with dataset statistics.
    :param output_path: Path to save the final image.
    """
    datasets = list(statistics_dict.keys())
    neg_percentages = [round((stats["negative samples"]/stats["total images"])*100 , 2) for stats in statistics_dict.values()]
    datasets, neg_percentages = sort_by_value(datasets, neg_percentages)

    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(datasets)*0.5), 12))

    # Negative sample percentage (Bar chart)
    ax.bar(datasets, neg_percentages, color='skyblue')
    ax.set_title('Negative Sample Percentages')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_ylabel('Negative Sample Percentage (%)')
    ax.set_ylim(0, 100)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved comparative statistics as {output_path}")

def plot_and_save_class_distribution(class_distribution, filename="data-annotations/pitfall-cameras/info/statistics/class_distribution.png"):
    """
    Plots and saves a bar chart of the class distribution.
    
    :param class_distribution: Dictionary {class_name: count}
    :param filename: Output file name
    """
    class_names = list(class_distribution.keys())
    class_counts = list(class_distribution.values())
    class_names, class_counts = sort_by_value(keys=class_names, vals=class_counts)
    plt.figure(figsize=(max(10, len(class_names)*0.5), 5))
    sns.barplot(x=class_names, y=class_counts, palette="viridis", legend=False)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Category")
    plt.ylabel("Number of Instances")
    # plt.ylim(0,30)
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved class distribution chart as {filename}")

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
    plot_and_save_class_distribution(global_stats["class distribution"])
    return {"overall": global_stats, "per file": per_file_stats}

ann_dir = "data-annotations/pitfall-cameras/merged-by-location"  


def main():
    stats = collect_statistics_from_directory(ann_dir)
    with open("data-annotations/pitfall-cameras/info/statistics/statistics.json", 'w') as f:
        json.dump(stats, f, indent=4)
    plot_negative_sample_percentages(stats["per file"])
    plot_category_prevalence_per_file(stats["per file"])

if __name__ == "__main__":
    main()
