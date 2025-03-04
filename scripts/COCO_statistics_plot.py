import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import argparse

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

ann_dir = "data-annotations/pitfall-cameras/merged-by-location"  


def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Plots different visualisations from a generated statistics-file.")
    parser.add_argument("stats_path", nargs="?", help="Path to statistics-.json file.", default="data-annotations/pitfall-cameras/info/statistics/statistics.json")
    
    # Parse the arguments
    args = parser.parse_args()

    # Open stats...
    with open(args.stats_path, 'r') as f:
        stats = json.load(f)
    
    # Plot!
    plot_and_save_class_distribution(stats["overall"]["class distribution"])
    plot_negative_sample_percentages(stats["per file"])
    plot_category_prevalence_per_file(stats["per file"])

if __name__ == "__main__":
    main()