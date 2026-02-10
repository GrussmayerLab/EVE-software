
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def extract_bias_numeric(name):
    """
    Tries to extract a numeric value from the directory name for sorting.
    Handles 'plus10', 'minus10', 'defaultbias', '_50', and '_refr_50' style.
    """
    # Handle default bias
    if 'defaultbias' in name.lower():
        return 0

    # Look for plus/minus followed by digits
    match = re.search(r'(?:plus|minus)(\d+)', name, re.IGNORECASE)
    if match:
        val = int(match.group(1))
        if 'minus' in name.lower():
            return -val
        return val
    
    # Look for underscore followed by digits at the end (e.g., _50)
    match = re.search(r'_(\d+)$', name)
    if match:
        return int(match.group(1))

    # Fallback: look for any sequence of digits at the end
    match = re.search(r'(\d+)$', name)
    if match:
        return int(match.group(1))

    # Fallback to alphabetical order if no number found
    return name

def plot_overlapping_metrics(analysis_data, analysis_name, output_dir, group_name=""):
    """
    Generates plots overlapping results from multiple datasets for a given analysis type.
    analysis_data: dictionary {dataset_name: dataframe}
    group_name: prefix for the output file (e.g., Green15, Red75)
    """
    # Collect all possible metrics across all datasets for this analysis
    all_metrics = set()
    for df in analysis_data.values():
        all_metrics.update([c.replace("_mean", "") for c in df.columns if c.endswith("_mean")])
    
    metrics = sorted(list(all_metrics))
    if not metrics:
        return []

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 6 * n_metrics), squeeze=False)
    
    # Distinctive color cycle
    colors = plt.cm.tab10(np.linspace(0, 1, len(analysis_data)))
    
    plot_files = []
    
    for i, metric in enumerate(metrics):
        ax = axes[i, 0]
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        
        for (dataset_name, df), color in zip(analysis_data.items(), colors):
            if mean_col not in df.columns:
                continue
                
            plot_df = df.sort_values('bias_val')
            x_values = plot_df['bias_val']
            y_values = plot_df[mean_col]
            
            # If bias_val has non-numeric entries, we might need to handle them
            # For plotting, we'll try to convert to numeric, others will be strings on x-axis
            x_is_numeric = all(isinstance(x, (int, float)) for x in x_values)
            
            label_name = dataset_name.replace("Green15_", "")
            
            if std_col in plot_df.columns:
                y_err = plot_df[std_col]
                ax.errorbar(x_values, y_values, yerr=y_err, fmt='o-', capsize=5, 
                            label=label_name, color=color, linewidth=2, markersize=8, alpha=0.8)
            else:
                ax.plot(x_values, y_values, 'o-', label=label_name, 
                        color=color, linewidth=2, markersize=8, alpha=0.8)

        ax.set_title(f"{analysis_name}: {metric}", fontsize=16, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.set_xlabel("Bias Value (Extracted)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Dataset / Bias Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    prefix = f"{group_name}_" if group_name else ""
    plot_path = os.path.join(output_dir, f"{prefix}{analysis_name}_overlap_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return [plot_path]

def plot_combined_dashboard(summary_data, output_dir, group_name=""):
    """
    Creates a single dashboard with key metrics from different analyses.
    group_name: prefix for output file
    """
    # Key metrics we want to show
    key_metrics = [
        ('areaOfContinuousContrast', 'area'),
        ('eventAnalysis', 'rate_keps'),
        ('eventAnalysis', 'total_events'),
        ('eventStructuraRatio', 'score'),
        ('eventAnalysis', 'fill_factor_percent'),
        ('eventAnalysis', 'on_off_ratio'),
        ('eventAnalysis', 'active_pixels')
    ]
    
    # Filter only available ones
    available_metrics = []
    for analysis, metric in key_metrics:
        if analysis in summary_data:
            # Check if metric exists in any of the datasets
            exists = False
            for df in summary_data[analysis].values():
                if f"{metric}_mean" in df.columns:
                    exists = True
                    break
            if exists:
                available_metrics.append((analysis, metric))
                
    if not available_metrics:
        return None

    n = len(available_metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)
    
    # Collect all dataset names for coloring
    all_datasets = set()
    for datasets in summary_data.values():
        all_datasets.update(datasets.keys())
    
    sorted_datasets = sorted(list(all_datasets))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_datasets)))
    dataset_colors = dict(zip(sorted_datasets, colors))

    for i, (analysis, metric) in enumerate(available_metrics):
        ax = axes[i, 0]
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        
        datasets = summary_data[analysis]
        for dataset_name, df in datasets.items():
            if mean_col not in df.columns:
                continue
                
            plot_df = df.sort_values('bias_val')
            x = plot_df['bias_val']
            y = plot_df[mean_col]
            label = dataset_name.replace("Green15_", "") # Clean up label
            
            color = dataset_colors[dataset_name]
            
            if std_col in plot_df.columns:
                ax.errorbar(x, y, yerr=plot_df[std_col], fmt='o-', capsize=5, 
                            label=label, color=color, linewidth=2, alpha=0.8)
            else:
                ax.plot(x, y, 'o-', label=label, color=color, linewidth=2, alpha=0.8)

        ax.set_title(f"{analysis}: {metric}", fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xlabel("Bias Level", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    prefix = f"{group_name}_" if group_name else ""
    plot_path = os.path.join(output_dir, f"{prefix}global_analysis_dashboard.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path

def find_and_analyze_csvs(root_dir): 
    """
    Recursively finds 'directory_summary.csv' files, groups them by analysis type AND dataset,
    separates Green15 vs Red75, and generates reports.
    """
    # Structure: { group_name: { analysis_type: { dataset_name: dataframe } } }
    grouped_data = {
        "Green15": {},
        "Green15_refr": {},
        "Red75": {},
        "Red75_refr": {},
        "Other": {}
    }
    
    target_name = "directory_summary.csv"
    
    for root, dirs, files in os.walk(root_dir):
        if target_name in files:
            file_path = os.path.join(root, target_name)
            
            # Extract analysis type from directory name
            # Leaf folder is usually 'Analysis_Results_<type>.run_analysis_...'
            parent_dir = os.path.basename(root)
            analysis_type = "Unknown"
            if "Analysis_Results_" in parent_dir:
                analysis_type = parent_dir.replace("Analysis_Results_", "").split(".")[0]
            else:
                analysis_type = parent_dir
            
            # Extract dataset name (parent of the analysis folder)
            dataset_dir = os.path.dirname(root)
            dataset_name = os.path.basename(dataset_dir)
            
            if dataset_name == "" or dataset_name == ".":
                dataset_name = "Root"

            # Determine Group
            if dataset_name.startswith("Green15"):
                if "refr" in dataset_name.lower():
                    group = "Green15_refr"
                else:
                    group = "Green15"
            elif dataset_name.startswith("Red75"):
                if "refr" in dataset_name.lower():
                    group = "Red75_refr"
                else:
                    group = "Red75"
            else:
                group = "Other"

            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    continue
                
                # Add bias sorting column
                if 'directory' in df.columns:
                    df['bias_val'] = df['directory'].apply(extract_bias_numeric)
                
                if analysis_type not in grouped_data[group]:
                    grouped_data[group][analysis_type] = {}
                grouped_data[group][analysis_type][dataset_name] = df
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Create output directory for batch summary
    report_dir = os.path.join(root_dir, "batch_comparison_reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Process each group separately
    for group_name, summary_data in grouped_data.items():
        if not summary_data:
            continue
            
        print(f"\n=== Processing Group: {group_name} ===")
        print(f"Found {len(summary_data)} analysis types.")

        for analysis_type, datasets in summary_data.items():
            print(f"\nGenerating overlapping plots for: {analysis_type} ({group_name})")
            print(f"  Datasets: {', '.join(datasets.keys())}")
            
            plot_files = plot_overlapping_metrics(datasets, analysis_type, report_dir, group_name)
            if plot_files:
                print(f"  Saved plot: {os.path.basename(plot_files[0])}")
                
            # Save individual summary tables grouped by analysis and group
            for dataset_name, df in datasets.items():
                out_name = f"{group_name}_{analysis_type}_{dataset_name}_table.csv"
                df.sort_values('bias_val').to_csv(os.path.join(report_dir, out_name), index=False)

        print(f"\nGenerating Global Dashboard for {group_name}...")
        dashboard_file = plot_combined_dashboard(summary_data, report_dir, group_name)
        if dashboard_file:
            print(f"  Saved dashboard: {os.path.basename(dashboard_file)}")

    print(f"\nAll reports and plots saved to: {report_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze batch results and overlap multiple datasets.")
    parser.add_argument("--root", type=str, default=".", help="Root directory containing multiple datasets.")
    args = parser.parse_args()

    find_and_analyze_csvs(args.root)
