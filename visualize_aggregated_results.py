#!/usr/bin/env python3
"""
Visualize aggregated results from repeated experiments.
Creates plots showing mean trends with 95% confidence intervals.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# LOAD DATA
# ============================================================================
aggregated_file = "./results_aggregated_by_size.csv"

if not os.path.exists(aggregated_file):
    print(f"❌ Error: {aggregated_file} not found!")
    print("Run 'python run_repeated_experiments.py' first.")
    exit(1)

df = pd.read_csv(aggregated_file)
print(f"Loaded {len(df)} aggregated statistics")

# ============================================================================
# PLOT CONFIGURATION
# ============================================================================
# Key metrics to visualize
key_metrics = ["λ₂", "AvgNodeConn", "GCC_5%", "EffResistance", "NatConnectivity"]
rewards = ["PBR", "EFFRES", "IVI", "NNSI"]
size_buckets = ["Small", "Medium", "Large"]

# Color palette
colors = {
    "PBR": "#2E86AB",
    "EFFRES": "#A23B72",
    "IVI": "#F18F01",
    "NNSI": "#C73E1D"
}

output_dir = "./aggregated_plots"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# PLOT 1: MEAN VALUES WITH 95% CI BY SIZE BUCKET
# ============================================================================
print("\nCreating mean value plots with 95% CI...")

for metric in key_metrics:
    df_metric = df[df["Metric"] == metric]
    
    if len(df_metric) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(size_buckets))
    width = 0.2
    
    for i, reward in enumerate(rewards):
        df_reward = df_metric[df_metric["Reward"] == reward]
        
        means = []
        ci_lows = []
        ci_highs = []
        
        for size in size_buckets:
            row = df_reward[df_reward["SizeBucket"] == size]
            if len(row) > 0:
                means.append(row["Mean"].values[0])
                ci_lows.append(row["CI_95_Low"].values[0])
                ci_highs.append(row["CI_95_High"].values[0])
            else:
                means.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
        
        # Calculate error bars (distance from mean to CI bounds)
        errors_low = np.array(means) - np.array(ci_lows)
        errors_high = np.array(ci_highs) - np.array(means)
        errors = [errors_low, errors_high]
        
        # Plot bars with error bars
        positions = x_positions + (i - 1.5) * width
        ax.bar(positions, means, width, label=reward, 
               color=colors[reward], alpha=0.8)
        ax.errorbar(positions, means, yerr=errors, fmt='none', 
                   color='black', capsize=3, linewidth=1, alpha=0.6)
    
    ax.set_xlabel('Network Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} by Network Size (Mean ± 95% CI)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(size_buckets)
    ax.legend(title='Reward Function', loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{output_dir}/mean_ci_{metric.replace('₂', '2').replace('%', 'pct')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ============================================================================
# PLOT 2: IMPROVEMENT PERCENTAGES WITH CI
# ============================================================================
print("\nCreating improvement plots...")

for metric in key_metrics:
    df_metric = df[df["Metric"] == metric]
    
    if len(df_metric) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(size_buckets))
    width = 0.2
    
    for i, reward in enumerate(rewards):
        df_reward = df_metric[df_metric["Reward"] == reward]
        
        improvements = []
        ci_lows = []
        ci_highs = []
        
        for size in size_buckets:
            row = df_reward[df_reward["SizeBucket"] == size]
            if len(row) > 0:
                improvements.append(row["Mean_Improvement_%"].values[0])
                ci_lows.append(row["CI_95_Improvement_Low"].values[0])
                ci_highs.append(row["CI_95_Improvement_High"].values[0])
            else:
                improvements.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)
        
        # Calculate error bars
        errors_low = np.array(improvements) - np.array(ci_lows)
        errors_high = np.array(ci_highs) - np.array(improvements)
        errors = [errors_low, errors_high]
        
        # Plot bars with error bars
        positions = x_positions + (i - 1.5) * width
        ax.bar(positions, improvements, width, label=reward, 
               color=colors[reward], alpha=0.8)
        ax.errorbar(positions, improvements, yerr=errors, fmt='none', 
                   color='black', capsize=3, linewidth=1, alpha=0.6)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Network Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Improvement vs Original (Mean ± 95% CI)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(size_buckets)
    ax.legend(title='Reward Function', loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{output_dir}/improvement_{metric.replace('₂', '2').replace('%', 'pct')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ============================================================================
# PLOT 3: HEATMAP OF MEAN VALUES
# ============================================================================
print("\nCreating heatmaps...")

for reward in rewards:
    df_reward = df[df["Reward"] == reward]
    
    # Create pivot table
    pivot = df_reward.pivot_table(
        index="Metric",
        columns="SizeBucket",
        values="Mean"
    )
    
    # Reorder columns
    pivot = pivot[size_buckets]
    
    # Only include key metrics
    pivot = pivot.loc[pivot.index.isin(key_metrics)]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'Mean Value'}, ax=ax)
    ax.set_title(f'{reward} - Mean Values by Size and Metric', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Network Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    filename = f"{output_dir}/heatmap_{reward}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ============================================================================
# PLOT 4: TREND LINES ACROSS SIZE BUCKETS
# ============================================================================
print("\nCreating trend line plots...")

for metric in key_metrics:
    df_metric = df[df["Metric"] == metric]
    
    if len(df_metric) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_numeric = {"Small": 1, "Medium": 2, "Large": 3}
    
    for reward in rewards:
        df_reward = df_metric[df_metric["Reward"] == reward]
        
        x_vals = []
        y_vals = []
        ci_lows = []
        ci_highs = []
        
        for size in size_buckets:
            row = df_reward[df_reward["SizeBucket"] == size]
            if len(row) > 0:
                x_vals.append(x_numeric[size])
                y_vals.append(row["Mean"].values[0])
                ci_lows.append(row["CI_95_Low"].values[0])
                ci_highs.append(row["CI_95_High"].values[0])
        
        # Plot line with markers
        ax.plot(x_vals, y_vals, marker='o', linewidth=2, 
               markersize=8, label=reward, color=colors[reward])
        
        # Add confidence interval as shaded area
        ax.fill_between(x_vals, ci_lows, ci_highs, 
                        alpha=0.2, color=colors[reward])
    
    ax.set_xlabel('Network Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Trend Across Network Sizes (with 95% CI)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(size_buckets)
    ax.legend(title='Reward Function', loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{output_dir}/trend_{metric.replace('₂', '2').replace('%', 'pct')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print(f"VISUALIZATION COMPLETE")
print(f"{'='*80}")
print(f"\nAll plots saved to: {output_dir}/")
print(f"\nPlot types created:")
print(f"  1. Mean values with 95% CI bars")
print(f"  2. Improvement percentages with 95% CI")
print(f"  3. Heatmaps of mean values by reward function")
print(f"  4. Trend lines across size buckets with CI bands")
print(f"\n{'='*80}\n")
