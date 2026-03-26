#!/usr/bin/env python3
"""
Visualize aggregated results from repeated experiments.
Creates plots showing mean trends with 95% confidence intervals.
Includes greedy baselines and attack curve AUC.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 9

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
key_metrics = ["λ₂", "AvgNodeConn", "GCC_5%", "AttackCurveAUC", "EffResistance", "NatConnectivity"]
rewards = ["PBR", "EFFRES", "IVI", "NNSI", "GREEDY_DEG", "GREEDY_BET", "GREEDY_ALG"]
size_buckets = ["Small", "Medium", "Large"]

# Color palette
colors = {
    "PBR": "#2E86AB",
    "EFFRES": "#A23B72",
    "IVI": "#F18F01",
    "NNSI": "#C73E1D",
    "GREEDY_DEG": "#6A994E",
    "GREEDY_BET": "#BC4749",
    "GREEDY_ALG": "#386641"
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
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_positions = np.arange(len(size_buckets))
    n_rewards = len(rewards)
    width = 0.8 / n_rewards
    
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
        
        # Calculate error bars
        errors_low = np.array(means) - np.array(ci_lows)
        errors_high = np.array(ci_highs) - np.array(means)
        errors = [errors_low, errors_high]
        
        # Plot bars with error bars
        positions = x_positions + (i - n_rewards/2 + 0.5) * width
        ax.bar(positions, means, width, label=reward, 
               color=colors[reward], alpha=0.8)
        ax.errorbar(positions, means, yerr=errors, fmt='none', 
                   color='black', capsize=2, linewidth=0.8, alpha=0.6)
    
    ax.set_xlabel('Network Size', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric}', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} by Network Size (Mean ± 95% CI)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(size_buckets)
    ax.legend(title='Method', loc='best', ncol=2, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{output_dir}/mean_ci_{metric.replace('₂', '2').replace('%', 'pct')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ============================================================================
# PLOT 2: IMPROVEMENT WITH CI (RL methods only)
# ============================================================================
print("\nCreating improvement plots (RL methods)...")

rl_methods = ["PBR", "EFFRES", "IVI", "NNSI"]

for metric in key_metrics:
    df_metric = df[df["Metric"] == metric]
    df_metric = df_metric[df_metric["Reward"].isin(rl_methods)]
    
    if len(df_metric) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_positions = np.arange(len(size_buckets))
    width = 0.2
    
    for i, reward in enumerate(rl_methods):
        df_reward = df_metric[df_metric["Reward"] == reward]
        
        improvements = []
        ci_lows = []
        ci_highs = []
        
        for size in size_buckets:
            row = df_reward[df_reward["SizeBucket"] == size]
            if len(row) > 0:
                improvements.append(row["Mean_Improvement"].values[0])
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
    ax.set_ylabel('Improvement', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Improvement vs Original (Mean ± 95% CI)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(size_buckets)
    ax.legend(title='RL Method', loc='best')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filename = f"{output_dir}/improvement_{metric.replace('₂', '2').replace('%', 'pct')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")

# ============================================================================
# PLOT 3: COMPARISON WITH GREEDY BASELINES
# ============================================================================
print("\nCreating comparison plots with greedy baselines...")

for metric in key_metrics:
    df_metric = df[df["Metric"] == metric]
    
    if len(df_metric) == 0:
        continue
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, size in enumerate(size_buckets):
        ax = axes[idx]
        df_size = df_metric[df_metric["SizeBucket"] == size]
        
        means = []
        ci_lows = []
        ci_highs = []
        labels = []
        
        for reward in rewards:
            row = df_size[df_size["Reward"] == reward]
            if len(row) > 0:
                means.append(row["Mean"].values[0])
                ci_lows.append(row["CI_95_Low"].values[0])
                ci_highs.append(row["CI_95_High"].values[0])
                labels.append(reward)
        
        if not means:
            continue
        
        # Calculate error bars
        errors_low = np.array(means) - np.array(ci_lows)
        errors_high = np.array(ci_highs) - np.array(means)
        errors = [errors_low, errors_high]
        
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, means, color=[colors[r] for r in labels], alpha=0.8)
        ax.errorbar(x_pos, means, yerr=errors, fmt='none', 
                   color='black', capsize=3, linewidth=1, alpha=0.6)
        
        ax.set_title(f'{size} Networks', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric}', fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'{metric} - Comparison Across Methods', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f"{output_dir}/comparison_{metric.replace('₂', '2').replace('%', 'pct')}.png"
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
print(f"  1. Mean values with 95% CI bars (all methods)")
print(f"  2. Improvement percentages with 95% CI (RL methods only)")
print(f"  3. Method comparison by size bucket")
print(f"\n{'='*80}\n")
