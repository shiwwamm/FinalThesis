#!/usr/bin/env python3
"""
Combine results from 10 separate experiment folders into one aggregated result.

Usage:
    python combine_separate_experiments.py <base_folder_name>
    
Reads from:
    <base_folder_name>_run1/output/results_run1_metrics.csv
    <base_folder_name>_run2/output/results_run1_metrics.csv
    ...
    <base_folder_name>_run10/output/results_run1_metrics.csv
    
Creates:
    <base_folder_name>_combined/
        ├── results_all_runs_combined.csv
        ├── results_aggregated_by_size.csv
        ├── results_summary_mean_by_size.csv
        ├── results_summary_ci95_by_size.csv
        └── results_summary_improvement_by_size.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

if len(sys.argv) < 2:
    print("Usage: python combine_separate_experiments.py <base_folder_name>")
    print("Example: python combine_separate_experiments.py experiment")
    exit(1)

BASE_NAME = sys.argv[1]
N_EXPERIMENTS = 10

# Size buckets
SIZE_SMALL_MAX = 40
SIZE_MEDIUM_MIN = 41
SIZE_MEDIUM_MAX = 93
SIZE_LARGE_MIN = 94

print(f"\n{'='*80}")
print(f"COMBINING SEPARATE EXPERIMENTS")
print(f"{'='*80}")
print(f"Base folder name: {BASE_NAME}")
print(f"Looking for: {BASE_NAME}_run1 through {BASE_NAME}_run10")
print(f"{'='*80}\n")

# ============================================================================
# LOAD ALL RESULTS
# ============================================================================
all_dfs = []
found_experiments = []

for i in range(1, N_EXPERIMENTS + 1):
    folder = f"{BASE_NAME}_run{i}"
    metrics_file = os.path.join(folder, "output", "results_run1_metrics.csv")
    
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        df['Run'] = i  # Renumber as run i
        
        # Add size bucket column
        df['SizeBucket'] = df['N'].apply(lambda n: 
            'Small' if n <= SIZE_SMALL_MAX 
            else 'Medium' if SIZE_MEDIUM_MIN <= n <= SIZE_MEDIUM_MAX 
            else 'Large'
        )
        
        all_dfs.append(df)
        found_experiments.append(i)
        print(f"✓ Loaded {folder}: {len(df)} graphs")
    else:
        print(f"⚠️  Warning: {metrics_file} not found")

if not all_dfs:
    print("\n❌ Error: No result files found!")
    print(f"Make sure folders {BASE_NAME}_run1 through {BASE_NAME}_run10 exist")
    exit(1)

print(f"\n✓ Found {len(all_dfs)} experiments")

# Combine all runs
df_all = pd.concat(all_dfs, ignore_index=True)
print(f"✓ Total graphs across all runs: {len(df_all)}")

# Create output directory
output_dir = f"{BASE_NAME}_combined"
os.makedirs(output_dir, exist_ok=True)
print(f"✓ Created output directory: {output_dir}/")

# Save combined results
combined_file = os.path.join(output_dir, "results_all_runs_combined.csv")
df_all.to_csv(combined_file, index=False)
print(f"✓ Combined results saved to: {combined_file}")

# ============================================================================
# COMPUTE AGGREGATED STATISTICS
# ============================================================================
print(f"\n{'='*80}")
print(f"COMPUTING AGGREGATED STATISTICS")
print(f"{'='*80}\n")

metrics = ["λ₂", "AvgNodeConn", "GCC_5%", "AttackCurveAUC", "ASPL", "Diameter",
           "ArticulationPoints", "Bridges", "BetCentralization", "NatConnectivity",
           "EffResistance", "Assortativity", "AvgClustering"]

# Metrics that can be 0 - use absolute change instead of percentage
absolute_change_metrics = ["AvgClustering", "Assortativity", "BetCentralization"]

def compute_ci_95(data):
    """Compute 95% confidence interval"""
    if len(data) < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
    return ci[0], ci[1]

# Step 1: Compute mean per (Run, SizeBucket)
print("Step 1: Computing run-level means per size bucket...")
run_level_means = df_all.groupby(["Run", "SizeBucket"]).mean(numeric_only=True).reset_index()

print(f"  Run-level means computed: {len(run_level_means)} rows")
print(f"  Runs per size bucket: {run_level_means.groupby('SizeBucket')['Run'].nunique().to_dict()}")

# Step 2: Compute statistics across run means
print("\nStep 2: Computing statistics across run means...")

aggregated_stats = []

for size_bucket in ["Small", "Medium", "Large"]:
    df_size = run_level_means[run_level_means["SizeBucket"] == size_bucket]
    
    if len(df_size) == 0:
        print(f"⚠️  No data for {size_bucket} bucket")
        continue
    
    print(f"\n{size_bucket} bucket: {len(df_size)} run-level means")
    
    # Process all reward types including greedy baselines
    reward_types = ["PBR", "EFFRES", "IVI", "NNSI", "GREEDY_DEG", "GREEDY_BET", "GREEDY_ALG"]
    
    for reward_type in reward_types:
        for metric in metrics:
            orig_col = f"Orig_{metric}"
            reward_col = f"{reward_type}_{metric}"
            
            if reward_col not in df_size.columns or orig_col not in df_size.columns:
                continue
            
            # Get run-level mean values
            orig_values = df_size[orig_col].dropna()
            reward_values = df_size[reward_col].dropna()
            
            if len(reward_values) == 0:
                continue
            
            # Compute statistics
            mean_val = reward_values.mean()
            min_val = reward_values.min()
            max_val = reward_values.max()
            std_val = reward_values.std()
            ci_low, ci_high = compute_ci_95(reward_values)
            
            # Compute improvement statistics
            if len(orig_values) > 0 and len(orig_values) == len(reward_values):
                if metric in absolute_change_metrics:
                    improvements = reward_values.values - orig_values.values
                    improvement_type = "Absolute"
                else:
                    improvements = ((reward_values.values - orig_values.values) / (np.abs(orig_values.values) + 1e-8) * 100)
                    improvement_type = "Percentage"
                
                improvements = improvements[~np.isnan(improvements)]
                
                if len(improvements) > 0:
                    mean_improvement = improvements.mean()
                    std_improvement = improvements.std()
                    ci_imp_low, ci_imp_high = compute_ci_95(improvements)
                else:
                    mean_improvement = np.nan
                    std_improvement = np.nan
                    ci_imp_low, ci_imp_high = np.nan, np.nan
            else:
                mean_improvement = np.nan
                std_improvement = np.nan
                ci_imp_low, ci_imp_high = np.nan, np.nan
                improvement_type = "N/A"
            
            aggregated_stats.append({
                "SizeBucket": size_bucket,
                "Reward": reward_type,
                "Metric": metric,
                "Mean": mean_val,
                "Std": std_val,
                "Min": min_val,
                "Max": max_val,
                "CI_95_Low": ci_low,
                "CI_95_High": ci_high,
                "Mean_Improvement": mean_improvement,
                "Std_Improvement": std_improvement,
                "CI_95_Improvement_Low": ci_imp_low,
                "CI_95_Improvement_High": ci_imp_high,
                "Improvement_Type": improvement_type,
                "N_Runs": len(reward_values),
            })

df_aggregated = pd.DataFrame(aggregated_stats)

# Save aggregated statistics
aggregated_file = os.path.join(output_dir, "results_aggregated_by_size.csv")
df_aggregated.to_csv(aggregated_file, index=False)
print(f"\n✓ Aggregated statistics saved to: {aggregated_file}")

# ============================================================================
# CREATE SUMMARY TABLES
# ============================================================================
print(f"\nCreating summary tables...")

# Pivot table for mean values
pivot_mean = df_aggregated.pivot_table(
    index=["SizeBucket", "Metric"],
    columns="Reward",
    values="Mean"
).reset_index()

pivot_mean_file = os.path.join(output_dir, "results_summary_mean_by_size.csv")
pivot_mean.to_csv(pivot_mean_file, index=False)
print(f"✓ Mean summary saved to: {pivot_mean_file}")

# Pivot table for 95% CI ranges
df_aggregated["CI_95_Range"] = df_aggregated.apply(
    lambda row: f"[{row['CI_95_Low']:.4f}, {row['CI_95_High']:.4f}]" if not pd.isna(row['CI_95_Low']) else "N/A",
    axis=1
)

pivot_ci = df_aggregated.pivot_table(
    index=["SizeBucket", "Metric"],
    columns="Reward",
    values="CI_95_Range",
    aggfunc='first'
).reset_index()

pivot_ci_file = os.path.join(output_dir, "results_summary_ci95_by_size.csv")
pivot_ci.to_csv(pivot_ci_file, index=False)
print(f"✓ 95% CI summary saved to: {pivot_ci_file}")

# Pivot table for mean improvements
pivot_improvement = df_aggregated.pivot_table(
    index=["SizeBucket", "Metric"],
    columns="Reward",
    values="Mean_Improvement"
).reset_index()

pivot_improvement_file = os.path.join(output_dir, "results_summary_improvement_by_size.csv")
pivot_improvement.to_csv(pivot_improvement_file, index=False)
print(f"✓ Improvement summary saved to: {pivot_improvement_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print(f"COMBINATION COMPLETE")
print(f"{'='*80}")
print(f"\nCombined {len(found_experiments)} experiments")
print(f"Total graphs: {len(df_all)}")
print(f"Experiments used: {found_experiments}")

print(f"\n{'='*80}")
print(f"OUTPUT FILES")
print(f"{'='*80}")
print(f"All files saved to: {output_dir}/")
print(f"  - {combined_file}")
print(f"  - {aggregated_file}")
print(f"  - {pivot_mean_file}")
print(f"  - {pivot_ci_file}")
print(f"  - {pivot_improvement_file}")

print(f"\n{'='*80}")
print(f"NEXT STEPS")
print(f"{'='*80}")
print(f"\nVisualize combined results:")
print(f"  python3 visualize_aggregated_results.py {BASE_NAME}_combined")

print(f"\nMain results file:")
print(f"  {aggregated_file}")

print(f"\n{'='*80}\n")
