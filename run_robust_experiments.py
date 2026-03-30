#!/usr/bin/env python3
"""
Robust experiment runner with proper randomization and confidence intervals.
Ensures each run uses different graph samples and handles large networks efficiently.

Usage:
    python run_robust_experiments.py [output_folder]
"""

import os
import sys
import glob
import random
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import igraph as ig
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
N_RUNS = 10
SAMPLE_PER_SIZE = 5  # Sample 5 graphs per size category per run (20 total)
GRAPH_DIR = "./real_world_topologies"
BASE_SEED = 42

# Size buckets (Jenks Natural Breaks)
SIZE_SMALL_MAX = 40
SIZE_MEDIUM_MIN = 41
SIZE_MEDIUM_MAX = 93
SIZE_LARGE_MIN = 94
SIZE_LARGE_MAX = 299
SIZE_XLARGE_MIN = 300

# Parse command line arguments
if len(sys.argv) > 1:
    BASE_FOLDER = sys.argv[1]
else:
    BASE_FOLDER = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

TEMP_DIR = os.path.join(BASE_FOLDER, "temp")
OUTPUT_DIR = os.path.join(BASE_FOLDER, "output")

# Create directories
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*80}")
print(f"ROBUST REPEATED EXPERIMENTS")
print(f"{'='*80}")
print(f"Configuration:")
print(f"  Runs: {N_RUNS}")
print(f"  Samples per size per run: {SAMPLE_PER_SIZE}")
print(f"  Size categories: Small, Medium, Large, XLarge")
print(f"  Total graphs per run: {SAMPLE_PER_SIZE * 4} = {SAMPLE_PER_SIZE}×4 categories")
print(f"  Total experiments: {N_RUNS * SAMPLE_PER_SIZE * 4} graphs")
print(f"  Base seed: {BASE_SEED}")
print(f"  Output folder: {BASE_FOLDER}/")
print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")

# ============================================================================
# LOAD AND CATEGORIZE GRAPHS
# ============================================================================
def load_and_categorize_graphs(graph_dir):
    """Load all .graphml files and categorize by size"""
    all_files = glob.glob(os.path.join(graph_dir, "*.graphml"))
    
    small_graphs = []
    medium_graphs = []
    large_graphs = []
    xlarge_graphs = []
    
    print(f"Scanning {len(all_files)} topology files...")
    
    for path in all_files:
        try:
            g = ig.Graph.Read_GraphML(path).as_undirected()
            n = g.vcount()
            name = os.path.basename(path).split(".")[0]
            
            if n <= SIZE_SMALL_MAX:
                small_graphs.append((name, path, n))
            elif SIZE_MEDIUM_MIN <= n <= SIZE_MEDIUM_MAX:
                medium_graphs.append((name, path, n))
            elif SIZE_LARGE_MIN <= n <= SIZE_LARGE_MAX:
                large_graphs.append((name, path, n))
            elif n >= SIZE_XLARGE_MIN:
                xlarge_graphs.append((name, path, n))
        except Exception as e:
            print(f"  Warning: Could not load {path}: {e}")
    
    print(f"\nGraph distribution:")
    print(f"  Small (≤{SIZE_SMALL_MAX}): {len(small_graphs)} graphs")
    print(f"  Medium ({SIZE_MEDIUM_MIN}-{SIZE_MEDIUM_MAX}): {len(medium_graphs)} graphs")
    print(f"  Large ({SIZE_LARGE_MIN}-{SIZE_LARGE_MAX}): {len(large_graphs)} graphs")
    print(f"  XLarge (≥{SIZE_XLARGE_MIN}): {len(xlarge_graphs)} graphs")
    print(f"  Total: {len(small_graphs) + len(medium_graphs) + len(large_graphs) + len(xlarge_graphs)} graphs\n")
    
    return small_graphs, medium_graphs, large_graphs, xlarge_graphs

small_pool, medium_pool, large_pool, xlarge_pool = load_and_categorize_graphs(GRAPH_DIR)

# FIXED: Verify we have enough graphs in ALL buckets, not just small
effective_sample_per_size = min(
    SAMPLE_PER_SIZE,
    len(small_pool),
    len(medium_pool),
    len(large_pool),
    len(xlarge_pool)
)

if effective_sample_per_size < SAMPLE_PER_SIZE:
    print(f"\n⚠️  Warning: Adjusting sample size from {SAMPLE_PER_SIZE} to {effective_sample_per_size}")
    print(f"   (Limited by smallest bucket)")
    SAMPLE_PER_SIZE = effective_sample_per_size

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
all_run_files = []
run_metadata = []

for run_num in range(1, N_RUNS + 1):
    print(f"\n{'='*80}")
    print(f"RUN {run_num}/{N_RUNS}")
    print(f"{'='*80}\n")
    
    # CRITICAL: Use different seed for each run to ensure different samples
    run_seed = BASE_SEED * 1000 + run_num * 137  # Large prime multiplier for better randomization
    random.seed(run_seed)
    np.random.seed(run_seed)
    
    print(f"Run seed: {run_seed}")
    
    # Sample graphs - DIFFERENT for each run
    sampled_small = random.sample(small_pool, min(SAMPLE_PER_SIZE, len(small_pool)))
    sampled_medium = random.sample(medium_pool, min(SAMPLE_PER_SIZE, len(medium_pool)))
    sampled_large = random.sample(large_pool, min(SAMPLE_PER_SIZE, len(large_pool)))
    sampled_xlarge = random.sample(xlarge_pool, min(SAMPLE_PER_SIZE, len(xlarge_pool)))
    
    sampled_graphs = sampled_small + sampled_medium + sampled_large + sampled_xlarge
    
    print(f"\nSampled graphs for Run {run_num}:")
    print(f"  Small:   {[name for name, _, _ in sampled_small[:3]]}... ({len(sampled_small)} total)")
    print(f"  Medium:  {[name for name, _, _ in sampled_medium[:3]]}... ({len(sampled_medium)} total)")
    print(f"  Large:   {[name for name, _, _ in sampled_large[:3]]}... ({len(sampled_large)} total)")
    print(f"  XLarge:  {[name for name, _, _ in sampled_xlarge[:3]]}... ({len(sampled_xlarge)} total)")
    print(f"  Total:   {len(sampled_graphs)} graphs\n")
    
    # Store metadata
    run_metadata.append({
        'run': run_num,
        'seed': run_seed,
        'n_graphs': len(sampled_graphs),
        'small_graphs': [name for name, _, _ in sampled_small],
        'medium_graphs': [name for name, _, _ in sampled_medium],
        'large_graphs': [name for name, _, _ in sampled_large],
        'xlarge_graphs': [name for name, _, _ in sampled_xlarge]
    })
    
    # Create graph list file for this run
    graph_list_file = os.path.join(TEMP_DIR, f"graph_list_run{run_num}.py")
    
    with open(graph_list_file, 'w') as f:
        f.write(f"# Auto-generated graph list for run {run_num}\n")
        f.write(f"# Seed: {run_seed}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("topo = {\n")
        for name, path, n in sampled_graphs:
            basename = os.path.basename(path)
            # Determine size bucket
            if n <= SIZE_SMALL_MAX:
                size = "Small"
            elif SIZE_MEDIUM_MIN <= n <= SIZE_MEDIUM_MAX:
                size = "Medium"
            elif SIZE_LARGE_MIN <= n <= SIZE_LARGE_MAX:
                size = "Large"
            else:
                size = "XLarge"
            f.write(f'    "{name}": "{basename}",  # {n} nodes ({size})\n')
        f.write("}\n")
    
    print(f"Created graph list file: {graph_list_file}")
    
    # Define output file paths
    metrics_file = os.path.join(OUTPUT_DIR, f"results_run{run_num}_metrics.csv")
    edges_all_file = os.path.join(OUTPUT_DIR, f"results_run{run_num}_evaluation_attempts.csv")
    edges_added_file = os.path.join(OUTPUT_DIR, f"results_run{run_num}_evaluation_successful.csv")
    checkpoint_file = os.path.join(OUTPUT_DIR, f"checkpoint_run{run_num}.csv")
    
    # FIXED: Run experiment with proper environment variables (no wrapper script needed)
    print(f"\n{'='*80}")
    print(f"EXECUTING EXPERIMENT {run_num}/{N_RUNS}")
    print(f"{'='*80}\n")
    
    # Build environment dict
    env_vars = os.environ.copy()
    env_vars.update({
        'OUTPUT_METRICS_FILE': metrics_file,
        'OUTPUT_EDGES_ALL_FILE': edges_all_file,
        'OUTPUT_EDGES_ADDED_FILE': edges_added_file,
        'CHECKPOINT_FILE': checkpoint_file,
        'RUN_SEED': str(run_seed),
        'RUN_NUMBER': str(run_num),
        'TOTAL_RUNS': str(N_RUNS),
    })
    
    # Run experiment directly with subprocess
    result = subprocess.run(
        ['python3', 'thesis_experiments_final_script.py', '--graph-list', graph_list_file],
        env=env_vars,
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\n Warning: Run {run_num} exited with code {result.returncode}")
    else:
        print(f"\n✓ Run {run_num} completed successfully")
    
    all_run_files.append(metrics_file)

# Save run metadata
metadata_file = os.path.join(OUTPUT_DIR, "run_metadata.csv")
pd.DataFrame(run_metadata).to_csv(metadata_file, index=False)
print(f"\n✓ Saved run metadata to: {metadata_file}")

print(f"\n{'='*80}")
print(f"ALL RUNS COMPLETE - AGGREGATING RESULTS")
print(f"{'='*80}\n")

# ============================================================================
# LOAD AND COMBINE ALL RESULTS
# ============================================================================
all_dfs = []

for run_num in range(1, N_RUNS + 1):
    metrics_file = os.path.join(OUTPUT_DIR, f"results_run{run_num}_metrics.csv")
    
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        df['Run'] = run_num
        
        # Add size bucket column
        df['SizeBucket'] = df['N'].apply(lambda n: 
            'Small' if n <= SIZE_SMALL_MAX 
            else 'Medium' if SIZE_MEDIUM_MIN <= n <= SIZE_MEDIUM_MAX 
            else 'Large' if SIZE_LARGE_MIN <= n <= SIZE_LARGE_MAX
            else 'XLarge'
        )
        
        all_dfs.append(df)
        print(f"Loaded Run {run_num}: {len(df)} graphs")
    else:
        print(f"Warning: {metrics_file} not found")

if not all_dfs:
    print("Error: No result files found!")
    exit(1)

# Combine all runs
df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal graphs across all runs: {len(df_all)}")

# Save combined results
combined_file = os.path.join(OUTPUT_DIR, "results_all_runs_combined.csv")
df_all.to_csv(combined_file, index=False)
print(f"Combined results saved to: {combined_file}")

# ============================================================================
# COMPUTE AGGREGATED STATISTICS WITH CONFIDENCE INTERVALS
# ============================================================================
print(f"\n{'='*80}")
print(f"COMPUTING STATISTICS WITH 95% CONFIDENCE INTERVALS")
print(f"{'='*80}\n")

# FIXED: Include ALL metrics from exact_metrics()
metrics = [
    # Connectivity metrics
    "λ₂", "AvgNodeConn", "EdgeConn", "SpectralGap",
    # Robustness metrics
    "GCC_5%", "GCC_10%", "AttackCurveAUC", "RobustnessCoeff",
    # Distance metrics
    "ASPL", "Diameter", "Efficiency", "ASPLVariance",
    # Structure metrics
    "ArticulationPoints", "Bridges", "BetCentralization",
    # Spectral metrics
    "NatConnectivity", "EffResistance", "λ₂_λₙ_Ratio",
    # Topology metrics
    "Assortativity", "AvgClustering", "Transitivity",
]

# FIXED: Metric directions for correct improvement calculation
METRIC_DIRECTIONS = {
    # Higher is better
    'λ₂': 'higher',
    'AvgNodeConn': 'higher',
    'EdgeConn': 'higher',
    'SpectralGap': 'higher',
    'GCC_5%': 'higher',
    'GCC_10%': 'higher',
    'AttackCurveAUC': 'higher',
    'RobustnessCoeff': 'higher',
    'Efficiency': 'higher',
    'NatConnectivity': 'higher',
    'λ₂_λₙ_Ratio': 'higher',
    'AvgClustering': 'higher',
    'Transitivity': 'higher',
    
    # Lower is better
    'ASPL': 'lower',
    'Diameter': 'lower',
    'ASPLVariance': 'lower',
    'ArticulationPoints': 'lower',
    'Bridges': 'lower',
    'BetCentralization': 'lower',
    'EffResistance': 'lower',
    
    # Special: negative is better
    'Assortativity': 'negative',
}

def compute_ci_95(data):
    """Compute 95% confidence interval using t-distribution"""
    if len(data) < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
    return ci[0], ci[1]

# Aggregate by (Run, SizeBucket) first
print("Computing run-level statistics...")
run_level_stats = []

for run_num in range(1, N_RUNS + 1):
    df_run = df_all[df_all['Run'] == run_num]
    
    for size_bucket in ["Small", "Medium", "Large", "XLarge"]:
        df_size = df_run[df_run['SizeBucket'] == size_bucket]
        
        if len(df_size) == 0:
            continue
        
        # Compute mean for this run and size bucket
        row = {'Run': run_num, 'SizeBucket': size_bucket}
        
        for metric in metrics:
            orig_col = f"Orig_{metric}"
            
            if orig_col in df_size.columns:
                row[f"Orig_{metric}"] = df_size[orig_col].mean()
            
            # All reward types
            for reward in ["PBR", "EFFRES", "IVI", "NNSI", "GREEDY_DEG", "GREEDY_BET", "GREEDY_ALG"]:
                reward_col = f"{reward}_{metric}"
                if reward_col in df_size.columns:
                    row[reward_col] = df_size[reward_col].mean()
        
        run_level_stats.append(row)

df_run_stats = pd.DataFrame(run_level_stats)
print(f"Computed statistics for {len(df_run_stats)} (run, size) combinations")

# Now compute cross-run statistics with confidence intervals
print("\nComputing cross-run statistics with confidence intervals...")

aggregated_stats = []

for size_bucket in ["Small", "Medium", "Large", "XLarge"]:
    df_size = df_run_stats[df_run_stats["SizeBucket"] == size_bucket]
    
    if len(df_size) == 0:
        print(f"⚠️  No data for {size_bucket} bucket")
        continue
    
    print(f"\n{size_bucket} bucket: {len(df_size)} runs")
    
    reward_types = ["PBR", "EFFRES", "IVI", "NNSI", "GREEDY_DEG", "GREEDY_BET", "GREEDY_ALG"]
    
    for reward_type in reward_types:
        for metric in metrics:
            orig_col = f"Orig_{metric}"
            reward_col = f"{reward_type}_{metric}"
            
            if reward_col not in df_size.columns or orig_col not in df_size.columns:
                continue
            
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
            
            # FIXED: Compute improvement statistics with correct semantics
            if len(orig_values) > 0 and len(orig_values) == len(reward_values):
                direction = METRIC_DIRECTIONS.get(metric, 'higher')
                
                if direction == 'higher':
                    # Higher is better: positive % = improvement
                    improvements = ((reward_values.values - orig_values.values) / (np.abs(orig_values.values) + 1e-8) * 100)
                elif direction == 'lower':
                    # Lower is better: positive % = improvement (flip sign)
                    improvements = ((orig_values.values - reward_values.values) / (np.abs(orig_values.values) + 1e-8) * 100)
                elif direction == 'negative':
                    # More negative is better: positive % = more negative
                    improvements = ((orig_values.values - reward_values.values) / (np.abs(orig_values.values) + 1e-8) * 100)
                
                improvement_type = f"Percentage ({direction} is better)"
                
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
aggregated_file = os.path.join(OUTPUT_DIR, "results_aggregated_by_size.csv")
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

pivot_mean_file = os.path.join(OUTPUT_DIR, "results_summary_mean_by_size.csv")
pivot_mean.to_csv(pivot_mean_file, index=False)
print(f"✓ Mean summary: {pivot_mean_file}")

# Pivot table for confidence intervals
pivot_ci = df_aggregated.pivot_table(
    index=["SizeBucket", "Metric"],
    columns="Reward",
    values=["CI_95_Low", "CI_95_High"],
    aggfunc='first'
).reset_index()

pivot_ci_file = os.path.join(OUTPUT_DIR, "results_summary_ci95_by_size.csv")
pivot_ci.to_csv(pivot_ci_file, index=False)
print(f"✓ 95% CI summary: {pivot_ci_file}")

# Pivot table for improvements
pivot_improvement = df_aggregated.pivot_table(
    index=["SizeBucket", "Metric"],
    columns="Reward",
    values="Mean_Improvement"
).reset_index()

pivot_improvement_file = os.path.join(OUTPUT_DIR, "results_summary_improvement_by_size.csv")
pivot_improvement.to_csv(pivot_improvement_file, index=False)
print(f"✓ Improvement summary: {pivot_improvement_file}")

# Pivot table for improvement confidence intervals
pivot_imp_ci = df_aggregated.pivot_table(
    index=["SizeBucket", "Metric"],
    columns="Reward",
    values=["CI_95_Improvement_Low", "CI_95_Improvement_High"],
    aggfunc='first'
).reset_index()

pivot_imp_ci_file = os.path.join(OUTPUT_DIR, "results_summary_improvement_ci95_by_size.csv")
pivot_imp_ci.to_csv(pivot_imp_ci_file, index=False)
print(f"✓ Improvement CI summary: {pivot_imp_ci_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*80}")
print(f"\nConfiguration:")
print(f"  Total runs: {N_RUNS}")
print(f"  Graphs per run: ~{len(sampled_graphs)}")
print(f"  Total experiments: {len(df_all)}")
print(f"  Output folder: {BASE_FOLDER}/")

print(f"\n{'='*80}")
print(f"OUTPUT FILES")
print(f"{'='*80}")
print(f"  Combined results:     {combined_file}")
print(f"  Aggregated stats:     {aggregated_file}")
print(f"  Mean summary:         {pivot_mean_file}")
print(f"  95% CI summary:       {pivot_ci_file}")
print(f"  Improvement summary:  {pivot_improvement_file}")
print(f"  Improvement CI:       {pivot_imp_ci_file}")
print(f"  Run metadata:         {metadata_file}")

print(f"\n{'='*80}")
print(f"VERIFICATION: Check for variation across runs")
print(f"{'='*80}")

# Quick check for variation
for size in ["Small", "Medium", "Large", "XLarge"]:
    subset = df_aggregated[df_aggregated["SizeBucket"] == size]
    if len(subset) > 0:
        avg_std = subset["Std"].mean()
        print(f"  {size:8s}: Average Std = {avg_std:.4f}")

print(f"\n{'='*80}\n")
