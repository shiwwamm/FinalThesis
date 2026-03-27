#!/usr/bin/env python3
"""
Wrapper script to run repeated experiments with random sampling.
Calls the original thesis_experiments_final_script.py multiple times with different graph samples.

Usage:
    python run_repeated_experiments.py [folder_name]
    
    If folder_name is provided, creates:
        <folder_name>/temp/    - Temporary files (graph lists, wrapper scripts)
        <folder_name>/output/  - All result files
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

# ============================================================================
# CONFIGURATION
# ============================================================================
N_RUNS = 1
SAMPLE_PER_SIZE = 5
GRAPH_DIR = "./real_world_topologies"
SEED = 42

# Size buckets (Jenks Natural Breaks)
SIZE_SMALL_MAX = 40
SIZE_MEDIUM_MIN = 41
SIZE_MEDIUM_MAX = 93
SIZE_LARGE_MIN = 94

# Parse command line arguments
if len(sys.argv) > 1:
    BASE_FOLDER = sys.argv[1]
    TEMP_DIR = os.path.join(BASE_FOLDER, "temp")
    OUTPUT_DIR = os.path.join(BASE_FOLDER, "output")
    
    # Create directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"ORGANIZING OUTPUT INTO FOLDER: {BASE_FOLDER}")
    print(f"{'='*80}")
    print(f"  Temporary files: {TEMP_DIR}/")
    print(f"  Output files: {OUTPUT_DIR}/")
else:
    BASE_FOLDER = "."
    TEMP_DIR = "."
    OUTPUT_DIR = "."

print(f"\n{'='*80}")
print(f"REPEATED EXPERIMENTS WRAPPER")
print(f"{'='*80}")
print(f"Configuration:")
print(f"  Runs: {N_RUNS}")
print(f"  Samples per size: {SAMPLE_PER_SIZE}")
print(f"  Total graphs per run: {SAMPLE_PER_SIZE * 3}")
print(f"  Base seed: {SEED}")
if BASE_FOLDER != ".":
    print(f"  Output folder: {BASE_FOLDER}/")
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
            elif n >= SIZE_LARGE_MIN:
                large_graphs.append((name, path, n))
        except Exception as e:
            print(f"  Warning: Could not load {path}: {e}")
    
    print(f"\nGraph distribution:")
    print(f"  Small (≤{SIZE_SMALL_MAX}): {len(small_graphs)} graphs")
    print(f"  Medium ({SIZE_MEDIUM_MIN}-{SIZE_MEDIUM_MAX}): {len(medium_graphs)} graphs")
    print(f"  Large (≥{SIZE_LARGE_MIN}): {len(large_graphs)} graphs")
    print(f"  Total: {len(small_graphs) + len(medium_graphs) + len(large_graphs)} graphs\n")
    
    return small_graphs, medium_graphs, large_graphs

small_pool, medium_pool, large_pool = load_and_categorize_graphs(GRAPH_DIR)

# ============================================================================
# CREATE TEMPORARY GRAPH LIST FILES FOR EACH RUN
# ============================================================================
def create_graph_list_file(run_num, sampled_graphs):
    """Create a Python file with the graph list for this run"""
    filename = f"graph_list_run{run_num}.py"
    
    with open(filename, 'w') as f:
        f.write("# Auto-generated graph list for run {}\n".format(run_num))
        f.write("topo = {\n")
        for name, path, n in sampled_graphs:
            basename = os.path.basename(path)
            f.write(f'    "{name}": "{basename}",  # {n} nodes\n')
        f.write("}\n")
    
    return filename

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
all_run_files = []

for run_num in range(1, N_RUNS + 1):
    print(f"\n{'='*80}")
    print(f"PREPARING RUN {run_num}/{N_RUNS}")
    print(f"{'='*80}\n")
    
    # Set seed for this run
    run_seed = SEED + run_num
    random.seed(run_seed)
    np.random.seed(run_seed)
    
    # Sample graphs
    sampled_small = random.sample(small_pool, min(SAMPLE_PER_SIZE, len(small_pool)))
    sampled_medium = random.sample(medium_pool, min(SAMPLE_PER_SIZE, len(medium_pool)))
    sampled_large = random.sample(large_pool, min(SAMPLE_PER_SIZE, len(large_pool)))
    
    sampled_graphs = sampled_small + sampled_medium + sampled_large
    
    print(f"Sampled graphs for Run {run_num}:")
    print(f"  Small: {[name for name, _, _ in sampled_small]}")
    print(f"  Medium: {[name for name, _, _ in sampled_medium]}")
    print(f"  Large: {[name for name, _, _ in sampled_large]}")
    print(f"  Total: {len(sampled_graphs)} graphs\n")
    
    # Create graph list file for this run
    graph_list_file = os.path.join(TEMP_DIR, f"graph_list_run{run_num}.py")
    
    with open(graph_list_file, 'w') as f:
        f.write(f"# Auto-generated graph list for run {run_num}\n")
        f.write("# This file is imported by thesis_experiments_final_script.py\n\n")
        f.write("topo = {\n")
        for name, path, n in sampled_graphs:
            basename = os.path.basename(path)
            # Determine size bucket
            if n <= SIZE_SMALL_MAX:
                size = "Small"
            elif SIZE_MEDIUM_MIN <= n <= SIZE_MEDIUM_MAX:
                size = "Medium"
            else:
                size = "Large"
            f.write(f'    "{name}": "{basename}",  # {n} nodes ({size})\n')
        f.write("}\n")
    
    print(f"Created graph list file: {graph_list_file}")
    
    # Create a wrapper script that sets environment variables and runs the experiment
    wrapper_script = os.path.join(TEMP_DIR, f"run_experiment_{run_num}.py")
    
    # Define output file paths
    metrics_file = os.path.join(OUTPUT_DIR, f"results_run{run_num}_metrics.csv")
    edges_all_file = os.path.join(OUTPUT_DIR, f"results_run{run_num}_edges_all.csv")
    edges_added_file = os.path.join(OUTPUT_DIR, f"results_run{run_num}_edges_added.csv")
    checkpoint_file = os.path.join(OUTPUT_DIR, f"checkpoint_run{run_num}.csv")
    
    with open(wrapper_script, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
# Wrapper script for run {run_num}
import os
import sys

# Set environment variables for output files
os.environ['OUTPUT_METRICS_FILE'] = '{metrics_file}'
os.environ['OUTPUT_EDGES_ALL_FILE'] = '{edges_all_file}'
os.environ['OUTPUT_EDGES_ADDED_FILE'] = '{edges_added_file}'
os.environ['CHECKPOINT_FILE'] = '{checkpoint_file}'

# Update sys.argv to pass graph list argument
sys.argv = ['thesis_experiments_final_script.py', '--graph-list', '{graph_list_file}']

# Execute the original script
with open('thesis_experiments_final_script.py', 'r') as script_file:
    exec(script_file.read())
""")
    
    print(f"Created wrapper script: {wrapper_script}")
    
    # Run the experiment
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT {run_num}/{N_RUNS}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(['python', wrapper_script], capture_output=False)
    
    if result.returncode != 0:
        print(f"\n⚠️  Warning: Run {run_num} exited with code {result.returncode}")
    else:
        print(f"\n✓ Run {run_num} completed successfully")
    
    all_run_files.append(metrics_file)
    
    # Clean up temporary script
    # os.remove(script_name)  # Keep for debugging

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
            else 'Large'
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
# COMPUTE AGGREGATED STATISTICS BY SIZE BUCKET
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

# Step 1: Compute mean per (Run, SizeBucket) for each reward and metric
print("Step 1: Computing run-level means per size bucket...")
run_level_means = df_all.groupby(["Run", "SizeBucket"]).mean(numeric_only=True).reset_index()

print(f"  Run-level means computed: {len(run_level_means)} rows")
print(f"  Runs per size bucket: {run_level_means.groupby('SizeBucket')['Run'].nunique().to_dict()}")

# Step 2: Compute statistics across the run means
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
            
            # Get run-level mean values (one per run)
            orig_values = df_size[orig_col].dropna()
            reward_values = df_size[reward_col].dropna()
            
            if len(reward_values) == 0:
                continue
            
            # Compute statistics across run means
            mean_val = reward_values.mean()
            min_val = reward_values.min()
            max_val = reward_values.max()
            std_val = reward_values.std()
            ci_low, ci_high = compute_ci_95(reward_values)
            
            # Compute improvement statistics
            if len(orig_values) > 0 and len(orig_values) == len(reward_values):
                # Use absolute change for metrics that can be 0
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

pivot_ci_file = os.path.join(OUTPUT_DIR, "results_summary_ci95_by_size.csv")
pivot_ci.to_csv(pivot_ci_file, index=False)
print(f"✓ 95% CI summary saved to: {pivot_ci_file}")

# Pivot table for mean improvements
pivot_improvement = df_aggregated.pivot_table(
    index=["SizeBucket", "Metric"],
    columns="Reward",
    values="Mean_Improvement"
).reset_index()

pivot_improvement_file = os.path.join(OUTPUT_DIR, "results_summary_improvement_by_size.csv")
pivot_improvement.to_csv(pivot_improvement_file, index=False)
print(f"✓ Improvement summary saved to: {pivot_improvement_file}")

# ============================================================================
# DISPLAY SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*80}")
print(f"\nTotal runs: {N_RUNS}")
print(f"Total graphs processed: {len(df_all)}")
print(f"Total experiments: {len(df_all) * 4}")

print(f"\n{'='*80}")
print(f"OUTPUT FILES")
print(f"{'='*80}")
print(f"Combined results: {combined_file}")
print(f"Aggregated stats: {aggregated_file}")
print(f"Mean summary: {pivot_mean_file}")
print(f"95% CI summary: {pivot_ci_file}")
print(f"Improvement summary: {pivot_improvement_file}")
print(f"\nPer-run files:")
for run_num in range(1, N_RUNS + 1):
    print(f"  Run {run_num}: {os.path.join(OUTPUT_DIR, f'results_run{run_num}_metrics.csv')}")

if BASE_FOLDER != ".":
    print(f"\nAll files organized in: {BASE_FOLDER}/")
    print(f"  Temporary files: {TEMP_DIR}/")
    print(f"  Output files: {OUTPUT_DIR}/")

# Display sample results
print(f"\n{'='*80}")
print(f"SAMPLE AGGREGATED RESULTS")
print(f"{'='*80}\n")

# Show key metrics for each size bucket
key_metrics = ["λ₂", "AvgNodeConn", "GCC_5%", "EffResistance", "NatConnectivity"]

for metric in key_metrics:
    print(f"\n{metric}:")
    sample = df_aggregated[df_aggregated["Metric"] == metric][
        ["SizeBucket", "Reward", "Mean", "CI_95_Low", "CI_95_High", "Mean_Improvement_%"]
    ].round(4)
    
    if len(sample) > 0:
        print(sample.to_string(index=False))

print(f"\n{'='*80}")
print(f"DONE")
print(f"{'='*80}\n")
