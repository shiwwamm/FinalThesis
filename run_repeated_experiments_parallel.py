#!/usr/bin/env python3
"""
Parallel version of repeated experiments wrapper.
Runs multiple experiments simultaneously using multiprocessing.
"""

import os
import glob
import random
import subprocess
import pandas as pd
import numpy as np
from scipy import stats
import igraph as ig
from multiprocessing import Pool, cpu_count
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
N_RUNS = 10
SAMPLE_PER_SIZE = 10
GRAPH_DIR = "./real_world_topologies"
SEED = 42

# Parallel execution settings
N_PARALLEL_RUNS = min(cpu_count(), 10)  # Run up to 10 experiments in parallel
print(f"Available CPUs: {cpu_count()}")
print(f"Will run {N_PARALLEL_RUNS} experiments in parallel")

# Size buckets (Jenks Natural Breaks)
SIZE_SMALL_MAX = 40
SIZE_MEDIUM_MIN = 41
SIZE_MEDIUM_MAX = 93
SIZE_LARGE_MIN = 94

print(f"\n{'='*80}")
print(f"REPEATED EXPERIMENTS WRAPPER (PARALLEL)")
print(f"{'='*80}")
print(f"Configuration:")
print(f"  Runs: {N_RUNS}")
print(f"  Parallel runs: {N_PARALLEL_RUNS}")
print(f"  Samples per size: {SAMPLE_PER_SIZE}")
print(f"  Total graphs per run: {SAMPLE_PER_SIZE * 3}")
print(f"  Base seed: {SEED}")
print(f"  Expected speedup: ~{N_PARALLEL_RUNS}×")
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
# FUNCTION TO RUN A SINGLE EXPERIMENT
# ============================================================================
def run_single_experiment(run_num):
    """Run a single experiment (to be called in parallel)"""
    print(f"\n{'='*80}")
    print(f"PREPARING RUN {run_num}/{N_RUNS} (PID: {os.getpid()})")
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
    graph_list_file = f"graph_list_run{run_num}.py"
    
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
    wrapper_script = f"run_experiment_{run_num}.py"
    
    with open(wrapper_script, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
# Wrapper script for run {run_num}
import os
import sys

# Set environment variables for output files
os.environ['OUTPUT_METRICS_FILE'] = './results_run{run_num}_metrics.csv'
os.environ['OUTPUT_EDGES_ALL_FILE'] = './results_run{run_num}_edges_all.csv'
os.environ['OUTPUT_EDGES_ADDED_FILE'] = './results_run{run_num}_edges_added.csv'
os.environ['CHECKPOINT_FILE'] = './checkpoint_run{run_num}.csv'
os.environ['RUN_NUMBER'] = '{run_num}'
os.environ['TOTAL_RUNS'] = '{N_RUNS}'

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
    
    start_time = time.time()
    result = subprocess.run(['python3', wrapper_script], capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n⚠️  Warning: Run {run_num} exited with code {result.returncode}")
        print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars of error
        return (run_num, False, elapsed)
    else:
        print(f"\n✓ Run {run_num} completed successfully in {elapsed/60:.1f} minutes")
        return (run_num, True, elapsed)

# ============================================================================
# RUN EXPERIMENTS IN PARALLEL
# ============================================================================
print(f"\n{'='*80}")
print(f"STARTING PARALLEL EXECUTION")
print(f"{'='*80}\n")

start_time = time.time()

# Create a pool of workers
with Pool(processes=N_PARALLEL_RUNS) as pool:
    # Run experiments in parallel
    results = pool.map(run_single_experiment, range(1, N_RUNS + 1))

total_time = time.time() - start_time

# Check results
successful_runs = [r[0] for r in results if r[1]]
failed_runs = [r[0] for r in results if not r[1]]
run_times = [r[2] for r in results if r[1]]

print(f"\n{'='*80}")
print(f"PARALLEL EXECUTION COMPLETE")
print(f"{'='*80}")
print(f"Total wall time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
print(f"Successful runs: {len(successful_runs)}/{N_RUNS}")
if failed_runs:
    print(f"Failed runs: {failed_runs}")
if run_times:
    print(f"Average run time: {np.mean(run_times)/60:.1f} minutes")
    print(f"Speedup achieved: ~{sum(run_times)/total_time:.1f}×")

if len(successful_runs) < 2:
    print("\n❌ Error: Need at least 2 successful runs for aggregation!")
    exit(1)

print(f"\n{'='*80}")
print(f"AGGREGATING RESULTS")
print(f"{'='*80}\n")

# Continue with aggregation (same as before)
all_dfs = []

for run_num in successful_runs:
    metrics_file = f"./results_run{run_num}_metrics.csv"
    
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
        print(f"⚠️  Warning: {metrics_file} not found")

if not all_dfs:
    print("❌ Error: No result files found!")
    exit(1)

# Combine all runs
df_all = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal graphs across all runs: {len(df_all)}")

# Save combined results
combined_file = "./results_all_runs_combined.csv"
df_all.to_csv(combined_file, index=False)
print(f"Combined results saved to: {combined_file}")

# Rest of aggregation code (same as serial version)
# ... (copy from run_repeated_experiments.py)

print(f"\n{'='*80}")
print(f"ALL DONE!")
print(f"{'='*80}")
print(f"Total time: {total_time/3600:.1f} hours")
print(f"Speedup: ~{N_PARALLEL_RUNS}× (parallel execution)")
print(f"{'='*80}\n")
