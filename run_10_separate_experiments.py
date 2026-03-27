#!/usr/bin/env python3
"""
Run 10 separate experiments, each with N_RUNS=1, into different folders.
This allows running experiments independently and combining results later.

Usage:
    python run_10_separate_experiments.py [base_folder_name]
    
Creates:
    <base_folder_name>_run1/
    <base_folder_name>_run2/
    ...
    <base_folder_name>_run10/
"""

import os
import sys
import subprocess
import time

# Configuration
N_EXPERIMENTS = 10
BASE_NAME = sys.argv[1] if len(sys.argv) > 1 else "experiment"

print(f"\n{'='*80}")
print(f"RUNNING 10 SEPARATE EXPERIMENTS")
print(f"{'='*80}")
print(f"Base folder name: {BASE_NAME}")
print(f"Will create: {BASE_NAME}_run1 through {BASE_NAME}_run10")
print(f"Each experiment: 1 run × 15 networks")
print(f"{'='*80}\n")

# Run experiments
start_time = time.time()
successful_runs = []
failed_runs = []

for i in range(1, N_EXPERIMENTS + 1):
    folder_name = f"{BASE_NAME}_run{i}"
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {i}/{N_EXPERIMENTS}: {folder_name}")
    print(f"{'='*80}\n")
    
    exp_start = time.time()
    
    # Run the experiment
    result = subprocess.run(['python3', 'run_repeated_experiments.py', folder_name], 
                          capture_output=False)
    
    exp_time = time.time() - exp_start
    
    if result.returncode == 0:
        print(f"\n✓ Experiment {i} completed successfully in {exp_time/60:.1f} minutes")
        successful_runs.append((i, folder_name, exp_time))
    else:
        print(f"\n❌ Experiment {i} failed with return code {result.returncode}")
        failed_runs.append((i, folder_name))

total_time = time.time() - start_time

# Summary
print(f"\n{'='*80}")
print(f"ALL EXPERIMENTS COMPLETE")
print(f"{'='*80}")
print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
print(f"Successful: {len(successful_runs)}/{N_EXPERIMENTS}")
if failed_runs:
    print(f"Failed: {len(failed_runs)}")
    for i, folder in failed_runs:
        print(f"  - Experiment {i}: {folder}")

if successful_runs:
    avg_time = sum(t for _, _, t in successful_runs) / len(successful_runs)
    print(f"Average time per experiment: {avg_time/60:.1f} minutes")

print(f"\n{'='*80}")
print(f"FOLDER STRUCTURE")
print(f"{'='*80}")
for i, folder, _ in successful_runs:
    print(f"{folder}/")
    print(f"  ├── temp/")
    print(f"  │   ├── graph_list_run1.py")
    print(f"  │   └── run_experiment_1.py")
    print(f"  └── output/")
    print(f"      ├── results_run1_metrics.csv")
    print(f"      ├── results_run1_edges_all.csv")
    print(f"      ├── results_run1_edges_added.csv")
    print(f"      └── results_aggregated_by_size.csv")
    if i < len(successful_runs):
        print()

print(f"\n{'='*80}")
print(f"NEXT STEPS")
print(f"{'='*80}")
print(f"\nTo combine all results into one aggregated file:")
print(f"  python3 combine_separate_experiments.py {BASE_NAME}")
print(f"\nTo visualize individual experiments:")
for i, folder, _ in successful_runs[:3]:  # Show first 3 as examples
    print(f"  python3 visualize_aggregated_results.py {folder}")
print(f"  ...")

print(f"\n{'='*80}\n")
