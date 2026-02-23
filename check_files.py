#!/usr/bin/env python3
"""
Check current state of checkpoint and result files
"""

import os
import pandas as pd

print("=" * 80)
print("FILE STATUS CHECK")
print("=" * 80)

# Old file names (from error)
old_files = {
    "test_thesis_4rewards_30networks_metrics.csv": "Old metrics file",
    "test_thesis_4rewards_30networks_edges_all_attempts.csv": "Old edges file",
    "test_thesis_4rewards_30networks_edges_added_only.csv": "Old added edges file",
    "thesis_4rewards_30networks_reward_landscape.csv": "Old landscape file",
}

# New file names
new_files = {
    "results_network_metrics.csv": "New metrics file",
    "results_edge_attempts_all.csv": "New edges file",
    "results_edge_attempts_successful.csv": "New added edges file",
    "results_reward_landscape.csv": "New landscape file",
}

checkpoint_file = "checkpoint_progress.csv"

print("\n1. CHECKPOINT FILE:")
print("-" * 80)
if os.path.exists(checkpoint_file):
    size = os.path.getsize(checkpoint_file)
    print(f"   {checkpoint_file}: {size} bytes")
    try:
        df = pd.read_csv(checkpoint_file)
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"   Column names: {list(df.columns)}")
        print(f"   Networks: {len(df)}")
    except Exception as e:
        print(f"   ERROR reading: {e}")
else:
    print(f"   {checkpoint_file}: NOT FOUND")

print("\n2. OLD FILE NAMES (should not exist):")
print("-" * 80)
old_exist = False
for filename, desc in old_files.items():
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"   {filename}: {size} bytes - {desc}")
        old_exist = True
        try:
            df = pd.read_csv(filename)
            print(f"      Rows: {len(df)}, Columns: {len(df.columns)}")
        except Exception as e:
            print(f"      ERROR: {e}")
    else:
        print(f"   {filename}: not found (good)")

if old_exist:
    print("\n   WARNING: Old files exist! You may be running an old version of the script.")

print("\n3. NEW FILE NAMES (should exist):")
print("-" * 80)
for filename, desc in new_files.items():
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"   {filename}: {size} bytes - {desc}")
        try:
            df = pd.read_csv(filename)
            print(f"      Rows: {len(df)}, Columns: {len(df.columns)}")
            if len(df.columns) > 0:
                print(f"      Column names: {list(df.columns)[:5]}...")  # First 5 columns
        except Exception as e:
            print(f"      ERROR: {e}")
    else:
        print(f"   {filename}: not found")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if old_exist:
    print("\nPROBLEM: You have old file names but are running the new script!")
    print("\nSOLUTION:")
    print("  Option 1: Rename old files to new names")
    print("    mv test_thesis_4rewards_30networks_metrics.csv results_network_metrics.csv")
    print("    mv test_thesis_4rewards_30networks_edges_all_attempts.csv results_edge_attempts_all.csv")
    print("    mv test_thesis_4rewards_30networks_edges_added_only.csv results_edge_attempts_successful.csv")
    print("    mv thesis_4rewards_30networks_reward_landscape.csv results_reward_landscape.csv")
    print("\n  Option 2: Delete old files and continue from checkpoint")
    print("    rm test_thesis_4rewards_30networks_*.csv")
    print("    rm thesis_4rewards_30networks_*.csv")
    print("    python thesis_experiments_final_script.py")
else:
    print("\nFile names look correct.")
    print("Check if any files are corrupted (0 bytes or error reading).")

print("\n" + "=" * 80)
