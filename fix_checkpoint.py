#!/usr/bin/env python3
"""
Fix corrupted checkpoint files
"""

import os
import pandas as pd
import shutil
from datetime import datetime

print("=" * 80)
print("CHECKPOINT REPAIR UTILITY")
print("=" * 80)

# File paths
checkpoint_file = "./checkpoint_progress.csv"
output_metrics_file = "./results_network_metrics.csv"
output_edges_all_file = "./results_edge_attempts_all.csv"
output_edges_added_file = "./results_edge_attempts_successful.csv"
output_landscape_file = "./results_reward_landscape.csv"

files_to_check = [
    checkpoint_file,
    output_metrics_file,
    output_edges_all_file,
    output_edges_added_file,
    output_landscape_file
]

print("\nChecking files for corruption...\n")

corrupted_files = []
valid_files = []

for filepath in files_to_check:
    if not os.path.exists(filepath):
        print(f"[ SKIP ] {filepath} - Does not exist")
        continue
    
    try:
        # Try to read the file
        df = pd.read_csv(filepath)
        
        # Check if it has data
        if len(df) == 0:
            print(f"[ WARN ] {filepath} - Empty (0 rows)")
            corrupted_files.append(filepath)
        elif len(df.columns) == 0:
            print(f"[ FAIL ] {filepath} - No columns")
            corrupted_files.append(filepath)
        else:
            print(f"[  OK  ] {filepath} - {len(df)} rows, {len(df.columns)} columns")
            valid_files.append(filepath)
            
    except pd.errors.EmptyDataError:
        print(f"[ FAIL ] {filepath} - Empty data error")
        corrupted_files.append(filepath)
    except Exception as e:
        print(f"[ FAIL ] {filepath} - Error: {e}")
        corrupted_files.append(filepath)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Valid files: {len(valid_files)}")
print(f"Corrupted files: {len(corrupted_files)}")

if corrupted_files:
    print("\nCorrupted files found:")
    for f in corrupted_files:
        print(f"  - {f}")
    
    print("\n" + "=" * 80)
    print("REPAIR OPTIONS")
    print("=" * 80)
    
    response = input("\nDo you want to backup and remove corrupted files? (yes/no): ").strip().lower()
    
    if response == 'yes':
        # Create backup directory
        backup_dir = f"./backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        print(f"\nCreating backup in: {backup_dir}")
        
        for filepath in corrupted_files:
            if os.path.exists(filepath):
                backup_path = os.path.join(backup_dir, os.path.basename(filepath))
                shutil.copy2(filepath, backup_path)
                print(f"  Backed up: {filepath} -> {backup_path}")
                
                # Remove corrupted file
                os.remove(filepath)
                print(f"  Removed: {filepath}")
        
        print("\n" + "=" * 80)
        print("REPAIR COMPLETE")
        print("=" * 80)
        print("\nCorrupted files have been backed up and removed.")
        print("The script will recreate them on next run.")
        print("\nTo restore from backup if needed:")
        print(f"  cp {backup_dir}/* .")
        
    else:
        print("\nNo changes made. To manually fix:")
        print("1. Backup corrupted files")
        print("2. Delete them")
        print("3. Re-run the experiment script")
else:
    print("\nAll files are valid. No repair needed.")

print("\n" + "=" * 80)
