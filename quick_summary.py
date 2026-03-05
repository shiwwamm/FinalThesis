#!/usr/bin/env python3
"""
Quick Summary of Thesis Results
Prints key findings to console
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("QUICK SUMMARY OF THESIS RESULTS")
print("="*80 + "\n")

# Load data
try:
    metrics = pd.read_csv("results_network_metrics.csv")
    print(f"✓ Loaded results for {len(metrics)} networks\n")
except FileNotFoundError:
    print("✗ results_network_metrics.csv not found!")
    print("  Run thesis_experiments_final_script.py first\n")
    exit(1)

rewards = ["PBR", "EFFRES", "IVI", "NNSI"]
key_metrics = ["λ₂", "AvgNodeConn", "GCC_5%", "EffResistance"]

# Overall performance
print("-"*80)
print("OVERALL PERFORMANCE (Mean % Improvement)")
print("-"*80)

for metric in key_metrics:
    print(f"\n{metric}:")
    higher_better = metric in ["λ₂", "AvgNodeConn", "GCC_5%"]
    
    results = {}
    for reward in rewards:
        col = f"%Δ_{reward}_vs_Orig_{metric}"
        if col in metrics.columns:
            mean_val = metrics[col].mean()
            results[reward] = mean_val
            
    # Sort by performance
    if higher_better:
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_results = sorted(results.items(), key=lambda x: abs(x[1]))
    
    for rank, (reward, value) in enumerate(sorted_results, 1):
        marker = "★" if rank == 1 else " "
        print(f"  {marker} {rank}. {reward:10s}: {value:+7.2f}%")

# Network size breakdown
print("\n" + "-"*80)
print("PERFORMANCE BY NETWORK SIZE")
print("-"*80)

metrics['Size'] = pd.cut(metrics['N'], bins=[0, 40, 93, 300],
                         labels=['Small', 'Medium', 'Large'])

for size in ['Small', 'Medium', 'Large']:
    subset = metrics[metrics['Size'] == size]
    print(f"\n{size} Networks (n={len(subset)}):")
    
    # Average across key metrics
    scores = {}
    for reward in rewards:
        reward_scores = []
        for metric in key_metrics:
            col = f"%Δ_{reward}_vs_Orig_{metric}"
            if col in subset.columns:
                higher_better = metric in ["λ₂", "AvgNodeConn", "GCC_5%"]
                mean_val = subset[col].mean()
                reward_scores.append(mean_val if higher_better else -mean_val)
        scores[reward] = np.mean(reward_scores) if reward_scores else 0
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (reward, score) in enumerate(sorted_scores, 1):
        marker = "★" if rank == 1 else " "
        print(f"  {marker} {rank}. {reward:10s}: {score:+7.2f}")

# Budget usage
print("\n" + "-"*80)
print("BUDGET STATISTICS")
print("-"*80)
print(f"  Mean budget: {metrics['BudgetEdges'].mean():.1f} edges")
print(f"  Min budget:  {metrics['BudgetEdges'].min():.0f} edges")
print(f"  Max budget:  {metrics['BudgetEdges'].max():.0f} edges")

# Network characteristics
print("\n" + "-"*80)
print("NETWORK CHARACTERISTICS")
print("-"*80)
print(f"  Nodes (N):  {metrics['N'].min():.0f} - {metrics['N'].max():.0f} "
      f"(mean: {metrics['N'].mean():.1f})")
print(f"  Edges (M):  {metrics['M'].min():.0f} - {metrics['M'].max():.0f} "
      f"(mean: {metrics['M'].mean():.1f})")

# Top improvements
print("\n" + "-"*80)
print("TOP 5 IMPROVEMENTS (λ₂)")
print("-"*80)

for reward in rewards:
    col = f"%Δ_{reward}_vs_Orig_λ₂"
    if col in metrics.columns:
        top5 = metrics.nlargest(5, col)[['Graph', col]]
        print(f"\n{reward}:")
        for idx, row in top5.iterrows():
            print(f"  {row['Graph']:20s}: {row[col]:+7.2f}%")

print("\n" + "="*80)
print("For detailed analysis, run: python analyze_thesis_results.py")
print("="*80 + "\n")
