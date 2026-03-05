#!/usr/bin/env python3
"""
Complete Analysis Script for Thesis Results
Analyzes output from thesis_experiments_final_script.py

Input files:
  - results_network_metrics.csv
  - results_edge_attempts_all.csv
  - results_edge_attempts_successful.csv

Outputs:
  - Summary statistics tables
  - Comparison figures
  - Statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
os.makedirs("analysis_output", exist_ok=True)

print("="*80)
print("THESIS RESULTS ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")

try:
    metrics = pd.read_csv("results_network_metrics.csv")
    print(f"   ✓ Loaded metrics: {len(metrics)} networks")
except FileNotFoundError:
    print("   ✗ results_network_metrics.csv not found!")
    exit(1)

try:
    attempts_all = pd.read_csv("results_edge_attempts_all.csv")
    print(f"   ✓ Loaded all attempts: {len(attempts_all)} records")
except FileNotFoundError:
    print("   ✗ results_edge_attempts_all.csv not found!")
    attempts_all = None

try:
    attempts_success = pd.read_csv("results_edge_attempts_successful.csv")
    print(f"   ✓ Loaded successful attempts: {len(attempts_success)} records")
except FileNotFoundError:
    print("   ✗ results_edge_attempts_successful.csv not found!")
    attempts_success = None

# ============================================================================
# 2. BASIC STATISTICS
# ============================================================================
print("\n2. Computing basic statistics...")

rewards = ["PBR", "EFFRES", "IVI", "NNSI"]
key_metrics = ["λ₂", "AvgNodeConn", "GCC_5%", "EffResistance", "ASPL", 
               "NatConnectivity", "BetCentralization"]

# Summary table
summary_data = []
for reward in rewards:
    for metric in key_metrics:
        col = f"%Δ_{reward}_vs_Orig_{metric}"
        if col in metrics.columns:
            values = metrics[col].dropna()
            summary_data.append({
                "Reward": reward,
                "Metric": metric,
                "Mean_%Δ": values.mean(),
                "Median_%Δ": values.median(),
                "Std_%Δ": values.std(),
                "Min_%Δ": values.min(),
                "Max_%Δ": values.max(),
                "N": len(values)
            })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("analysis_output/summary_statistics.csv", index=False)
print(f"   ✓ Saved: analysis_output/summary_statistics.csv")

# ============================================================================
# 3. REWARD FUNCTION COMPARISON
# ============================================================================
print("\n3. Comparing reward functions...")

# For each metric, which reward function performed best?
comparison_data = []
for metric in key_metrics:
    metric_results = {}
    for reward in rewards:
        col = f"%Δ_{reward}_vs_Orig_{metric}"
        if col in metrics.columns:
            metric_results[reward] = metrics[col].mean()
    
    if metric_results:
        # Determine if higher is better
        higher_better = metric in ["λ₂", "AvgNodeConn", "GCC_5%", "NatConnectivity"]
        
        if higher_better:
            best_reward = max(metric_results, key=metric_results.get)
            best_value = metric_results[best_reward]
        else:
            best_reward = min(metric_results, key=lambda k: abs(metric_results[k]))
            best_value = metric_results[best_reward]
        
        comparison_data.append({
            "Metric": metric,
            "Higher_Better": higher_better,
            "Best_Reward": best_reward,
            "Best_Mean_%Δ": best_value,
            **{f"{r}_Mean_%Δ": metric_results.get(r, np.nan) for r in rewards}
        })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("analysis_output/reward_comparison.csv", index=False)
print(f"   ✓ Saved: analysis_output/reward_comparison.csv")

# ============================================================================
# 4. NETWORK SIZE ANALYSIS
# ============================================================================
print("\n4. Analyzing by network size...")

# Add size categories
metrics['N_Category'] = pd.cut(metrics['N'], 
                                bins=[0, 40, 93, 300],
                                labels=['Small (≤40)', 'Medium (41-93)', 'Large (>93)'])

size_analysis = []
for category in ['Small (≤40)', 'Medium (41-93)', 'Large (>93)']:
    subset = metrics[metrics['N_Category'] == category]
    for reward in rewards:
        for metric in key_metrics:
            col = f"%Δ_{reward}_vs_Orig_{metric}"
            if col in subset.columns:
                values = subset[col].dropna()
                if len(values) > 0:
                    size_analysis.append({
                        "Size_Category": category,
                        "Reward": reward,
                        "Metric": metric,
                        "Mean_%Δ": values.mean(),
                        "N_Networks": len(values)
                    })

size_df = pd.DataFrame(size_analysis)
size_df.to_csv("analysis_output/size_analysis.csv", index=False)
print(f"   ✓ Saved: analysis_output/size_analysis.csv")

# ============================================================================
# 5. STATISTICAL TESTS
# ============================================================================
print("\n5. Running statistical tests...")

# Pairwise comparisons between reward functions
stat_tests = []
for metric in key_metrics:
    # Get data for each reward
    data_by_reward = {}
    for reward in rewards:
        col = f"%Δ_{reward}_vs_Orig_{metric}"
        if col in metrics.columns:
            data_by_reward[reward] = metrics[col].dropna().values
    
    # Pairwise t-tests
    reward_list = list(data_by_reward.keys())
    for i in range(len(reward_list)):
        for j in range(i+1, len(reward_list)):
            r1, r2 = reward_list[i], reward_list[j]
            if len(data_by_reward[r1]) > 1 and len(data_by_reward[r2]) > 1:
                t_stat, p_value = stats.ttest_rel(data_by_reward[r1], data_by_reward[r2])
                stat_tests.append({
                    "Metric": metric,
                    "Reward_1": r1,
                    "Reward_2": r2,
                    "Mean_Diff": data_by_reward[r1].mean() - data_by_reward[r2].mean(),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "Significant_0.05": p_value < 0.05,
                    "Significant_0.01": p_value < 0.01
                })

stat_df = pd.DataFrame(stat_tests)
stat_df.to_csv("analysis_output/statistical_tests.csv", index=False)
print(f"   ✓ Saved: analysis_output/statistical_tests.csv")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n6. Creating visualizations...")

# 6.1 Mean improvement by reward function
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, metric in enumerate(key_metrics):
    ax = axes[idx]
    means = []
    errors = []
    
    for reward in rewards:
        col = f"%Δ_{reward}_vs_Orig_{metric}"
        if col in metrics.columns:
            values = metrics[col].dropna()
            means.append(values.mean())
            errors.append(1.96 * values.std() / np.sqrt(len(values)))  # 95% CI
        else:
            means.append(0)
            errors.append(0)
    
    x = np.arange(len(rewards))
    ax.bar(x, means, yerr=errors, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(rewards, rotation=45)
    ax.set_title(metric)
    ax.set_ylabel('Mean %Δ')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis_output/mean_improvement_by_reward.png", dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: analysis_output/mean_improvement_by_reward.png")
plt.close()

# 6.2 Heatmap of improvements
pivot_data = summary_df.pivot(index='Metric', columns='Reward', values='Mean_%Δ')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
            cbar_kws={'label': 'Mean %Δ'})
plt.title('Mean Improvement by Reward Function and Metric')
plt.tight_layout()
plt.savefig("analysis_output/improvement_heatmap.png", dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: analysis_output/improvement_heatmap.png")
plt.close()

# 6.3 Distribution plots for key metrics
for metric in ["λ₂", "AvgNodeConn", "GCC_5%", "EffResistance"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = []
    labels = []
    for reward in rewards:
        col = f"%Δ_{reward}_vs_Orig_{metric}"
        if col in metrics.columns:
            data_to_plot.append(metrics[col].dropna())
            labels.append(reward)
    
    if data_to_plot:
        ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), 
                      showmeans=True, showmedians=True)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(f'%Δ {metric}')
        ax.set_title(f'Distribution of {metric} Improvements')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"analysis_output/distribution_{metric.replace('₂', '2')}.png", 
                    dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: analysis_output/distribution_{metric.replace('₂', '2')}.png")
        plt.close()

# 6.4 Network size effect
if 'N_Category' in metrics.columns:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(["λ₂", "AvgNodeConn", "GCC_5%"]):
        ax = axes[idx]
        
        for reward in rewards:
            col = f"%Δ_{reward}_vs_Orig_{metric}"
            if col in metrics.columns:
                means_by_size = []
                categories = ['Small (≤40)', 'Medium (41-93)', 'Large (>93)']
                
                for cat in categories:
                    subset = metrics[metrics['N_Category'] == cat]
                    means_by_size.append(subset[col].mean())
                
                ax.plot(categories, means_by_size, marker='o', label=reward, linewidth=2)
        
        ax.set_xlabel('Network Size')
        ax.set_ylabel(f'Mean %Δ {metric}')
        ax.set_title(f'{metric} by Network Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("analysis_output/size_effect.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: analysis_output/size_effect.png")
    plt.close()

# ============================================================================
# 7. EDGE ADDITION ANALYSIS (if available)
# ============================================================================
if attempts_success is not None:
    print("\n7. Analyzing edge additions...")
    
    # Average reward per step
    step_rewards = attempts_success.groupby(['Reward', 'Step'])['StepReward'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    for reward in rewards:
        subset = step_rewards[step_rewards['Reward'] == reward]
        plt.plot(subset['Step'], subset['StepReward'], marker='o', label=reward, linewidth=2)
    
    plt.xlabel('Step')
    plt.ylabel('Average Step Reward')
    plt.title('Average Reward per Step by Reward Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("analysis_output/reward_per_step.png", dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: analysis_output/reward_per_step.png")
    plt.close()

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================
print("\n8. Generating summary report...")

with open("analysis_output/ANALYSIS_REPORT.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("THESIS RESULTS ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total Networks Analyzed: {len(metrics)}\n")
    f.write(f"Reward Functions: {', '.join(rewards)}\n")
    f.write(f"Key Metrics: {', '.join(key_metrics)}\n\n")
    
    f.write("-"*80 + "\n")
    f.write("BEST PERFORMING REWARD FUNCTION PER METRIC\n")
    f.write("-"*80 + "\n")
    for _, row in comparison_df.iterrows():
        f.write(f"{row['Metric']:20s} -> {row['Best_Reward']:10s} "
                f"(Mean %Δ: {row['Best_Mean_%Δ']:+.2f}%)\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("OVERALL PERFORMANCE RANKING\n")
    f.write("-"*80 + "\n")
    
    # Calculate overall score (average across all metrics)
    overall_scores = {}
    for reward in rewards:
        scores = []
        for metric in key_metrics:
            col = f"%Δ_{reward}_vs_Orig_{metric}"
            if col in metrics.columns:
                # Normalize: higher is better for some metrics, lower for others
                higher_better = metric in ["λ₂", "AvgNodeConn", "GCC_5%", "NatConnectivity"]
                mean_val = metrics[col].mean()
                scores.append(mean_val if higher_better else -mean_val)
        overall_scores[reward] = np.mean(scores) if scores else 0
    
    ranked = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (reward, score) in enumerate(ranked, 1):
        f.write(f"{rank}. {reward:10s} (Overall Score: {score:+.2f})\n")
    
    f.write("\n" + "-"*80 + "\n")
    f.write("SIGNIFICANT DIFFERENCES (p < 0.05)\n")
    f.write("-"*80 + "\n")
    sig_tests = stat_df[stat_df['Significant_0.05'] == True]
    f.write(f"Found {len(sig_tests)} significant pairwise differences\n\n")
    for _, row in sig_tests.head(10).iterrows():
        f.write(f"{row['Metric']:20s}: {row['Reward_1']} vs {row['Reward_2']} "
                f"(p={row['p_value']:.4f}, diff={row['Mean_Diff']:+.2f}%)\n")
    
    if len(sig_tests) > 10:
        f.write(f"... and {len(sig_tests)-10} more (see statistical_tests.csv)\n")

print(f"   ✓ Saved: analysis_output/ANALYSIS_REPORT.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nOutput files in 'analysis_output/' directory:")
print("  - summary_statistics.csv       : Basic stats for all metrics")
print("  - reward_comparison.csv        : Best reward per metric")
print("  - size_analysis.csv            : Performance by network size")
print("  - statistical_tests.csv        : Pairwise statistical tests")
print("  - ANALYSIS_REPORT.txt          : Human-readable summary")
print("  - *.png                        : Visualization figures")
print("\n")
