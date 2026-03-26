#!/usr/bin/env python3
"""
Apply runtime optimizations to thesis_experiments_final_script.py
Creates an optimized version with faster execution.
"""

import sys

print("Applying runtime optimizations...")
print("="*80)

# Read original script
with open('thesis_experiments_final_script.py', 'r') as f:
    script = f.read()

# Track changes
changes = []

# 1. Reduce attack curve resolution (20 → 10 points)
old_attack = '''    if attack_fractions is None:
        attack_fractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                           0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]'''

new_attack = '''    if attack_fractions is None:
        # Optimized: 10 points instead of 20 for 2× speedup
        attack_fractions = [0.02, 0.04, 0.06, 0.08, 0.10,
                           0.12, 0.14, 0.16, 0.18, 0.20]'''

if old_attack in script:
    script = script.replace(old_attack, new_attack)
    changes.append("✓ Reduced attack curve resolution (20 → 10 points)")
else:
    print("⚠️  Could not find attack curve code to optimize")

# 2. Skip greedy λ₂ for large graphs
old_greedy = '''    # Greedy Algebraic Connectivity
    print(f"    [3/3] Greedy λ₂...", end=" ", flush=True)
    g_alg, _ = greedy_algebraic_connectivity_baseline(orig_g, B)
    alg_met = exact_metrics(g_alg)
    for k, v in alg_met.items():
        row[f"GREEDY_ALG_{k}"] = v
    print(f"✓")'''

new_greedy = '''    # Greedy Algebraic Connectivity (skip for large graphs - optimization)
    if n <= 50:  # Only for small/medium graphs
        print(f"    [3/3] Greedy λ₂...", end=" ", flush=True)
        g_alg, _ = greedy_algebraic_connectivity_baseline(orig_g, B)
        alg_met = exact_metrics(g_alg)
        for k, v in alg_met.items():
            row[f"GREEDY_ALG_{k}"] = v
        print(f"✓")
    else:
        print(f"    [3/3] Greedy λ₂... SKIPPED (large graph, n={n})")
        for k in metrics:
            row[f"GREEDY_ALG_{k}"] = np.nan'''

if old_greedy in script:
    script = script.replace(old_greedy, new_greedy)
    changes.append("✓ Skip greedy λ₂ for large graphs (n > 50)")
else:
    print("⚠️  Could not find greedy λ₂ code to optimize")

# 3. Reduce training steps (optional - commented out by default)
old_training = '''            print("25k steps...", end=" ", flush=True)
            model.learn(total_timesteps=25000)'''

new_training = '''            # Optimized: 15k steps instead of 25k (1.7× speedup)
            # Change back to 25000 if you want full quality
            print("15k steps...", end=" ", flush=True)
            model.learn(total_timesteps=15000)'''

# Uncomment to apply training reduction
# if old_training in script:
#     script = script.replace(old_training, new_training)
#     changes.append("✓ Reduced training steps (25k → 15k)")

# Write optimized script
output_file = 'thesis_experiments_final_script_optimized.py'
with open(output_file, 'w') as f:
    f.write(script)

print("\nChanges applied:")
for change in changes:
    print(f"  {change}")

print(f"\n{'='*80}")
print(f"Optimized script saved to: {output_file}")
print(f"{'='*80}")

print("\nExpected speedup:")
print("  Attack curve: 2× faster")
print("  Greedy λ₂: 3× faster (skipped for large graphs)")
print("  Overall: ~2-3× faster per graph")
print()
print("To use optimized version:")
print(f"  1. Review {output_file}")
print(f"  2. Backup original: cp thesis_experiments_final_script.py thesis_experiments_final_script_backup.py")
print(f"  3. Replace: cp {output_file} thesis_experiments_final_script.py")
print(f"  4. Run: python run_repeated_experiments_parallel.py")
print()
print("Combined with parallel execution (10 cores):")
print("  Original: 80 hours")
print("  Parallel only: 8 hours")
print("  Parallel + optimizations: 3-4 hours")
print(f"{'='*80}\n")
