#!/usr/bin/env python3
"""
Test the ResilienceEnv specifically to isolate segfault.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"

import sys
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

# Verify thread limits are set by environment variables
print(f"Thread check: PyTorch threads={torch.get_num_threads()}, interop={torch.get_num_interop_threads()}")
print()

# Import the environment from the main script
sys.path.insert(0, '.')
from thesis_experiments_final_script import ResilienceEnv, CleanGNNExtractor

print("="*80)
print("RESILIENCE ENV TEST")
print("="*80)
print()

# Test 1: Create environment
print("Test 1: Creating ResilienceEnv...")
try:
    env = ResilienceEnv(
        "./real_world_topologies/Arpanet19706.graphml",
        reward_type="pbr",
        budget_edges=3,
        gamma=5.0,
        beta=1.0,
        delta=1.0,
        shortlist_size=32,  # Small shortlist
        max_candidate_pool=100,
    )
    print(f"  ✓ Environment created")
    print(f"    Observation space: {env.observation_space.shape}")
    print(f"    Action space: {env.action_space.n}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 2: Reset environment
print("Test 2: Resetting environment...")
try:
    obs, info = env.reset()
    print(f"  ✓ Reset successful")
    print(f"    Observation shape: {obs.shape}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 3: Get action masks
print("Test 3: Getting action masks...")
try:
    masks = get_action_masks(env)
    print(f"  ✓ Action masks retrieved")
    print(f"    Mask shape: {masks.shape}")
    print(f"    Valid actions: {masks.sum()}/{len(masks)}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 4: Take a step
print("Test 4: Taking environment step...")
try:
    valid_actions = np.where(masks)[0]
    if len(valid_actions) > 0:
        action = valid_actions[0]
        obs, reward, done, truncated, info = env.step(action)
        print(f"  ✓ Step successful")
        print(f"    Action: {action}")
        print(f"    Reward: {reward:.4f}")
        print(f"    Done: {done}")
    else:
        print(f"  ⚠️  No valid actions available")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 5: Create GNN extractor
print("Test 5: Creating GNN feature extractor...")
try:
    extractor = CleanGNNExtractor(env.observation_space, env, features_dim=128)
    print(f"  ✓ GNN extractor created")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 6: Test GNN forward pass with real observation
print("Test 6: Testing GNN forward pass with environment observation...")
try:
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    features = extractor(obs_tensor)
    print(f"  ✓ GNN forward pass successful")
    print(f"    Features shape: {features.shape}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 7: Create MaskablePPO model
print("Test 7: Creating MaskablePPO model...")
try:
    policy_kwargs = dict(
        features_extractor_class=CleanGNNExtractor,
        features_extractor_kwargs=dict(env=env),
        net_arch=[128, 128],
    )
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=64,  # Very small
        batch_size=32,  # Very small
        n_epochs=2,  # Very small
        gamma=0.99,
        device="cpu",
        verbose=0,
        seed=42,
    )
    print(f"  ✓ Model created successfully")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 8: Train for very few steps
print("Test 8: Training for 500 steps (this is where segfault likely occurs)...")
try:
    model.learn(total_timesteps=500)
    print(f"  ✓ Training completed successfully!")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

print("="*80)
print("ALL TESTS PASSED!")
print("="*80)
print()
print("The environment and GNN work correctly.")
print("If segfault still occurs with 50k steps, it may be:")
print("  1. Memory accumulation over many steps")
print("  2. Specific edge case in environment logic")
print("  3. Issue with longer training runs")
print()
