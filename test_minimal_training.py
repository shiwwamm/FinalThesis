#!/usr/bin/env python3
"""
Minimal test to isolate the segfault cause.
Tests each component incrementally.
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
import igraph as ig
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("="*80)
print("MINIMAL TRAINING TEST")
print("="*80)
print()

# Test 1: Load a small graph
print("Test 1: Loading graph...")
g = ig.Graph.Read_GraphML("./real_world_topologies/Arpanet19706.graphml").as_undirected()
print(f"  ✓ Loaded graph: N={g.vcount()}, M={g.ecount()}")
print()

# Test 2: Create simple environment without GNN
print("Test 2: Testing MaskablePPO with simple MLP policy...")
try:
    env = gym.make("CartPole-v1")
    model = MaskablePPO("MlpPolicy", env, verbose=0, n_steps=128, batch_size=32)
    print("  Training 1000 steps...")
    model.learn(total_timesteps=1000)
    print("  ✓ MaskablePPO training works")
    del model, env
except Exception as e:
    print(f"  ✗ Failed: {e}")
print()

# Test 3: Create simple GNN
print("Test 3: Testing GNN forward pass...")
try:
    class SimpleGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = pyg_nn.GraphConv(5, 32)
            self.conv2 = pyg_nn.GraphConv(32, 32)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return x
    
    gnn = SimpleGNN()
    x = torch.randn(10, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out = gnn(x, edge_index)
    print(f"  ✓ GNN forward pass works: {out.shape}")
    del gnn, x, edge_index, out
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 4: Create GNN feature extractor
print("Test 4: Testing GNN feature extractor...")
try:
    class TestGNNExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=64):
            super().__init__(observation_space, features_dim)
            self.conv1 = pyg_nn.GraphConv(5, 32)
            self.conv2 = pyg_nn.GraphConv(32, 32)
            self.proj = nn.Linear(32, features_dim)
            
        def forward(self, obs):
            # Simple test: just return projection of mean
            return self.proj(torch.randn(obs.shape[0], 32))
    
    # Create dummy environment
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
    extractor = TestGNNExtractor(obs_space, features_dim=64)
    test_obs = torch.randn(4, 50)
    out = extractor(test_obs)
    print(f"  ✓ GNN extractor works: {out.shape}")
    del extractor, test_obs, out
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 5: MaskablePPO with custom feature extractor
print("Test 5: Testing MaskablePPO with custom feature extractor...")
try:
    class SimpleExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=64):
            super().__init__(observation_space, features_dim)
            self.net = nn.Sequential(
                nn.Linear(observation_space.shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, features_dim)
            )
            
        def forward(self, obs):
            return self.net(obs)
    
    env = gym.make("CartPole-v1")
    policy_kwargs = dict(
        features_extractor_class=SimpleExtractor,
        features_extractor_kwargs=dict(features_dim=64),
    )
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, 
                        verbose=0, n_steps=128, batch_size=32)
    print("  Training 1000 steps...")
    model.learn(total_timesteps=1000)
    print("  ✓ MaskablePPO with custom extractor works")
    del model, env
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
print()

print("="*80)
print("MINIMAL TEST COMPLETE")
print("="*80)
print()
print("If all tests passed, the issue is likely in:")
print("  1. The ResilienceEnv implementation")
print("  2. The specific GNN architecture with graph batching")
print("  3. The interaction between environment and GNN during rollouts")
print()
