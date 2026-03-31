#!/usr/bin/env python3
"""
Diagnostic script to identify the cause of segmentation faults.
Run this to check your VM environment before running the full experiment.
"""

import os
import sys

# Set thread limits FIRST
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

print("="*80)
print("SEGFAULT DIAGNOSTIC TOOL")
print("="*80)
print()

# Check 1: System info
print("1. System Information:")
print(f"   Python version: {sys.version}")
print(f"   Platform: {sys.platform}")
import platform
print(f"   Machine: {platform.machine()}")
print(f"   Processor: {platform.processor()}")
print()

# Check 2: Memory
print("2. Memory Information:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   Total RAM: {mem.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {mem.available / (1024**3):.1f} GB")
    print(f"   Used RAM: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
except ImportError:
    print("   Install psutil for memory info: pip install psutil")
print()

# Check 3: Import libraries
print("3. Testing Library Imports:")
libs = [
    ("numpy", "np"),
    ("pandas", "pd"),
    ("igraph", "ig"),
    ("torch", "torch"),
    ("torch_geometric", "pyg"),
    ("stable_baselines3", "sb3"),
    ("sb3_contrib", "sb3c"),
]

for lib_name, alias in libs:
    try:
        if lib_name == "torch_geometric":
            import torch_geometric
            print(f"   ✓ {lib_name}")
        elif lib_name == "stable_baselines3":
            import stable_baselines3
            print(f"   ✓ {lib_name}")
        elif lib_name == "sb3_contrib":
            import sb3_contrib
            print(f"   ✓ {lib_name}")
        else:
            exec(f"import {lib_name} as {alias}")
            print(f"   ✓ {lib_name}")
    except Exception as e:
        print(f"   ✗ {lib_name}: {e}")
print()

# Check 4: Thread settings
print("4. Thread Settings:")
import torch
print(f"   PyTorch threads: {torch.get_num_threads()}")
print(f"   PyTorch interop threads: {torch.get_num_interop_threads()}")
print(f"   OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"   MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
print()

# Check 5: NumPy BLAS backend
print("5. NumPy BLAS Backend:")
import numpy as np
try:
    config = np.__config__.show()
    print("   See output above for BLAS/LAPACK info")
except:
    print("   Could not determine BLAS backend")
print()

# Check 6: Test small training
print("6. Testing Small PPO Training:")
print("   Creating simple environment...")
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    
    env = gym.make("CartPole-v1")
    print("   ✓ Environment created")
    
    print("   Creating PPO model...")
    model = PPO("MlpPolicy", env, verbose=0, n_steps=128, batch_size=32)
    print("   ✓ Model created")
    
    print("   Training for 1000 steps...")
    model.learn(total_timesteps=1000)
    print("   ✓ Training completed successfully!")
    
    del model
    del env
    import gc
    gc.collect()
    
except Exception as e:
    print(f"   ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
print()

# Check 7: Test graph loading
print("7. Testing Graph Loading:")
try:
    import igraph as ig
    import glob
    
    graph_files = glob.glob("./real_world_topologies/*.graphml")[:5]
    print(f"   Found {len(graph_files)} test graphs")
    
    for gf in graph_files:
        g = ig.Graph.Read_GraphML(gf).as_undirected()
        name = os.path.basename(gf).split(".")[0]
        print(f"   ✓ Loaded {name}: N={g.vcount()}, M={g.ecount()}")
    
except Exception as e:
    print(f"   ✗ Graph loading failed: {e}")
print()

# Check 8: Test PyTorch Geometric
print("8. Testing PyTorch Geometric:")
try:
    import torch
    import torch_geometric.nn as pyg_nn
    
    # Create simple GNN
    gnn = pyg_nn.Sequential("x, edge_index", [
        (pyg_nn.GraphConv(5, 32), "x, edge_index -> x"),
        torch.nn.ReLU(),
        (pyg_nn.GraphConv(32, 32), "x, edge_index -> x"),
    ])
    
    # Test forward pass
    x = torch.randn(10, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    out = gnn(x, edge_index)
    
    print(f"   ✓ GNN forward pass successful: {out.shape}")
    
    del gnn, x, edge_index, out
    import gc
    gc.collect()
    
except Exception as e:
    print(f"   ✗ PyG test failed: {e}")
    import traceback
    traceback.print_exc()
print()

print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print()
print("If all tests passed, the segfault may be caused by:")
print("  1. Specific graph properties (try smaller graphs first)")
print("  2. Interaction between libraries during longer training")
print("  3. Memory fragmentation over time")
print()
print("Next steps:")
print("  - If test 6 (PPO training) failed: Library installation issue")
print("  - If test 8 (PyG) failed: PyTorch Geometric installation issue")
print("  - If all passed: Try running with a single small graph first")
print()
