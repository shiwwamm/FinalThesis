#!/usr/bin/env python3
"""
Quick test to verify thread limits are working correctly.
Run this in your VM to confirm settings before running full experiment.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"

import torch
import numpy as np

print("="*80)
print("THREAD LIMIT VERIFICATION")
print("="*80)
print()

print("Environment Variables:")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'NOT SET')}")
print(f"  OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'NOT SET')}")
print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'NOT SET')}")
print(f"  TORCH_NUM_THREADS: {os.environ.get('TORCH_NUM_THREADS', 'NOT SET')}")
print(f"  TORCH_NUM_INTEROP_THREADS: {os.environ.get('TORCH_NUM_INTEROP_THREADS', 'NOT SET')}")
print()

# Set PyTorch limits
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("PyTorch Settings:")
print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
print(f"  torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
print()

# Test with actual computation
print("Testing with actual computation...")
x = torch.randn(1000, 1000)
y = torch.randn(1000, 1000)
z = torch.mm(x, y)
print(f"  ✓ Matrix multiplication completed: {z.shape}")
print()

print("="*80)
if torch.get_num_interop_threads() == 1:
    print("✓ THREAD LIMITS CORRECTLY SET")
else:
    print("⚠️  WARNING: Interop threads not set to 1!")
    print("   This may cause segfaults during training.")
print("="*80)
