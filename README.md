# Network Resilience Experiments

Reinforcement learning experiments that add edges to real-world network topologies to improve resilience metrics. Agents are trained with **Maskable PPO** and a GNN feature extractor, comparing four reward functions:

- **PBR** — Path Betweenness Reward  
- **EFFRES** — Effective Resistance  
- **IVI** — Information Vulnerability Index  
- **NNSI** — Network Node Significance Index  

## Repository layout

```
.
├── thesis_experiments_final_script.py   # Core training & evaluation loop
├── run_repeated_experiments.py          # Sample graphs by size and run one experiment
├── run_10_separate_experiments.py       # Launch 10 independent experiment folders
├── combine_separate_experiments.py      # Aggregate the 10 runs + CIs
├── visualize_aggregated_results.py      # Plots from aggregated CSVs
├── real_world_topologies/               # Topology GraphML files
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `torch-geometric` fails to install, follow the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your PyTorch / CUDA (or CPU) build.

## How to run

### Full pipeline (recommended)

Run 10 separate experiments, then combine and plot:

```bash
# 1) 10 independent runs → experiment_run1/ … experiment_run10/
python3 run_10_separate_experiments.py experiment

# 2) Merge into experiment_combined/ with means and 95% CIs
python3 combine_separate_experiments.py experiment

# 3) Visualize a single run folder (or the combined folder)
python3 visualize_aggregated_results.py experiment_run1
# or
python3 visualize_aggregated_results.py experiment_combined
```

### Single experiment

Samples graphs from `real_world_topologies/` (small / medium / large by Jenks size buckets) and runs the core script once:

```bash
python3 run_repeated_experiments.py my_run
```

Outputs go to `my_run/temp/` and `my_run/output/`.

### Core script only

```bash
python3 thesis_experiments_final_script.py
# or with an explicit graph list:
python3 thesis_experiments_final_script.py --graph-list path/to/graph_list.py
```

## Outputs

Typical CSVs written under each run’s `output/` folder:

| File | Contents |
|------|----------|
| `results_run*_metrics.csv` | Per-network metrics for each reward |
| `results_run*_edges_all.csv` | Edge-addition attempts |
| `results_run*_edges_added.csv` | Successfully added edges |
| `results_aggregated_by_size.csv` | Aggregated by size bucket |

After combining, `*_combined/` also includes mean and 95% CI summaries by size.

## Notes

- Topology files are GraphML (`.graphml`) under `real_world_topologies/`.
- Size buckets: small ≤ 40, medium 41–93, large ≥ 94 nodes.
- Thread env vars in the core script limit BLAS/Torch oversubscription (useful on VMs / shared hosts).
