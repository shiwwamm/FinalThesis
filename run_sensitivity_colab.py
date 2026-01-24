#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab-Ready Sensitivity Analysis
Automatically runs sensitivity tests without command-line arguments
"""

### COLAB SETUP - Uncomment these lines when running in Colab
# !pip install -q torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.9.1+cu128.html
# !pip install -q torch-geometric
# from google.colab import drive
# drive.mount('/content/drive')

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import igraph as ig
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric import nn as pyg_nn

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================
# For Local (default)
TOPOLOGY_DIR = "./real_world_topologies"
OUTPUT_CSV = "./sensitivity_results.csv"
BUDGET_SUMMARY_CSV = "./budget_sweep_summary.csv"

# For Colab (uncomment these and comment above when using Colab)
# TOPOLOGY_DIR = "/content/drive/MyDrive/real_world_topologies"
# OUTPUT_CSV = "/content/drive/MyDrive/sensitivity_results.csv"
# BUDGET_SUMMARY_CSV = "/content/drive/MyDrive/budget_sweep_summary.csv"

# Experiment settings
MAX_NETWORKS = 10  # Number of networks to test
TIMESTEPS = 10000  # Training steps per run (10k for quick tests, 25k for full)
SEEDS = [42, 43, 44]  # Random seeds for reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Budget sensitivity sweep
BUDGET_SWEEP = False  # Set True to test different budget formulas
BUDGET_ALPHAS = [0.10, 0.15, 0.20]  # Node fraction values
BUDGET_BETAS = [0.05, 0.10, 0.15]   # Edge fraction values
BUDGET_CAPS = [10, 20]               # Maximum budget values
ALL_SETTINGS_WITH_BUDGET = False    # If True, tests all reward settings with all budgets (expensive)

print(f"Device: {DEVICE}")
print(f"Networks: {MAX_NETWORKS} | Timesteps: {TIMESTEPS} | Seeds: {len(SEEDS)}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def edge_budget(n: int, m: int, alpha_n: float = 0.15, beta_m: float = 0.10, 
                cap: int = 20, min_edges: int = 2) -> int:
    """Episode edge-addition budget."""
    node_term = int(np.ceil(alpha_n * n))
    edge_term = int(np.ceil(beta_m * m))
    return int(max(min_edges, min(node_term, min(edge_term, cap))))

def exact_metrics(g: ig.Graph) -> Dict[str, float]:
    """Compute evaluation metrics."""
    n = g.vcount()
    if n < 2:
        return {
            "λ₂": 0.0, "AvgNodeConn": 0.0, "GCC_5%": 1.0, "ASPL": 0.0,
            "Diameter": 0.0, "ArticulationPoints": 0, "Bridges": 0,
            "BetCentralization": 0.0, "NatConnectivity": 0.0,
        }

    # λ₂ (normalized Laplacian)
    L = np.array(g.laplacian(normalized=True))
    eigs = np.sort(np.linalg.eigvalsh(L))
    lambda2 = eigs[1] if len(eigs) > 1 else 0.0

    # Average Node Connectivity
    if n <= 50:
        try:
            avg_node_conn = g.vertex_connectivity()
        except:
            avg_node_conn = 0.0
    else:
        sample_size = min(50, n)
        sample_nodes = np.random.choice(n, size=sample_size, replace=False)
        connectivities = []
        for i in range(len(sample_nodes)):
            for j in range(i + 1, min(i + 6, len(sample_nodes))):
                try:
                    conn = g.vertex_connectivity(sample_nodes[i], sample_nodes[j])
                    connectivities.append(conn)
                except:
                    pass
        avg_node_conn = float(np.mean(connectivities)) if connectivities else 0.0

    # GCC after attack
    k = max(1, int(0.05 * n))
    top_deg = np.argsort(-np.array(g.degree()))[:k]
    gc = g.copy()
    gc.delete_vertices(top_deg)
    try:
        gcc = gc.clusters().giant().vcount() / max(1, (n - k))
    except:
        gcc = 0.0

    # ASPL & Diameter
    try:
        aspl = g.average_path_length()
    except:
        aspl = np.inf
    diam = g.diameter() if g.is_connected() else np.inf

    # Articulation points & bridges
    art_pts = len(g.articulation_points())
    bridges = len(g.bridges())

    # Betweenness centralization
    bet = np.array(g.betweenness())
    if n > 2:
        bet_central = (bet.max() * (n - 1)) / ((n - 1) * (n - 2) / 2)
    else:
        bet_central = 0.0

    # Natural connectivity
    A = np.array(g.get_adjacency().data)
    eigA = np.linalg.eigvals(A)
    natconn = float(np.log(np.mean(np.exp(eigA.real)) + 1e-12))

    return {
        "λ₂": float(lambda2), 
        "AvgNodeConn": float(avg_node_conn),
        "GCC_5%": float(gcc),
        "ASPL": float(aspl), 
        "Diameter": float(diam),
        "ArticulationPoints": int(art_pts), 
        "Bridges": int(bridges),
        "BetCentralization": float(bet_central), 
        "NatConnectivity": float(natconn),
    }

# ============================================================================
# GNN FEATURE EXTRACTOR
# ============================================================================
class CleanGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, edge_index, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.n_nodes = observation_space.shape[0] // 5
        self.register_buffer("edge_index", edge_index)

        self.gnn = pyg_nn.Sequential("x, edge_index", [
            (pyg_nn.GraphConv(5, 64), "x, edge_index -> x"),
            nn.ReLU(),
            (pyg_nn.GraphConv(64, 64), "x, edge_index -> x"),
            nn.ReLU(),
            (pyg_nn.GraphConv(64, 32), "x, edge_index -> x"),
        ])
        self.pool = pyg_nn.global_mean_pool
        self.proj = nn.Linear(32, features_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        x = obs.view(b, self.n_nodes, 5).view(-1, 5)
        x = self.gnn(x, self.edge_index)
        batch = torch.arange(b, device=obs.device).repeat_interleave(self.n_nodes)
        x = self.pool(x, batch)
        x = self.proj(x)
        return x

# ============================================================================
# ENVIRONMENT
# ============================================================================
class GraphEdgeAddEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, path: str, reward_type: str = "pbr", budget_edges: Optional[int] = None,
                 budget_alpha_n: float = 0.15, budget_beta_m: float = 0.10,
                 budget_cap: int = 20, budget_min_edges: int = 2,
                 gamma: float = 5.0, beta: float = 1.0, delta: float = 1.0,
                 w_fc: float = 0.5, w_m: float = 0.3, w_lam: float = 0.2):
        super().__init__()
        g = ig.Graph.Read_GraphML(path).as_undirected()
        self.g_orig = g.copy()
        self.g = g.copy()
        self.reward_type = reward_type
        self.n = self.g.vcount()
        self.m = self.g.ecount()
        
        self.budget = int(budget_edges) if budget_edges is not None else edge_budget(
            self.n, self.m, alpha_n=budget_alpha_n, beta_m=budget_beta_m, 
            cap=budget_cap, min_edges=budget_min_edges)

        # Candidate edges
        existing = set(tuple(sorted(e)) for e in self.g.get_edgelist())
        self.candidates = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)
                          if (i, j) not in existing]

        self.action_space = spaces.Discrete(len(self.candidates))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(self.n * 5,), dtype=np.float32)

        # Normalization
        feats = self._compute_node_features(self.g_orig)
        self.mean, self.std = feats.mean(0), feats.std(0) + 1e-8

        # Edge index
        edges = g.get_edgelist()
        ei = torch.tensor(edges, dtype=torch.long).t()
        self.edge_index_base = torch.cat([ei, ei.flip(0)], dim=1)

        # Params
        self.gamma, self.beta, self.delta = float(gamma), float(beta), float(delta)
        self.w_fc, self.w_m, self.w_lam = float(w_fc), float(w_m), float(w_lam)

        self.added_edges = set()
        self.successful_additions = 0
        self.steps = 0
        self.reset()

    def _compute_node_features(self, g: ig.Graph) -> np.ndarray:
        n = g.vcount()
        deg = np.array(g.degree(), dtype=float)
        try:
            close = np.array(g.closeness(), dtype=float)
        except:
            close = np.zeros(n, dtype=float)
        pr = np.array(g.pagerank(), dtype=float)
        core = np.array(g.coreness(), dtype=float)
        clust = np.array(g.transitivity_local_undirected(mode="zero"), dtype=float)
        return np.vstack([deg, close, pr, core, clust]).T

    def _approx_lambda2(self, g: ig.Graph) -> float:
        deg = np.array(g.degree(), dtype=float)
        mu1 = deg.mean()
        mu2 = (deg ** 2).mean()
        var = max(0.0, mu2 - mu1 ** 2)
        return float(max(0.0, mu1 - np.sqrt(var)))

    def _effective_graph_resistance(self, g: ig.Graph) -> float:
        if not g.is_connected():
            return 1e9
        L = np.array(g.laplacian())
        eigs = np.sort(np.linalg.eigvalsh(L))
        eigs = eigs[1:]
        return float(g.vcount() * np.sum(1.0 / (eigs + 1e-12)))

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.g = self.g_orig.copy()
        self.steps = 0
        self.added_edges = set()
        self.successful_additions = 0
        self.prev_lambda2 = self._approx_lambda2(self.g)

        if self.reward_type == "pbr":
            deg = np.array(self.g.degree(), dtype=float)
            k_mean = deg.mean()
            k2_mean = (deg ** 2).mean()
            self.prev_moment = k2_mean / (k_mean + 1e-8)
            self.prev_fc = 1.0 - 1.0 / (self.prev_moment - 1.0 + 1e-8)
        elif self.reward_type == "effres":
            self.prev_effres = float(self._effective_graph_resistance(self.g))

        return self._obs(), {}

    def _obs(self):
        X = self._compute_node_features(self.g)
        Xn = (X - self.mean) / self.std
        return Xn.astype(np.float32).reshape(-1)

    def step(self, action: int):
        u, v = self.candidates[action]
        edge = tuple(sorted((u, v)))

        if edge in self.added_edges or self.g.are_connected(u, v):
            reward = -0.5
        else:
            self.g.add_edge(u, v)
            self.added_edges.add(edge)
            self.successful_additions += 1
            reward = self._compute_reward()

        self.steps += 1
        max_attempts = self.budget * 10
        done = self.successful_additions >= self.budget or self.steps >= max_attempts
        obs = self._obs()
        return obs, float(reward), done, False, {}

    def _compute_reward(self) -> float:
        curr_lambda2 = self._approx_lambda2(self.g)
        d_lambda2 = curr_lambda2 - self.prev_lambda2

        if self.reward_type == "pbr":
            r = self._reward_pbr(d_lambda2)
        elif self.reward_type == "effres":
            r = self._reward_effres(d_lambda2)
        else:
            r = 0.0

        self.prev_lambda2 = curr_lambda2
        return float(max(min(r, 5.0), -5.0))

    def _reward_pbr(self, d_lambda2: float) -> float:
        deg = np.array(self.g.degree(), dtype=float)
        k_mean = deg.mean()
        k2_mean = (deg ** 2).mean()
        curr_moment = k2_mean / (k_mean + 1e-8)
        curr_fc = 1.0 - 1.0 / (curr_moment - 1.0 + 1e-8)
        d_fc = curr_fc - self.prev_fc
        d_moment = self.prev_moment - curr_moment
        r = self.w_fc * d_fc + self.w_m * d_moment + self.w_lam * d_lambda2
        self.prev_fc = curr_fc
        self.prev_moment = curr_moment
        return float(r)

    def _reward_effres(self, d_lambda2: float) -> float:
        curr_effres = self._effective_graph_resistance(self.g)
        prev = float(self.prev_effres) if self.prev_effres is not None else 1e9
        d_effres = (prev - float(curr_effres)) / (abs(prev) + 1e-9)
        lambda_bonus = self.beta * max(0.0, d_lambda2)
        lambda_penalty = self.gamma * max(0.0, -d_lambda2)
        r = self.delta * d_effres + lambda_bonus - lambda_penalty
        self.prev_effres = float(curr_effres)
        return float(r)

# ============================================================================
# SETTINGS
# ============================================================================
@dataclass(frozen=True)
class BudgetSetting:
    name: str
    alpha_n: float = 0.15
    beta_m: float = 0.10
    cap: int = 20
    min_edges: int = 2

@dataclass(frozen=True)
class Setting:
    name: str
    reward_type: str
    gamma: float = 5.0
    beta: float = 1.0
    delta: float = 1.0
    w_fc: float = 0.5
    w_m: float = 0.3
    w_lam: float = 0.2

def build_budget_settings(sweep: bool, alphas: List[float], betas: List[float], 
                         caps: List[int], min_edges: int = 2) -> List[BudgetSetting]:
    if not sweep:
        return [BudgetSetting(name="B_base", alpha_n=0.15, beta_m=0.10, cap=20, min_edges=min_edges)]
    
    out = []
    for a in alphas:
        for b in betas:
            for c in caps:
                out.append(BudgetSetting(
                    name=f"B_a{a:.2f}_b{b:.2f}_c{c}", 
                    alpha_n=a, beta_m=b, cap=c, min_edges=min_edges))
    return out

def build_settings() -> List[Setting]:
    settings = []
    
    # PBR baseline + sensitivity + ablations
    settings.append(Setting(name="PBR_base", reward_type="pbr", w_fc=0.5, w_m=0.3, w_lam=0.2))
    settings += [
        Setting(name="PBR_fc_high", reward_type="pbr", w_fc=0.6, w_m=0.25, w_lam=0.15),
        Setting(name="PBR_fc_low",  reward_type="pbr", w_fc=0.4, w_m=0.35, w_lam=0.25),
        Setting(name="PBR_m_high",  reward_type="pbr", w_fc=0.5, w_m=0.4,  w_lam=0.1),
        Setting(name="PBR_m_low",   reward_type="pbr", w_fc=0.55,w_m=0.2,  w_lam=0.25),
        Setting(name="PBR_lam_high",reward_type="pbr", w_fc=0.45,w_m=0.25, w_lam=0.30),
        Setting(name="PBR_lam_low", reward_type="pbr", w_fc=0.55,w_m=0.35, w_lam=0.10),
        Setting(name="PBR_ablate_m",   reward_type="pbr", w_fc=0.7, w_m=0.0, w_lam=0.3),
        Setting(name="PBR_ablate_lam", reward_type="pbr", w_fc=0.7, w_m=0.3, w_lam=0.0),
        Setting(name="PBR_fc_only",    reward_type="pbr", w_fc=1.0, w_m=0.0, w_lam=0.0),
    ]
    
    # EffRes baseline + gamma sweep + ablation
    settings.append(Setting(name="EFFRES_base", reward_type="effres", gamma=5.0, beta=1.0, delta=1.0))
    for gam in [1.0, 2.0, 5.0, 10.0]:
        settings.append(Setting(name=f"EFFRES_gamma_{int(gam)}", reward_type="effres", 
                               gamma=gam, beta=1.0, delta=1.0))
    settings.append(Setting(name="EFFRES_no_lambda_shaping", reward_type="effres", 
                           gamma=0.0, beta=0.0, delta=1.0))
    
    return settings

# ============================================================================
# NETWORK SELECTION
# ============================================================================
DEFAULT_NETWORKS = [
    "Arpanet19706.graphml", "Abilene.graphml", "Belnet2005.graphml",
    "Cernet.graphml", "Dfn.graphml", "Tw.graphml",
    "Interoute.graphml", "TataNld.graphml", "DialtelecomCz.graphml", "Cogentco.graphml",
]

def resolve_networks(topology_dir: str, max_networks: int) -> List[str]:
    paths = []
    for name in DEFAULT_NETWORKS:
        p = os.path.join(topology_dir, name)
        if os.path.exists(p):
            paths.append(p)
        else:
            if not name.lower().endswith(".graphml"):
                p2 = os.path.join(topology_dir, name + ".graphml")
                if os.path.exists(p2):
                    paths.append(p2)
    
    if not paths:
        raise FileNotFoundError(f"No GraphML files found in {topology_dir}")
    
    return paths[:max_networks]

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================
def train_and_eval_one(graph_path: str, setting: Setting, budget_setting: BudgetSetting,
                      seed: int, timesteps: int, device: str) -> Dict[str, object]:
    set_all_seeds(seed)

    env = GraphEdgeAddEnv(
        graph_path, reward_type=setting.reward_type,
        budget_alpha_n=budget_setting.alpha_n, budget_beta_m=budget_setting.beta_m,
        budget_cap=budget_setting.cap, budget_min_edges=budget_setting.min_edges,
        gamma=setting.gamma, beta=setting.beta, delta=setting.delta,
        w_fc=setting.w_fc, w_m=setting.w_m, w_lam=setting.w_lam,
    )
    base_ei = env.edge_index_base.clone()

    policy_kwargs = dict(
        features_extractor_class=CleanGNNExtractor,
        features_extractor_kwargs=dict(edge_index=base_ei),
        net_arch=[256, 256],
    )

    model = DQN(
        "MlpPolicy", env, policy_kwargs=policy_kwargs,
        learning_rate=2e-3, buffer_size=50000, batch_size=128,
        learning_starts=1000, train_freq=4, target_update_interval=1000,
        gamma=0.98, device=device, verbose=0, seed=seed,
    )

    base = exact_metrics(env.g_orig)
    model.learn(total_timesteps=timesteps)

    obs, _ = env.reset()
    for _ in range(env.budget * 10):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        if done:
            break

    final = exact_metrics(env.g)

    eps = 1e-8
    pct = {}
    for k, v0 in base.items():
        v1 = final[k]
        pct[f"%Δ_{k}"] = float((v1 - v0) / (v0 + eps) * 100.0)

    row = {
        "network": os.path.basename(graph_path),
        "n": env.n, "m": env.m, "budget": env.budget,
        "budget_name": budget_setting.name,
        "budget_alpha_n": budget_setting.alpha_n,
        "budget_beta_m": budget_setting.beta_m,
        "budget_cap": budget_setting.cap,
        "setting": setting.name,
        "reward_type": setting.reward_type,
        "seed": seed, "timesteps": timesteps,
        "gamma": setting.gamma, "beta": setting.beta, "delta": setting.delta,
        "w_fc": setting.w_fc, "w_m": setting.w_m, "w_lam": setting.w_lam,
    }
    for k, v in base.items():
        row[f"base_{k}"] = v
    for k, v in final.items():
        row[f"final_{k}"] = v
    row.update(pct)
    return row

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS - COLAB MODE")
    print(f"{'='*80}\n")
    
    # Build configurations
    settings = build_settings()
    budget_settings = build_budget_settings(
        sweep=BUDGET_SWEEP,
        alphas=BUDGET_ALPHAS,
        betas=BUDGET_BETAS,
        caps=BUDGET_CAPS,
        min_edges=2
    )
    
    # Filter settings if budget sweep
    if BUDGET_SWEEP and not ALL_SETTINGS_WITH_BUDGET:
        keep = {"PBR_base", "EFFRES_base"}
        settings = [s for s in settings if s.name in keep]
    
    networks = resolve_networks(TOPOLOGY_DIR, MAX_NETWORKS)
    
    rows = []
    t0 = time.time()
    total = len(networks) * len(settings) * len(SEEDS) * len(budget_settings)
    done = 0

    print(f"Networks: {len(networks)} | Settings: {len(settings)} | Budget configs: {len(budget_settings)} | Seeds: {len(SEEDS)}")
    print(f"Total runs: {total} | Timesteps/run: {TIMESTEPS} | Device: {DEVICE}")
    print(f"{'='*80}\n")

    for gp in networks:
        for bs in budget_settings:
            for st in settings:
                for sd in SEEDS:
                    done += 1
                    print(f"[{done:4d}/{total}] {os.path.basename(gp):20s} | {bs.name:15s} | {st.name:20s} | seed={sd}", flush=True)
                    try:
                        row = train_and_eval_one(gp, st, bs, sd, TIMESTEPS, DEVICE)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        row = {
                            "network": os.path.basename(gp),
                            "setting": st.name,
                            "reward_type": st.reward_type,
                            "seed": sd,
                            "error": str(e),
                        }
                    rows.append(row)
                    
                    # Incremental save
                    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    dt = time.time() - t0
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total time: {dt/60:.1f} minutes")
    
    # Budget sweep summary
    if BUDGET_SWEEP:
        try:
            df_all = pd.DataFrame(rows)
            df_ok = df_all[df_all.get("error").isna()] if "error" in df_all.columns else df_all
            df_ok = df_ok.replace([np.inf, -np.inf], np.nan).dropna(
                subset=["%Δ_λ₂", "%Δ_AvgNodeConn", "%Δ_GCC_5%", "%Δ_NatConnectivity", "%Δ_ASPL"])
            
            df_ok["score"] = (
                df_ok["%Δ_λ₂"] + df_ok["%Δ_AvgNodeConn"] + 
                df_ok["%Δ_GCC_5%"] + df_ok["%Δ_NatConnectivity"] - df_ok["%Δ_ASPL"]
            ) / 5.0

            grp = df_ok.groupby(
                ["budget_name", "budget_alpha_n", "budget_beta_m", "budget_cap", "reward_type", "setting"], 
                as_index=False
            ).agg(
                mean_score=("score", "mean"),
                std_score=("score", "std"),
                mean_budget=("budget", "mean"),
                mean_lam2=("%Δ_λ₂", "mean"),
                mean_gcc=("%Δ_GCC_5%", "mean"),
                mean_avgconn=("%Δ_AvgNodeConn", "mean"),
                mean_natconn=("%Δ_NatConnectivity", "mean"),
                mean_aspl=("%Δ_ASPL", "mean"),
                n_runs=("score", "count"),
            )
            grp.to_csv(BUDGET_SUMMARY_CSV, index=False)

            print(f"\n{'='*80}")
            print(f"BUDGET SWEEP: Best configs by reward_type")
            print(f"{'='*80}")
            for rt in sorted(grp["reward_type"].unique()):
                g2 = grp[grp["reward_type"] == rt].sort_values("mean_score", ascending=False)
                if len(g2) == 0:
                    continue
                best = g2.iloc[0]
                print(f"{rt:6s} best={best['budget_name']:15s} "
                      f"(α={best['budget_alpha_n']:.2f}, β={best['budget_beta_m']:.2f}, cap={int(best['budget_cap'])}) "
                      f"mean_budget={best['mean_budget']:.1f} score={best['mean_score']:.3f}±{best['std_score']:.3f}")
            print(f"\nBudget summary saved to: {BUDGET_SUMMARY_CSV}")
        except Exception as e:
            print(f"Budget summary failed: {e}")

if __name__ == "__main__":
    main()
