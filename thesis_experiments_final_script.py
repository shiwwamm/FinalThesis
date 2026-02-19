

### IMPORTS
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import igraph as ig, gymnasium as gym, torch, torch.nn as nn, torch_geometric.nn as pyg_nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
import random, warnings, time
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*80}")
print(f"THESIS EXPERIMENT: 4 Reward Functions on 30 Networks")
print(f"{'='*80}")
print(f"Device: {DEVICE} | Seed: {SEED}")
print(f">>> SCRIPT START - PID: {os.getpid()} <<<")
print(f">>> If you see this PID twice, the script is being run multiple times <<<\n")

# Graph directory
GRAPH_DIR = "./real_world_topologies"  # For local

# Network list - Jenks Natural Breaks: Small (≤40), Medium (41-93), Large (>93)
topo = {
    # Small networks (≤ 40 nodes) - 10 networks
    "Arpanet19706": "Arpanet19706.graphml",        # 9 nodes
    "Abilene": "Abilene.graphml",                  # 11 nodes
    "Cesnet1997": "Cesnet1997.graphml",            # 13 nodes
    "Eenet": "Eenet.graphml",                      # 13 nodes
    "Claranet": "Claranet.graphml",                # 15 nodes
    "Atmnet": "Atmnet.graphml",                    # 21 nodes
    "Belnet2005": "Belnet2005.graphml",            # 23 nodes
    "Bbnplanet": "Bbnplanet.graphml",              # 27 nodes
    "Arpanet19728": "Arpanet19728.graphml",        # 29 nodes
    "Bics": "Bics.graphml",                        # 33 nodes
    # Medium networks (41-93 nodes) - 10 networks
    "Cernet": "Cernet.graphml",                    # 41 nodes
    "Chinanet": "Chinanet.graphml",                # 42 nodes
    "Cesnet200706": "Cesnet200706.graphml",        # 44 nodes
    "Surfnet": "Surfnet.graphml",                  # 50 nodes
    "Garr200902": "Garr200902.graphml",            # 54 nodes
    "Dfn": "Dfn.graphml",                          # 58 nodes
    "Internode": "Internode.graphml",              # 66 nodes
    "Esnet": "Esnet.graphml",                      # 68 nodes
    "Tw": "Tw.graphml",                            # 76 nodes
    "VtlWavenet2011": "VtlWavenet2011.graphml",    # 92 nodes
    # Large networks (> 93 nodes) - 10 networks
    "Interoute": "Interoute.graphml",              # 110 nodes
    "Deltacom": "Deltacom.graphml",                # 113 nodes
    "Ion": "Ion.graphml",                          # 125 nodes
    "Pern": "Pern.graphml",                        # 127 nodes
    "TataNld": "TataNld.graphml",                  # 145 nodes
    "GtsCe": "GtsCe.graphml",                      # 149 nodes
    "Colt": "Colt.graphml",                        # 153 nodes
    "UsCarrier": "UsCarrier.graphml",              # 158 nodes
    "DialtelecomCz": "DialtelecomCz.graphml",      # 193 nodes
    "Cogentco": "Cogentco.graphml",                # 197 nodes
}
GRAPH_FILES = [os.path.join(GRAPH_DIR, fname) for fname in topo.values()]

start_time = time.time()

# ============================================================================
# BUDGET FUNCTION
# ============================================================================
def edge_budget(n: int, m: int) -> int:
    """Budget B(N,M) = clamp(ceil(0.15*N), 2, min(ceil(0.1*M), 20))"""
    node_based = int(np.ceil(0.15 * n))
    edge_based = int(np.ceil(0.10 * m))
    upper = min(edge_based if edge_based > 0 else 20, 20)
    return max(2, min(node_based, upper))

# ============================================================================
# RESILIENCE ENVIRONMENT WITH PBR AND EFFRES
# ============================================================================
class ResilienceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, path, reward_type="pbr", budget_edges=None, 
                 gamma=5.0, beta=1.0, delta=1.0):
        super().__init__()
        g = ig.Graph.Read_GraphML(path).as_undirected()
        self.g_orig = g.copy()
        self.n = g.vcount()
        self.m = g.ecount()
        assert reward_type in ["pbr", "effres", "ivi", "nnsi"]
        self.reward_type = reward_type
        self.gamma = gamma
        self.beta = beta
        self.delta = delta

        if budget_edges is None:
            self.budget = edge_budget(self.n, self.m)
        else:
            self.budget = budget_edges

        existing = set(tuple(sorted(e)) for e in g.get_edgelist())
        self.candidates = [
            (i, j) for i in range(self.n) for j in range(i + 1, self.n)
            if (i, j) not in existing
        ]
        self.action_space = spaces.Discrete(len(self.candidates))
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(self.n * 5,), dtype=np.float32
        )

        feats = self._compute_node_features(self.g_orig)
        self.mean, self.std = feats.mean(0), feats.std(0) + 1e-8

        edges = g.get_edgelist()
        ei = torch.tensor(edges, dtype=torch.long).t()
        self.edge_index_base = torch.cat([ei, ei.flip(0)], dim=1)
        self.added_edges = set()
        self.reset()

    def _compute_node_features(self, g: ig.Graph):
        n = g.vcount()
        deg = np.array(g.degree(), dtype=float)
        try:
            close = np.array(g.closeness(), dtype=float)
        except:
            close = np.zeros(n, dtype=float)
        pr = np.array(g.pagerank(), dtype=float)
        core = np.array(g.coreness(), dtype=float)
        clust = np.array(g.transitivity_local_undirected(mode="zero"), dtype=float)
        return np.stack([deg, close, pr, core, clust], axis=1).astype(np.float32)

    def _approx_lambda2(self, g: ig.Graph):
        """Approximate λ₂ from degree statistics"""
        d = np.array(g.degree(), dtype=float)
        if d.size < 2:
            return 0.0
        d_mean = d.mean()
        d2_mean = (d ** 2).mean()
        var = max(0.0, d2_mean - d_mean ** 2)
        return max(0.0, d_mean - np.sqrt(var))
    
    def action_masks(self):
        """Return boolean mask: True = valid action, False = invalid (already added)"""
        masks = np.ones(len(self.candidates), dtype=bool)
        
        for idx, (u, v) in enumerate(self.candidates):
            edge = tuple(sorted((u, v)))
            # Mask if edge already added OR already exists in graph
            if edge in self.added_edges or self.g.are_connected(u, v):
                masks[idx] = False
        
        return masks

    def _effective_graph_resistance(self, g: ig.Graph):
        """
        Effective graph resistance (Kirchhoff index)
        R_G = n * sum_{i=2..n} 1/λ_i
        Citation: Klein & Randić (1993)
        """
        n = g.vcount()
        if n <= 1:
            return 0.0
        
        try:
            if not g.is_connected():
                return 1e9
        except:
            return 1e9

        L = np.array(g.laplacian(), dtype=float)
        evals = np.linalg.eigvalsh(L)
        evals.sort()
        nonzero = evals[1:]
        nonzero = nonzero[nonzero > 1e-12]
        
        if nonzero.size == 0:
            return 1e9
        
        return float(n * np.sum(1.0 / nonzero))

    # ========== IVI/NNSI Helper Methods ==========
    def _minmax(self, arr: np.ndarray):
        """Min-max normalization to [0,1]"""
        arr = np.asarray(arr, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def _h_index_vec(self, deg: np.ndarray, g: ig.Graph):
        """H-index for each node"""
        n = g.vcount()
        H = np.zeros(n, dtype=float)
        for v in range(n):
            neigh = g.neighbors(v)
            if not neigh:
                continue
            neigh_deg = deg[neigh]
            neigh_deg_sorted = np.sort(neigh_deg)[::-1]
            h = 0
            for i, d in enumerate(neigh_deg_sorted, 1):
                if d >= i:
                    h = i
                else:
                    break
            H[v] = h
        return H

    def _local_h_index_vec(self, H: np.ndarray, g: ig.Graph):
        """Local H-index"""
        n = g.vcount()
        LH = np.zeros(n, dtype=float)
        for v in range(n):
            neigh = g.neighbors(v)
            LH[v] = H[v] + H[neigh].sum() if neigh else H[v]
        return LH

    def _neighborhood_connectivity(self, deg: np.ndarray, g: ig.Graph):
        """Average neighbor degree"""
        n = g.vcount()
        NC = np.zeros(n, dtype=float)
        for v in range(n):
            neigh = g.neighbors(v)
            if not neigh:
                continue
            NC[v] = deg[neigh].mean()
        return NC

    def _cluster_rank(self, deg: np.ndarray, clust: np.ndarray, g: ig.Graph):
        """ClusterRank (Chen et al. 2013)"""
        n = g.vcount()
        CR = np.zeros(n, dtype=float)
        f_c = 10.0 ** clust
        for v in range(n):
            neigh = g.neighbors(v)
            if not neigh:
                continue
            CR[v] = f_c[v] * np.sum(deg[neigh] + 1.0)
        return CR

    def _collective_influence(self, deg: np.ndarray, g: ig.Graph, ell: int = 2):
        """Collective Influence (ℓ = 2)"""
        n = g.vcount()
        CI = np.zeros(n, dtype=float)
        for v in range(n):
            k_i = deg[v]
            if k_i <= 1:
                continue
            one_hop = set(g.neighbors(v))
            two_hop = set()
            for u in one_hop:
                two_hop.update(g.neighbors(u))
            two_hop.discard(v)
            two_hop.difference_update(one_hop)
            if not two_hop:
                continue
            CI[v] = (k_i - 1.0) * np.sum(deg[list(two_hop)] - 1.0)
        return CI

    def _ivi_scores(self, g: ig.Graph):
        """Integrated Value of Influence (Salavaty et al.)"""
        n = g.vcount()
        deg = np.array(g.degree(), dtype=float)
        clust = np.array(g.transitivity_local_undirected(mode="zero"), dtype=float)
        NC = self._neighborhood_connectivity(deg, g)
        H = self._h_index_vec(deg, g)
        LH = self._local_h_index_vec(H, g)
        BC = np.array(g.betweenness(), dtype=float)
        CI = self._collective_influence(deg, g, ell=2)
        CR = self._cluster_rank(deg, clust, g)

        DC0 = self._minmax(deg)
        LH0 = self._minmax(LH)
        NC0 = self._minmax(NC)
        CR0 = self._minmax(CR)
        BC0 = self._minmax(BC)
        CI0 = self._minmax(CI)

        hub = DC0 + LH0
        spread = (NC0 + CR0) / (BC0 + CI0 + 1e-8)
        ivi = hub * spread
        return ivi

    def _nnsi_scores(self, g: ig.Graph):
        """Network Node Significance Index"""
        n = g.vcount()
        deg = np.array(g.degree(), dtype=float)
        close = np.array(g.closeness() or [0.0] * n, dtype=float)
        BC = np.array(g.betweenness(), dtype=float)
        core = np.array(g.coreness(), dtype=float)
        pr = np.array(g.pagerank(), dtype=float)
        CI = self._collective_influence(deg, g, ell=2)

        DC0 = self._minmax(deg)
        CC0 = self._minmax(close)
        BC0 = self._minmax(BC)
        K0  = self._minmax(core)
        PR0 = self._minmax(pr)
        CI0 = self._minmax(CI)

        CS  = DC0 + CC0
        FCS = BC0 + K0
        IPS = PR0 + CI0
        nnsi = CS * FCS * IPS
        return nnsi
    # ========== End IVI/NNSI Helpers ==========


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.g = self.g_orig.copy()
        self.added_edges = set()
        self.steps = 0
        self.successful_additions = 0  # Track only successful edge additions

        # Initialize baselines based on reward type
        self.prev_lambda2 = self._approx_lambda2(self.g)
        
        if self.reward_type == "pbr":
            # PBR baselines
            degrees = np.array(self.g.degree(), dtype=float)
            k_mean = degrees.mean()
            k2_mean = (degrees ** 2).mean()
            self.prev_moment = k2_mean / (k_mean + 1e-8)
            self.prev_fc = 1.0 - 1.0 / (self.prev_moment - 1.0 + 1e-8)
        
        elif self.reward_type == "effres":
            # EffRes baselines
            self.prev_effres = self._effective_graph_resistance(self.g)
        
        elif self.reward_type == "ivi":
            # IVI baselines
            self.prev_max_ivi = self._ivi_scores(self.g).max()
        
        elif self.reward_type == "nnsi":
            # NNSI baselines
            self.prev_max_nnsi = self._nnsi_scores(self.g).max()

        obs = self._obs()
        return obs, {}

    def _obs(self):
        feats = self._compute_node_features(self.g)
        return ((feats - self.mean) / self.std).flatten().astype(np.float32)

    def step(self, action):
        u, v = self.candidates[action]
        edge = tuple(sorted((u, v)))

        if edge in self.added_edges or self.g.are_connected(u, v):
            reward = -0.5
        else:
            self.g.add_edge(u, v)
            self.added_edges.add(edge)
            self.successful_additions += 1  # Only count successful additions
            reward = self._compute_reward()
            
            # Update edge_index_base to include the new edge for GNN
            new_edge = torch.tensor([[u, v], [v, u]], dtype=torch.long).t()
            self.edge_index_base = torch.cat([self.edge_index_base, new_edge], dim=1)

        self.steps += 1
        
        # Done when budget edges added OR too many failed attempts (safety limit)
        max_attempts = self.budget * 10  # Allow up to 10x budget in total attempts
        done = self.successful_additions >= self.budget or self.steps >= max_attempts
        
        obs = self._obs()
        return obs, float(reward), done, False, {}

    def _compute_reward(self):
        curr_lambda2 = self._approx_lambda2(self.g)
        d_lambda2 = curr_lambda2 - self.prev_lambda2

        if self.reward_type == "pbr":
            r = self._reward_pbr(curr_lambda2, d_lambda2)
        elif self.reward_type == "effres":
            r = self._reward_effres(d_lambda2)
        elif self.reward_type == "ivi":
            r = self._reward_ivi(d_lambda2)
        elif self.reward_type == "nnsi":
            r = self._reward_nnsi(d_lambda2)
        else:
            r = 0.0

        self.prev_lambda2 = curr_lambda2
        return max(min(r, 5.0), -5.0)

    def _reward_pbr(self, curr_lambda2, d_lambda2):
        """
        Percolation-Based Resilience (PBR)
        Based on Cohen et al. (2000), Callaway et al. (2000), Schneider et al. (2011)
        
        r = 0.5 * Δf_c + 0.3 * Δ(moment) + 0.2 * Δλ₂
        """
        degrees = np.array(self.g.degree(), dtype=float)
        k_mean = degrees.mean()
        k2_mean = (degrees ** 2).mean()
        
        # Degree distribution moment (Callaway et al. 2000)
        curr_moment = k2_mean / (k_mean + 1e-8)
        
        # Critical fraction (Cohen et al. 2000)
        curr_fc = 1.0 - 1.0 / (curr_moment - 1.0 + 1e-8)
        
        # Compute deltas
        d_fc = curr_fc - self.prev_fc
        d_moment = self.prev_moment - curr_moment  # Lower is better
        
        # PBR reward (weights from Schneider et al. 2011)
        r = 0.5 * d_fc + 0.3 * d_moment + 0.2 * d_lambda2
        
        # Update baselines
        self.prev_fc = curr_fc
        self.prev_moment = curr_moment
        
        return r

    def _reward_effres(self, d_lambda2):
        """
        Effective Resistance + λ₂ shaping
        Based on Klein & Randić (1993)
        
        r = δ * ΔEffRes_rel + β * max(0, Δλ₂) - γ * max(0, -Δλ₂)
        """
        curr_effres = self._effective_graph_resistance(self.g)
        
        # Relative change in effective resistance
        prev = float(self.prev_effres) if self.prev_effres is not None else 1e9
        d_effres = (prev - float(curr_effres)) / (abs(prev) + 1e-9)
        
        # λ₂ shaping
        lambda_bonus = self.beta * max(0.0, d_lambda2)
        lambda_penalty = self.gamma * max(0.0, -d_lambda2)
        
        r = self.delta * d_effres + lambda_bonus - lambda_penalty
        
        # Update baseline
        self.prev_effres = float(curr_effres)
        
        return r

    def _reward_ivi(self, d_lambda2):
        """
        IVI-based reward: Reduce maximum IVI (target high-influence nodes)
        r = ΔIVI (pure IVI optimization, no λ₂ shaping)
        """
        curr_max_ivi = self._ivi_scores(self.g).max()
        d_ivi = self.prev_max_ivi - curr_max_ivi  # >0 => IVI reduced (good)
        
        r = d_ivi  # Pure IVI reward, no connectivity penalty
        
        # Update baseline
        self.prev_max_ivi = curr_max_ivi
        
        return r

    def _reward_nnsi(self, d_lambda2):
        """
        NNSI-based reward: Reduce maximum NNSI (target structurally significant nodes)
        r = ΔNNSI (pure NNSI optimization, no λ₂ shaping)
        """
        curr_max_nnsi = self._nnsi_scores(self.g).max()
        d_nnsi = self.prev_max_nnsi - curr_max_nnsi  # >0 => NNSI reduced (good)
        
        r = d_nnsi  # Pure NNSI reward, no connectivity penalty
        
        # Update baseline
        self.prev_max_nnsi = curr_max_nnsi
        
        return r


# ============================================================================
# GNN FEATURE EXTRACTOR (with dynamic edge index support)
# ============================================================================
class CleanGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, env, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.n_nodes = observation_space.shape[0] // 5
        self.env = env  # Store reference to environment to get updated edge_index

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
        
        # Get current edge_index from environment (updated after each edge addition)
        edge_index = self.env.edge_index_base.to(obs.device)
        
        x = self.gnn(x, edge_index)
        batch = torch.arange(b, device=obs.device).repeat_interleave(self.n_nodes)
        x = self.pool(x, batch)
        x = self.proj(x)
        return x

# ============================================================================
# EVALUATION METRICS
# ============================================================================
def exact_metrics(g: ig.Graph):
    """Compute all evaluation metrics"""
    n = g.vcount()
    if n < 2:
        return {
            "λ₂": 0.0, "AvgNodeConn": 0.0, "GCC_5%": 1.0, "ASPL": 0.0,
            "Diameter": 0.0, "ArticulationPoints": 0, "Bridges": 0,
            "BetCentralization": 0.0, "NatConnectivity": 0.0,
            "EffResistance": 0.0, "Assortativity": 0.0, "AvgClustering": 0.0,
        }

    # λ₂ (Algebraic Connectivity)
    L = np.array(g.laplacian(normalized=True))
    eigs = np.sort(np.linalg.eigvalsh(L))
    lambda2 = eigs[1] if len(eigs) > 1 else 0.0

    # Average Node Connectivity (sample-based for large graphs)
    # More discriminative than min-cut
    if n <= 50:
        # Exact for small graphs
        try:
            avg_node_conn = g.vertex_connectivity()
        except:
            avg_node_conn = 0.0
    else:
        # Sample-based approximation for large graphs
        sample_size = min(50, n)
        sample_nodes = np.random.choice(n, size=sample_size, replace=False)
        connectivities = []
        for i in range(len(sample_nodes)):
            for j in range(i + 1, min(i + 6, len(sample_nodes))):  # Limit pairs
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

    # Effective Graph Resistance (Kirchhoff Index)
    # Klein & Randić (1993) - lower is better for resilience
    try:
        if g.is_connected():
            L_unnorm = np.array(g.laplacian())
            evals = np.linalg.eigvalsh(L_unnorm)
            evals.sort()
            nonzero = evals[1:]  # Skip first eigenvalue (≈0)
            nonzero = nonzero[nonzero > 1e-12]
            if nonzero.size > 0:
                eff_res = float(n * np.sum(1.0 / nonzero))
            else:
                eff_res = 1e9
        else:
            eff_res = 1e9
    except:
        eff_res = 1e9

    # Assortativity (degree correlation)
    # Negative values indicate resilient "onion-like" structure
    try:
        assort = g.assortativity_degree()
    except:
        assort = 0.0

    # Average Clustering Coefficient
    try:
        avg_clust = g.transitivity_avglocal_undirected(mode="zero")
    except:
        avg_clust = 0.0

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
        "EffResistance": float(eff_res),
        "Assortativity": float(assort),
        "AvgClustering": float(avg_clust),
    }

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================
print(f"\nTesting 2 reward functions on {len(GRAPH_FILES)} networks")
print(f"Total experiments: {len(GRAPH_FILES) * 4} (30 networks × 4 reward functions)")
print(f"{'='*80}\n")

results = []

# Rollout trace: log EVERY attempt (added edges + invalid/duplicate actions).
# This enables post-hoc diagnosis of *why* each reward leads to the observed final metrics.
attempt_records = []

# Reward landscape: potential reward for ALL candidate edges at initial state
all_reward_landscapes = []

network_times = []
metrics = ["λ₂", "AvgNodeConn", "GCC_5%", "ASPL", "Diameter",
           "ArticulationPoints", "Bridges", "BetCentralization", "NatConnectivity",
           "EffResistance", "Assortativity", "AvgClustering"]

for idx, path in enumerate(tqdm(GRAPH_FILES, desc="Overall Progress"), 1):
    network_start = time.time()
    name = os.path.basename(path).split(".")[0]
    orig_g = ig.Graph.Read_GraphML(path).as_undirected()
    n, m = orig_g.vcount(), orig_g.ecount()

    print(f"\n{'='*80}")
    print(f"Network {idx}/{len(GRAPH_FILES)}: {name} (N={n}, M={m})")
    print(f"{'='*80}")

    # Original metrics
    orig_met = exact_metrics(orig_g)
    row = {"Graph": name, "N": n, "M": m}
    for k, v in orig_met.items():
        row[f"Orig_{k}"] = v

    # Budget
    B = edge_budget(n, m)
    row["BudgetEdges"] = B
    print(f"Budget: {B} edges")

    # Test all four reward functions
    for reward_idx, reward_type in enumerate(["pbr", "effres", "ivi", "nnsi"], 1):
        print(f"  [{reward_idx}/4] {reward_type.upper():10s} - Training...", end=" ", flush=True)
        
        try:
            env = ResilienceEnv(path, reward_type=reward_type, budget_edges=B,
                              gamma=5.0, beta=1.0, delta=1.0)
            
            policy_kwargs = dict(
                features_extractor_class=CleanGNNExtractor,
                features_extractor_kwargs=dict(env=env),
                net_arch=[256, 256],
            )
            
            model = DQN(
                "MlpPolicy", env, policy_kwargs=policy_kwargs,
                learning_rate=2e-3, buffer_size=50000, batch_size=128,
                learning_starts=1000, train_freq=4, target_update_interval=1000,
                gamma=0.98, device=DEVICE, verbose=0, seed=SEED,
            )
            
            # Training with progress
            print("25k steps...", end=" ", flush=True)
            model.learn(total_timesteps=25000)
            print("Evaluating...", end=" ", flush=True)
            
            # Final rollout - log EVERY attempt (added edges + invalid/duplicate actions)
            obs, _ = env.reset()
            rollout_attempts = []
            successful_count = 0
            reward_landscape = []
            
            # Helper function to compute reward landscape at current state
            def compute_reward_landscape_at_step(current_g, step_num):
                """Compute potential rewards for all remaining candidate edges"""
                step_landscape = []
                
                # Get current graph properties for context
                degrees_current = {i: deg for i, deg in enumerate(current_g.degree())}
                betweenness_current = current_g.betweenness()
                
                for cand_idx, (u, v) in enumerate(env.candidates):
                    # Skip if edge already exists or was already added
                    if current_g.are_connected(u, v):
                        continue
                    
                    # Temporarily add edge and compute reward
                    temp_g = current_g.copy()
                    temp_g.add_edge(u, v)
                    
                    # Save current env state
                    saved_g = env.g
                    saved_lambda2 = env.prev_lambda2
                    saved_attrs = {}
                    
                    if env.reward_type == "pbr":
                        saved_attrs['moment'] = env.prev_moment
                        saved_attrs['fc'] = env.prev_fc
                    elif env.reward_type == "effres":
                        saved_attrs['effres'] = env.prev_effres
                    elif env.reward_type == "ivi":
                        saved_attrs['max_ivi'] = env.prev_max_ivi
                    elif env.reward_type == "nnsi":
                        saved_attrs['max_nnsi'] = env.prev_max_nnsi
                    
                    # Set env to current state and compute reward
                    env.g = temp_g
                    env.prev_lambda2 = env._approx_lambda2(current_g)
                    
                    if env.reward_type == "pbr":
                        degrees = np.array(current_g.degree(), dtype=float)
                        k_mean = degrees.mean()
                        k2_mean = (degrees ** 2).mean()
                        env.prev_moment = k2_mean / (k_mean + 1e-8)
                        env.prev_fc = 1.0 - 1.0 / (env.prev_moment - 1.0 + 1e-8)
                    elif env.reward_type == "effres":
                        env.prev_effres = env._effective_graph_resistance(current_g)
                    elif env.reward_type == "ivi":
                        env.prev_max_ivi = env._ivi_scores(current_g).max()
                    elif env.reward_type == "nnsi":
                        env.prev_max_nnsi = env._nnsi_scores(current_g).max()
                    
                    # Compute reward for this edge
                    reward = env._compute_reward()
                    
                    # Restore env state
                    env.g = saved_g
                    env.prev_lambda2 = saved_lambda2
                    if env.reward_type == "pbr":
                        env.prev_moment = saved_attrs['moment']
                        env.prev_fc = saved_attrs['fc']
                    elif env.reward_type == "effres":
                        env.prev_effres = saved_attrs['effres']
                    elif env.reward_type == "ivi":
                        env.prev_max_ivi = saved_attrs['max_ivi']
                    elif env.reward_type == "nnsi":
                        env.prev_max_nnsi = saved_attrs['max_nnsi']
                    
                    step_landscape.append({
                        "Graph": name,
                        "Reward": reward_type.upper(),
                        "Step": step_num,
                        "CandidateIndex": cand_idx,
                        "Node_U": int(u),
                        "Node_V": int(v),
                        "Edge": f"{u}-{v}",
                        "PotentialReward": float(reward),
                        "NodeU_Degree": degrees_current[u],
                        "NodeV_Degree": degrees_current[v],
                        "NodeU_Betweenness": float(betweenness_current[u]),
                        "NodeV_Betweenness": float(betweenness_current[v]),
                    })
                
                return step_landscape
            
            # Helper function to get greedy (highest reward) action
            def get_greedy_action(landscape_at_step):
                """Return the candidate index with highest potential reward"""
                if not landscape_at_step:
                    return None
                max_entry = max(landscape_at_step, key=lambda x: x['PotentialReward'])
                return max_entry['CandidateIndex'], max_entry['PotentialReward'], max_entry['Edge']
            
            # Compute initial reward landscape (Step 0)
            print("Computing reward landscape...", end=" ", flush=True)
            initial_landscape = compute_reward_landscape_at_step(env.g.copy(), step_num=0)
            reward_landscape.extend(initial_landscape)
            
            # Get greedy baseline for step 0
            greedy_action_0, greedy_reward_0, greedy_edge_0 = get_greedy_action(initial_landscape)
            print(f"Step 0: {len(initial_landscape)} candidates (greedy={greedy_edge_0}, r={greedy_reward_0:.4f})...", end=" ", flush=True)

            for step_num in range(env.budget):  # Only need budget steps now!
                # Get valid action mask
                valid_actions = env.action_masks()
                
                # If no valid actions left, stop
                if not valid_actions.any():
                    break
                
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # If action is invalid, pick random valid action
                if not valid_actions[action]:
                    valid_indices = np.where(valid_actions)[0]
                    action = np.random.choice(valid_indices)
                
                u, v = env.candidates[action]
                edge = tuple(sorted((u, v)))
                
                # Get greedy choice for comparison
                current_landscape = [entry for entry in initial_landscape if entry['Step'] == successful_count]
                if not current_landscape:
                    # Recompute if not available
                    current_landscape = compute_reward_landscape_at_step(env.g.copy(), step_num=successful_count)
                
                greedy_info = get_greedy_action(current_landscape)
                greedy_action, greedy_reward, greedy_edge = greedy_info if greedy_info else (None, None, None)
                
                obs, reward, done, _, _ = env.step(action)
                
                # Log attempt (always log, even if not added)
                rollout_attempts.append({
                    "Graph": name,
                    "Reward": reward_type.upper(),
                    "Step": successful_count + 1,
                    "Node_U": int(u),
                    "Node_V": int(v),
                    "Edge": f"{u}-{v}",
                    "WasAdded": True,  # With action masking, all attempts succeed
                    "StepReward": float(reward),
                    "AgentAction": int(action),
                    "GreedyAction": int(greedy_action) if greedy_action is not None else -1,
                    "GreedyEdge": greedy_edge if greedy_edge else "N/A",
                    "GreedyReward": float(greedy_reward) if greedy_reward is not None else np.nan,
                    "AgentMatchedGreedy": (action == greedy_action) if greedy_action is not None else False,
                })
                
                successful_count += 1
                
                # Compute reward landscape after this edge addition
                if successful_count < env.budget:  # Only compute if more edges to add
                    step_landscape = compute_reward_landscape_at_step(env.g.copy(), step_num=successful_count)
                    reward_landscape.extend(step_landscape)
                
                if done:
                    break
            
            # Store attempt trace for this rollout
            attempt_records.extend(rollout_attempts)
            
            # Store reward landscape for this network/reward combination
            all_reward_landscapes.extend(reward_landscape)
            
            # Evaluate
            final_met = exact_metrics(env.g)
            prefix = reward_type.upper()
            for k, v in final_met.items():
                row[f"{prefix}_{k}"] = v
            
            print(f"✓ ({len(reward_landscape)} landscape entries)")
        except Exception as e:
            print(f"✗ Error: {e}")
            for k in metrics:
                row[f"{reward_type.upper()}_{k}"] = np.nan

    network_time = time.time() - network_start
    network_times.append(network_time)
    avg_time = np.mean(network_times)
    remaining = len(GRAPH_FILES) - idx
    eta_minutes = (avg_time * remaining) / 60
    
    print(f"  Network completed in {network_time/60:.1f} min | Avg: {avg_time/60:.1f} min | ETA: {eta_minutes:.1f} min ({eta_minutes/60:.1f}h)")
    results.append(row)

df = pd.DataFrame(results)

# ============================================================================
# COMPUTE IMPROVEMENTS
# ============================================================================
for k in metrics:
    orig_col = f"Orig_{k}"
    for r in ["PBR", "EFFRES", "IVI", "NNSI"]:
        col = f"{r}_{k}"
        if col in df.columns:
            df[f"%Δ_{r}_vs_Orig_{k}"] = (
                (df[col] - df[orig_col]) / (df[orig_col].abs() + 1e-8) * 100
            ).round(2)

# PBR vs EffRes comparison
for k in metrics:
    df[f"%Δ_PBR_vs_EFFRES_{k}"] = (
        (df[f"PBR_{k}"] - df[f"EFFRES_{k}"]) / (df[f"EFFRES_{k}"].abs() + 1e-8) * 100
    ).round(2)

# ============================================================================
# SAVE RESULTS
# ============================================================================
output_file = "./test_thesis_4rewards_30networks_metrics.csv"  # For local

output_edges_file = "./test_thesis_4rewards_30networks_edges_all_attempts.csv"  # For local

output_edges_added_file = "./test_thesis_4rewards_30networks_edges_added_only.csv"  # For local

df.to_csv(output_file, index=False)

# Save attempt trace
print(f"\n>>> Saving attempt records... ({len(attempt_records)} records)")
df_edges = pd.DataFrame(attempt_records)
print(f">>> Created DataFrame with {len(df_edges)} rows")
df_edges.to_csv(output_edges_file, index=False)
print(f">>> Saved to: {output_edges_file}")

# Save reward landscape (all candidate edges with their potential rewards)
if len(all_reward_landscapes) > 0:
    output_landscape_file = "./thesis_4rewards_30networks_reward_landscape.csv"
    print(f"\n>>> Saving reward landscape... ({len(all_reward_landscapes)} candidate edges)")
    df_landscape = pd.DataFrame(all_reward_landscapes)
    print(f">>> Created DataFrame with {len(df_landscape)} rows")
    df_landscape.to_csv(output_landscape_file, index=False)
    print(f">>> Saved to: {output_landscape_file}")
    print(f">>> This file contains the potential reward for EVERY candidate edge at EACH step (0 to budget-1)")
    print(f">>> Step 0 = initial state, Step 1 = after 1st edge added, etc.")


# Save added-only subset (one row per edge actually added)
print(f">>> Filtering for added edges only...")
df_edges_added = df_edges[df_edges["WasAdded"] == True].copy()
print(f">>> Found {len(df_edges_added)} added edges")
df_edges_added.to_csv(output_edges_added_file, index=False)
print(f">>> Saved to: {output_edges_added_file}")

print(f"\n{'='*80}")
print(f"FILES SAVED:")
print(f"{'='*80}")
print(f"📊 Metrics: {output_file}")
print(f"🔗 Edges:   {output_edges_file}")
print(f"➕ Added-only edges: {output_edges_added_file}")
print(f"{'='*80}")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)

print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETE!")
print(f"{'='*80}")
print(f"\n✅ Results saved to: {output_file}")
print(f"✅ Edges saved to: {output_edges_file}")
print(f"✅ Added-only edges saved to: {output_edges_added_file}")
print(f"⏱️  Total time: {hours}h {minutes}m")
print(f"📊 Networks: {len(df)}")
print(f"📈 Experiments: {len(df) * 4}")  # Updated to 4 reward functions
print(f"🔁 Total attempts tracked: {len(df_edges)}")
print(f"➕ Total edges added tracked: {len(df_edges_added)}")

# Display sample results
print(f"\n>>> About to display sample results...")
print(f"\n{'='*80}")
print(f"SAMPLE RESULTS (first 5 networks)")
print(f"{'='*80}\n")
display_cols = ["Graph", "N", "M", "BudgetEdges"]
for k in ["λ₂", "AvgNodeConn", "GCC_5%", "NatConnectivity", "EffResistance"]:
    display_cols += [f"PBR_{k}", f"EFFRES_{k}", f"IVI_{k}", f"NNSI_{k}"]

print(df[display_cols].head().to_string(index=False))

print(f"\n{'='*80}")
print(f"Full results in: {output_file}")
print(f"{'='*80}")
print(f"\n🎉 EXPERIMENT FINISHED - Script will now exit")
print(f"{'='*80}\n")

# Explicitly exit to prevent re-running
print(">>> EXITING NOW - If you see this message twice, something is wrong <<<")
print(f">>> Final PID: {os.getpid()} <<<")
import sys
print(">>> Calling sys.exit(0) now...")
sys.exit(0)
print(">>> THIS LINE SHOULD NEVER PRINT - IF YOU SEE THIS, sys.exit() FAILED <<<")
