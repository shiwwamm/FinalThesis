
# ============================================================================
# CRITICAL: Prevent thread oversubscription in VMs (must be before imports)
# ============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_INTEROP_THREADS"] = "1"

### IMPORTS
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import igraph as ig, gymnasium as gym, torch, torch.nn as nn, torch_geometric.nn as pyg_nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from tqdm import tqdm
import random, warnings, time
warnings.filterwarnings("ignore")

# Verify thread limits (environment variables should have set these)
print(f"PyTorch threads: {torch.get_num_threads()}, interop: {torch.get_num_interop_threads()}")

# ============================================================================
# OPTIMIZATION HELPERS FOR LARGE GRAPHS
# ============================================================================
def approximate_betweenness(g, sample_size=100):
    """
    Approximate betweenness centrality using sampling for large graphs.
    Much faster than exact computation for graphs with >300 nodes.
    
    FIXED: Use 'sources' parameter, not 'vertices' parameter.
    'vertices' selects which nodes' betweenness to return.
    'sources' selects which nodes to use as sources in shortest path computation.
    """
    n = g.vcount()
    if n <= LARGE_GRAPH_THRESHOLD:
        return np.array(g.betweenness(), dtype=float)
    
    # Use sampling-based approximation
    try:
        # Sample a subset of vertices as sources
        sample_size = min(sample_size, n)
        sampled_sources = random.sample(range(n), sample_size)
        
        # FIXED: Use 'sources' parameter for sampling-based approximation
        # This computes betweenness for ALL nodes using only sampled sources
        bet = np.array(g.betweenness(sources=sampled_sources), dtype=float)
        
        # Scale up the estimate (betweenness scales with number of source-target pairs)
        bet = bet * (n / sample_size)
        return bet
    except:
        # Fallback to degree-based approximation if sampling fails
        deg = np.array(g.degree(), dtype=float)
        return deg / deg.sum() * n if deg.sum() > 0 else np.zeros(n)

def safe_compute_metric(func, default_value, *args, **kwargs):
    """
    Safely compute a metric with timeout and error handling.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"  Warning: Metric computation failed, using default. Error: {e}")
        return default_value

def approximate_diameter(g):
    """
    Approximate diameter for large graphs using sampling.
    """
    n = g.vcount()
    if n <= LARGE_GRAPH_THRESHOLD or not g.is_connected():
        try:
            return g.diameter()
        except:
            return np.inf
    
    # Sample-based approximation
    try:
        sample_size = min(50, n)
        sampled = random.sample(range(n), sample_size)
        max_dist = 0
        for i in sampled:
            dists = g.shortest_paths(source=i)[0]
            max_dist = max(max_dist, max([d for d in dists if d != np.inf]))
        return max_dist
    except:
        return np.inf

def approximate_avg_path_length(g):
    """
    Approximate average path length for large graphs.
    """
    n = g.vcount()
    if n <= LARGE_GRAPH_THRESHOLD:
        try:
            return g.average_path_length()
        except:
            return np.inf
    
    # Sample-based approximation
    try:
        sample_size = min(50, n)
        sampled = random.sample(range(n), sample_size)
        total_dist = 0
        count = 0
        for i in sampled:
            dists = g.shortest_paths(source=i)[0]
            valid_dists = [d for d in dists if d != np.inf and d > 0]
            if valid_dists:
                total_dist += sum(valid_dists)
                count += len(valid_dists)
        return total_dist / count if count > 0 else np.inf
    except:
        return np.inf

def approximate_lambda2(g):
    """
    Compute λ₂ (algebraic connectivity) with optimization for large graphs.
    Uses sparse eigenvalue solver for graphs >300 nodes.
    """
    n = g.vcount()
    if n < 2:
        return 0.0
    
    try:
        # For small graphs, use exact method
        if n <= LARGE_GRAPH_THRESHOLD:
            L = np.array(g.laplacian(normalized=True))
            eigs = np.sort(np.linalg.eigvalsh(L))
            lambda2 = eigs[1] if len(eigs) > 1 else 0.0
            return float(lambda2)
        
        # For large graphs, use sparse eigenvalue solver
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        # Get Laplacian as sparse matrix
        L = g.laplacian(normalized=True)
        L_sparse = csr_matrix(L)
        
        # Compute only the 2 smallest eigenvalues (much faster)
        eigs = eigsh(L_sparse, k=2, which='SM', return_eigenvectors=False)
        eigs = np.sort(eigs)
        lambda2 = eigs[1] if len(eigs) > 1 else 0.0
        return float(lambda2)
    except:
        # Fallback
        try:
            L = np.array(g.laplacian(normalized=True))
            eigs = np.sort(np.linalg.eigvalsh(L))
            lambda2 = eigs[1] if len(eigs) > 1 else 0.0
            return float(lambda2)
        except:
            return 0.0

# ============================================================================
# CONFIGURATION
# ============================================================================
# FIXED: Use RUN_SEED from environment if provided, otherwise default to 42
SEED = int(os.environ.get('RUN_SEED', 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimization thresholds for large graphs
LARGE_GRAPH_THRESHOLD = 300  # Use approximations for graphs with >300 nodes
BETWEENNESS_SAMPLE_SIZE = 100  # Sample size for betweenness approximation

# Check for run information from environment
run_number = os.environ.get('RUN_NUMBER', None)
total_runs = os.environ.get('TOTAL_RUNS', None)

print(f"\n{'='*80}")
if run_number and total_runs:
    print(f"THESIS EXPERIMENT RUN {run_number}/{total_runs}")
else:
    print(f"THESIS EXPERIMENT: 4 Reward Functions on Networks")
print(f"{'='*80}")
print(f"Device: {DEVICE} | Seed: {SEED}")
print(f"Script PID: {os.getpid()}\n")

# Graph directory
GRAPH_DIR = "./real_world_topologies"  # For local

# Check if custom graph list is provided via environment variable
import sys
if len(sys.argv) > 1 and sys.argv[1] == "--graph-list":
    # Load graph list from file specified in second argument
    graph_list_file = sys.argv[2]
    print(f"Loading custom graph list from: {graph_list_file}")
    
    # Import the graph list
    import importlib.util
    spec = importlib.util.spec_from_file_location("graph_list", graph_list_file)
    graph_list_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_list_module)
    topo = graph_list_module.topo
    
    print(f"Loaded {len(topo)} graphs from custom list")
else:
    # Default network list - Jenks Natural Breaks: Small (≤40), Medium (41-93), Large (>93)
    topo = {
        # Small networks (≤ 40 nodes) - 10 networks
        "Arpanet19706": "Arpanet19706.graphml",        # 9 nodes
    }

GRAPH_FILES = [os.path.join(GRAPH_DIR, fname) for fname in topo.values()]

start_time = time.time(  )

# ============================================================================
# BUDGET FUNCTION
# ============================================================================
def edge_budget(
    n: int,
    m: int,
    delta_k_bar: float = 0.5,  # target increase in average degree
    max_edges_compute: int | None = None,
) -> int:
    """
    Compute edge budget based on target average degree increase.
    
    Args:
        n: Number of nodes
        m: Number of existing edges
        delta_k_bar: Target increase in average degree (default: 0.5)
        max_edges_compute: Optional maximum budget for computational reasons
    
    Returns:
        Number of edges to add
    
    Formula:
        - Each added edge raises average degree by 2/n
        - To increase avg degree by delta_k_bar, need: delta_k_bar * n / 2 edges
        - Capped by available headroom (non-edges)
    """
    headroom = n * (n - 1) // 2 - m  # number of non-edges available
    if headroom <= 0:
        return 0
    
    # Each added edge raises average degree by 2/n
    b = int(np.ceil(delta_k_bar * n / 2.0))
    
    if max_edges_compute is not None:
        b = min(b, max_edges_compute)
    
    return min(b, headroom)

# ============================================================================
# REWARD-AGNOSTIC CANDIDATE SHORTLIST BUILDER
# ============================================================================
def build_candidate_shortlist(
    g: ig.Graph,
    k: int,
    rng: random.Random,
    bet_sample_size: int = 100,
    max_pool: int = 5000,
):
    """
    Build a fixed-size shortlist of feasible non-edges.
    
    Channels:
    1) long shortest-path pairs
    2) high betweenness endpoint pairs
    3) low-degree endpoint pairs
    4) random feasible pairs
    
    Returns a list of length <= k containing (u, v) tuples.
    """
    n = g.vcount()
    existing = set(tuple(sorted(e)) for e in g.get_edgelist())
    
    all_non_edges = [
        (i, j) for i in range(n) for j in range(i + 1, n)
        if (i, j) not in existing
    ]
    if not all_non_edges:
        return []
    
    # Optional presampling for large graphs
    if len(all_non_edges) > max_pool:
        pool = rng.sample(all_non_edges, max_pool)
    else:
        pool = all_non_edges
    
    degrees = np.array(g.degree(), dtype=float)
    bet = approximate_betweenness(g, sample_size=bet_sample_size)
    
    # Distance cache only for sources we need
    sources = sorted({u for u, _ in pool})
    dist_cache = {}
    for s in sources:
        try:
            sp = g.shortest_paths(source=s)
            # Ensure it's a list, not a single value
            if isinstance(sp, list) and len(sp) > 0:
                dist_cache[s] = sp[0]
            else:
                dist_cache[s] = [np.inf] * n
        except Exception:
            dist_cache[s] = [np.inf] * n
    
    scored = []
    for (u, v) in pool:
        # Safely get distance
        if u in dist_cache and isinstance(dist_cache[u], (list, np.ndarray)) and v < len(dist_cache[u]):
            d = dist_cache[u][v]
        else:
            d = np.inf
            
        if d == np.inf or np.isinf(d):
            d = float(n)  # disconnected pairs get highest priority
        scored.append({
            "edge": (u, v),
            "dist_score": float(d),                    # higher is better
            "bet_score": float(bet[u] + bet[v]),      # higher is better
            "deg_score": float(-(degrees[u] + degrees[v])),  # lower-degree endpoints preferred
        })
    
    quarter = max(1, k // 4)
    
    top_dist = [x["edge"] for x in sorted(scored, key=lambda z: z["dist_score"], reverse=True)[:quarter]]
    top_bet  = [x["edge"] for x in sorted(scored, key=lambda z: z["bet_score"], reverse=True)[:quarter]]
    top_deg  = [x["edge"] for x in sorted(scored, key=lambda z: z["deg_score"], reverse=True)[:quarter]]
    
    remaining = list({x["edge"] for x in scored} - set(top_dist) - set(top_bet) - set(top_deg))
    rand_pick = rng.sample(remaining, min(quarter, len(remaining))) if remaining else []
    
    shortlist = []
    seen = set()
    for edge in top_dist + top_bet + top_deg + rand_pick:
        if edge not in seen:
            shortlist.append(edge)
            seen.add(edge)
        if len(shortlist) >= k:
            break
    
    # Refill if duplicates reduced the size
    if len(shortlist) < k:
        refill = [x["edge"] for x in sorted(
            scored,
            key=lambda z: (z["dist_score"], z["bet_score"], z["deg_score"]),
            reverse=True
        )]
        for edge in refill:
            if edge not in seen:
                shortlist.append(edge)
                seen.add(edge)
            if len(shortlist) >= k:
                break
    
    return shortlist[:k]

# ============================================================================
# RESILIENCE ENVIRONMENT WITH PBR AND EFFRES
# ============================================================================
class ResilienceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        path,
        reward_type="pbr",
        budget_edges=None,
        gamma=5.0,
        beta=1.0,
        delta=1.0,
        shortlist_size=None,
        max_candidate_pool=5000,
    ):
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
        
        # Fixed action-space size
        if shortlist_size is None:
            self.shortlist_size = min(512, max(128, 16 * self.budget))
        else:
            self.shortlist_size = int(shortlist_size)
        
        self.max_candidate_pool = max_candidate_pool
        self.rng = random.Random(SEED)
        
        self.action_space = spaces.Discrete(self.shortlist_size)
        # Observation: node features (n * 5) + progress features (2)
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(self.n * 5 + 2,), dtype=np.float32
        )
        
        feats = self._compute_node_features(self.g_orig)
        self.mean, self.std = feats.mean(0), feats.std(0) + 1e-8
        
        edges = g.get_edgelist()
        ei = torch.tensor(edges, dtype=torch.long).t()
        self.edge_index_orig = torch.cat([ei, ei.flip(0)], dim=1)
        self.edge_index_base = self.edge_index_orig.clone()
        
        self.added_edges = set()
        self.current_shortlist = []
        self._feature_cache = None
        self._cache_valid = False
        self.reset()

    def _compute_node_features(self, g: ig.Graph):
        """
        Compute node features for observation.
        Optimized to avoid expensive computations for large graphs.
        """
        n = g.vcount()
        deg = np.array(g.degree(), dtype=float)
        
        # Closeness is expensive - use approximation for large graphs
        if n <= 100:
            try:
                close = np.array(g.closeness(), dtype=float)
            except:
                close = np.zeros(n, dtype=float)
        else:
            # For large graphs, use degree-based approximation
            # Closeness ≈ degree / (n-1) for well-connected graphs
            close = deg / (n - 1 + 1e-8)
        
        pr = np.array(g.pagerank(), dtype=float)
        core = np.array(g.coreness(), dtype=float)
        clust = np.array(g.transitivity_local_undirected(mode="zero"), dtype=float)
        return np.stack([deg, close, pr, core, clust], axis=1).astype(np.float32)
    
    def _refresh_shortlist(self):
        """Rebuild the candidate shortlist after graph changes."""
        self.current_shortlist = build_candidate_shortlist(
            self.g,
            k=self.shortlist_size,
            rng=self.rng,
            bet_sample_size=BETWEENNESS_SAMPLE_SIZE,
            max_pool=self.max_candidate_pool,
        )
    
    def action_masks(self):
        """Return boolean mask: True = valid action, False = invalid (already added)"""
        masks = np.zeros(self.shortlist_size, dtype=bool)
        for idx, edge in enumerate(self.current_shortlist):
            if edge is None:
                continue
            u, v = edge
            if (u, v) in self.added_edges or self.g.are_connected(u, v):
                masks[idx] = False
            else:
                masks[idx] = True
        return masks

    def _exact_lambda2(self, g: ig.Graph):
        """Exact λ₂ (Algebraic Connectivity) - same as evaluation"""
        n = g.vcount()
        if n < 2:
            return 0.0
        
        try:
            L = np.array(g.laplacian(normalized=True))
            eigs = np.sort(np.linalg.eigvalsh(L))
            lambda2 = eigs[1] if len(eigs) > 1 else 0.0
            return float(lambda2)
        except:
            return 0.0
    
    def _approx_lambda2(self, g: ig.Graph):
        """
        Approximate λ₂ for large graphs using power iteration.
        For graphs >300 nodes, uses iterative method instead of full eigendecomposition.
        """
        n = g.vcount()
        if n < 2:
            return 0.0
        
        # Use exact method for small graphs
        if n <= LARGE_GRAPH_THRESHOLD:
            return self._exact_lambda2(g)
        
        try:
            # For large graphs, use scipy's sparse eigenvalue solver
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            
            # Get Laplacian as sparse matrix
            L = g.laplacian(normalized=True)
            L_sparse = csr_matrix(L)
            
            # Compute only the 2 smallest eigenvalues
            # This is much faster than full eigendecomposition
            eigs = eigsh(L_sparse, k=2, which='SM', return_eigenvectors=False)
            eigs = np.sort(eigs)
            lambda2 = eigs[1] if len(eigs) > 1 else 0.0
            return float(lambda2)
        except:
            # Fallback to exact method if sparse solver fails
            return self._exact_lambda2(g)

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
    
    def _compute_fc_from_graph(self, g: ig.Graph) -> float:
        """
        Compute critical fraction (f_c) from graph.
        Based on Cohen et al. (2000) percolation theory.
        """
        degrees = np.array(g.degree(), dtype=float)
        k_mean = degrees.mean()
        if k_mean <= 1e-12:
            return 0.0
        
        k2_mean = (degrees ** 2).mean()
        kappa = k2_mean / (k_mean + 1e-8)
        
        if kappa <= 2.0:
            return 0.0
        
        fc = 1.0 - 1.0 / (kappa - 1.0 + 1e-8)
        return float(np.clip(fc, 0.0, 1.0))

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
        BC = approximate_betweenness(g, sample_size=BETWEENNESS_SAMPLE_SIZE)
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
        BC = approximate_betweenness(g, sample_size=BETWEENNESS_SAMPLE_SIZE)
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
        
        # FIXED: Restore edge_index_base to original (though GNN no longer uses it)
        self.edge_index_base = self.edge_index_orig.clone()
        
        # Invalidate feature cache
        self._cache_valid = False
        
        # Refresh shortlist
        self._refresh_shortlist()

        # Initialize baselines based on reward type
        if self.reward_type == "pbr":
            # PBR baselines
            self.prev_fc = self._compute_fc_from_graph(self.g)
        
        elif self.reward_type == "effres":
            # EffRes baselines
            self.prev_effres = self._effective_graph_resistance(self.g)
        
        elif self.reward_type == "ivi":
            # IVI baselines - no λ₂ needed
            self.prev_max_ivi = self._ivi_scores(self.g).max()
        
        elif self.reward_type == "nnsi":
            # NNSI baselines - no λ₂ needed
            self.prev_max_nnsi = self._nnsi_scores(self.g).max()

        obs = self._obs()
        return obs, {}

    def _obs(self):
        """
        Get observation with feature caching and progress features.
        Features are cached and only recomputed when graph structure changes.
        """
        if not self._cache_valid:
            feats = self._compute_node_features(self.g)
            node_feats = ((feats - self.mean) / self.std).flatten().astype(np.float32)
            self._feature_cache = node_feats
            self._cache_valid = True
        
        # Add progress features
        budget_used = self.successful_additions / max(1, self.budget)
        valid_frac = self.action_masks().mean()
        
        progress_feats = np.array([budget_used, valid_frac], dtype=np.float32)
        
        return np.concatenate([self._feature_cache, progress_feats])

    def step(self, action):
        action = int(action)
        
        # Invalid slot index
        if action < 0 or action >= len(self.current_shortlist):
            reward = -1.0
            self.steps += 1
            done = self.successful_additions >= self.budget or self.steps >= self.budget * 10
            return self._obs(), float(reward), done, False, {}
        
        edge = self.current_shortlist[action]
        
        # Empty slot
        if edge is None:
            reward = -1.0
            self.steps += 1
            done = self.successful_additions >= self.budget or self.steps >= self.budget * 10
            return self._obs(), float(reward), done, False, {}
        
        u, v = edge
        edge = tuple(sorted((u, v)))
        
        if edge in self.added_edges or self.g.are_connected(u, v):
            reward = -1.0
        else:
            self.g.add_edge(u, v)
            self.added_edges.add(edge)
            self.successful_additions += 1
            self._cache_valid = False
            reward = self._compute_reward()
            self._refresh_shortlist()
        
        self.steps += 1
        done = self.successful_additions >= self.budget or self.steps >= self.budget * 10
        return self._obs(), float(reward), done, False, {}

    def _compute_reward(self):
        """
        Compute reward based on the specific reward type.
        Each reward type computes only the metrics it needs.
        """
        if self.reward_type == "pbr":
            r = self._reward_pbr()
            
        elif self.reward_type == "effres":
            r = self._reward_effres()  # d_lambda2 not used in new version
            
        elif self.reward_type == "ivi":
            # IVI only needs IVI scores - no λ₂ computation needed
            r = self._reward_ivi()
            
        elif self.reward_type == "nnsi":
            # NNSI only needs NNSI scores - no λ₂ computation needed
            r = self._reward_nnsi()
            
        else:
            r = 0.0
        
        # Clip to reasonable range
        r_scaled = float(np.clip(r, -1.0, 1.0))
        
        return r_scaled

    def _reward_pbr(self):
        """
        Percolation-Based Resilience (PBR) - normalized marginal objective.
        """
        curr_fc = self._compute_fc_from_graph(self.g)
        prev_fc = getattr(self, "prev_fc", curr_fc)
        
        r = (curr_fc - prev_fc) / (abs(prev_fc) + 1e-8)
        
        self.prev_fc = curr_fc
        return r
    
    def _reward_effres(self):
        """
        Effective Resistance - normalized marginal objective.
        """
        curr_effres = self._effective_graph_resistance(self.g)
        prev_effres = getattr(self, "prev_effres", curr_effres)
        r = (prev_effres - curr_effres) / (abs(prev_effres) + 1e-8)
        self.prev_effres = curr_effres
        return r
    
    def _reward_ivi(self):
        """
        IVI-based reward - normalized marginal objective.
        """
        curr_ivi_scores = self._ivi_scores(self.g)
        curr_max_ivi = curr_ivi_scores.max()
        prev_max_ivi = getattr(self, "prev_max_ivi", curr_max_ivi)
        r = (prev_max_ivi - curr_max_ivi) / (abs(prev_max_ivi) + 1e-8)
        self.prev_max_ivi = curr_max_ivi
        return r
    
    def _reward_nnsi(self):
        """
        NNSI-based reward - normalized marginal objective.
        """
        curr_nnsi_scores = self._nnsi_scores(self.g)
        curr_max_nnsi = curr_nnsi_scores.max()
        prev_max_nnsi = getattr(self, "prev_max_nnsi", curr_max_nnsi)
        r = (prev_max_nnsi - curr_max_nnsi) / (abs(prev_max_nnsi) + 1e-8)
        self.prev_max_nnsi = curr_max_nnsi
        return r

# ============================================================================
# GNN FEATURE EXTRACTOR - FIXED FOR REPLAY BUFFER
# ============================================================================
class CleanGNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, env, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.n_nodes = env.n
        self.n_features = 5
        self.n_progress = 2
        self.base_edge_index = env.edge_index_orig.clone()
        
        self.gnn = pyg_nn.Sequential("x, edge_index", [
            (pyg_nn.GraphConv(self.n_features, 64), "x, edge_index -> x"),
            nn.ReLU(),
            (pyg_nn.GraphConv(64, 64), "x, edge_index -> x"),
            nn.ReLU(),
            (pyg_nn.GraphConv(64, 32), "x, edge_index -> x"),
        ])
        self.pool = pyg_nn.global_mean_pool
        # graph embedding (32) + progress features (2)
        self.proj = nn.Linear(32 + self.n_progress, features_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b = obs.shape[0]
        node_dim = self.n_nodes * self.n_features
        node_obs = obs[:, :node_dim].contiguous()
        progress_obs = obs[:, node_dim: node_dim + self.n_progress]
        
        x = node_obs.reshape(b, self.n_nodes, self.n_features).reshape(-1, self.n_features)
        edge_index = self.base_edge_index.to(obs.device)
        
        if b > 1:
            edge_indices = []
            for i in range(b):
                edge_indices.append(edge_index + (i * self.n_nodes))
            edge_index = torch.cat(edge_indices, dim=1)
        
        x = self.gnn(x, edge_index)
        batch = torch.arange(b, device=obs.device).repeat_interleave(self.n_nodes)
        x = self.pool(x, batch)
        x = torch.cat([x, progress_obs], dim=1)
        x = self.proj(x)
        return x

# ============================================================================
# EVALUATION METRICS
# ============================================================================
def average_node_connectivity_exact(g: ig.Graph) -> float:
    """
    Compute exact average pairwise node connectivity for small graphs.
    This is the average of vertex_connectivity(i, j) over all pairs.
    """
    n = g.vcount()
    if n < 2:
        return 0.0
    
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                vals.append(g.vertex_connectivity(i, j))
            except:
                pass
    return float(np.mean(vals)) if vals else 0.0

def compute_attack_curve_auc(g: ig.Graph, attack_fractions=None):
    """
    Compute AUC of attack curve (targeted degree-based removal)
    
    Args:
        g: Graph to attack
        attack_fractions: List of fractions to remove (default: 1% to 20%)
    
    Returns:
        AUC value (area under the GCC curve)
    """
    if attack_fractions is None:
        attack_fractions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                           0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    
    n = g.vcount()
    if n < 2:
        return 0.0
    
    gcc_values = [1.0]  # Start at 100% (no attack)
    fractions = [0.0] + attack_fractions
    
    for frac in attack_fractions:
        k = max(1, int(frac * n))
        top_deg = np.argsort(-np.array(g.degree()))[:k]
        gc = g.copy()
        gc.delete_vertices(top_deg)
        
        try:
            gcc = gc.clusters().giant().vcount() / max(1, (n - k))
        except:
            gcc = 0.0
        
        gcc_values.append(gcc)
    
    # Compute AUC using trapezoidal rule
    try:
        auc = np.trapz(gcc_values, fractions)
    except AttributeError:
        # numpy >= 2.0 moved trapz to trapezoid
        from numpy import trapezoid
        auc = trapezoid(gcc_values, fractions)
    
    return float(auc)

def sanitize_value(val, default=0.0):
    """
    Sanitize a value to ensure it's not NaN, inf, or None.
    Returns a valid float or the default value.
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if np.isnan(val) or np.isinf(val):
            return default
        return float(val)
    return default

def exact_metrics(g: ig.Graph):
    """
    Compute comprehensive resilience evaluation metrics.
    All metrics are guaranteed to return valid numeric values (no NaN/inf).
    
    Metrics are organized by category:
    - Connectivity: λ₂, AvgNodeConn, EdgeConn, SpectralGap
    - Robustness: GCC_5%, GCC_10%, AttackCurveAUC, RobustnessCoeff
    - Distance: ASPL, Diameter, Efficiency, ASPLVariance
    - Structure: ArticulationPoints, Bridges, BetCentralization
    - Spectral: NatConnectivity, EffResistance, λ₂/λₙ ratio
    - Topology: Assortativity, AvgClustering, Transitivity
    """
    n = g.vcount()
    if n < 2:
        return {
            # Connectivity metrics
            "λ₂": 0.0, "AvgNodeConn": 0.0, "EdgeConn": 0.0, "SpectralGap": 0.0,
            # Robustness metrics
            "GCC_5%": 1.0, "GCC_10%": 1.0, "AttackCurveAUC": 0.0, "RobustnessCoeff": 0.0,
            # Distance metrics
            "ASPL": 0.0, "Diameter": 0.0, "Efficiency": 0.0, "ASPLVariance": 0.0,
            # Structure metrics
            "ArticulationPoints": 0, "Bridges": 0, "BetCentralization": 0.0,
            # Spectral metrics
            "NatConnectivity": 0.0, "EffResistance": 0.0, "λ₂_λₙ_Ratio": 0.0,
            # Topology metrics
            "Assortativity": 0.0, "AvgClustering": 0.0, "Transitivity": 0.0,
        }

    # λ₂ (Algebraic Connectivity) - use optimized computation for large graphs
    lambda2 = approximate_lambda2(g)
    
    # Compute full Laplacian spectrum for spectral metrics
    try:
        L = np.array(g.laplacian(normalized=True))
        eigs = np.sort(np.linalg.eigvalsh(L))
        lambda_n = eigs[-1] if len(eigs) > 0 else 0.0
        spectral_gap = lambda2  # For normalized Laplacian, gap is λ₂ - λ₁ = λ₂ - 0
        lambda2_lambdan_ratio = lambda2 / (lambda_n + 1e-12)
    except:
        lambda_n = 0.0
        spectral_gap = lambda2
        lambda2_lambdan_ratio = 0.0

    # Average Node Connectivity (pairwise average, consistent across graph sizes)
    if n <= 50:
        # Exact pairwise average for small graphs
        avg_node_conn = average_node_connectivity_exact(g)
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
    
    # Edge Connectivity (minimum cut)
    try:
        edge_conn = g.edge_connectivity()
    except:
        edge_conn = 0.0

    # GCC after 5% attack (backward compatibility)
    k5 = max(1, int(0.05 * n))
    top_deg_5 = np.argsort(-np.array(g.degree()))[:k5]
    gc5 = g.copy()
    gc5.delete_vertices(top_deg_5)
    try:
        gcc_5 = gc5.clusters().giant().vcount() / max(1, (n - k5))
    except:
        gcc_5 = 0.0
    
    # FIXED: GCC after 10% attack (more aggressive)
    # Must recompute top nodes for 10%, not reuse 5% list
    k10 = max(1, int(0.10 * n))
    top_deg_10 = np.argsort(-np.array(g.degree()))[:k10]
    gc10 = g.copy()
    gc10.delete_vertices(top_deg_10)
    try:
        gcc_10 = gc10.clusters().giant().vcount() / max(1, (n - k10))
    except:
        gcc_10 = 0.0
    
    # Attack curve AUC (1% to 20% targeted removals)
    attack_auc = compute_attack_curve_auc(g)
    
    # FIXED: Robustness Coefficient R (Schneider et al. 2011)
    # R = 1/N * sum of relative size of largest component during attack
    # Higher R = more robust
    # FIX: Delete vertices by always targeting highest degree in CURRENT graph
    try:
        robustness_coeff = 0.0
        g_temp = g.copy()
        
        for i in range(n):
            try:
                largest_comp_size = g_temp.clusters().giant().vcount()
                robustness_coeff += largest_comp_size / n
            except:
                pass
            
            # FIXED: Recompute highest-degree node in current graph
            # (vertex IDs change after deletion)
            if g_temp.vcount() > 0:
                current_degrees = np.array(g_temp.degree())
                highest_deg_node = int(np.argmax(current_degrees))
                g_temp.delete_vertices([highest_deg_node])
        
        robustness_coeff /= n
    except:
        robustness_coeff = 0.0

    # ASPL & Diameter - use approximations for large graphs
    aspl = approximate_avg_path_length(g)
    diam = approximate_diameter(g)
    
    # Network Efficiency (Latora & Marchiori 2001)
    # E = 1/(n(n-1)) * sum(1/d_ij) where d_ij is shortest path
    try:
        if n <= 100:
            # Exact for small graphs
            efficiency = 0.0
            for i in range(n):
                paths = g.shortest_paths(source=i)[0]
                for j in range(n):
                    if i != j and paths[j] > 0 and paths[j] < float('inf'):
                        efficiency += 1.0 / paths[j]
            efficiency /= (n * (n - 1))
        else:
            # Sample-based for large graphs
            sample_size = min(50, n)
            sample_nodes = np.random.choice(n, size=sample_size, replace=False)
            efficiency = 0.0
            count = 0
            for i in sample_nodes:
                paths = g.shortest_paths(source=i)[0]
                for j in sample_nodes:
                    if i != j and paths[j] > 0 and paths[j] < float('inf'):
                        efficiency += 1.0 / paths[j]
                        count += 1
            efficiency = efficiency / count if count > 0 else 0.0
    except:
        efficiency = 0.0
    
    # ASPL Variance (measure of path length heterogeneity)
    try:
        if n <= 100:
            all_paths = []
            for i in range(n):
                paths = g.shortest_paths(source=i)[0]
                for j in range(i + 1, n):
                    if paths[j] > 0 and paths[j] < float('inf'):
                        all_paths.append(paths[j])
            aspl_variance = float(np.var(all_paths)) if all_paths else 0.0
        else:
            # Sample-based
            sample_size = min(50, n)
            sample_nodes = np.random.choice(n, size=sample_size, replace=False)
            all_paths = []
            for i in sample_nodes:
                paths = g.shortest_paths(source=i)[0]
                for j in sample_nodes:
                    if i != j and paths[j] > 0 and paths[j] < float('inf'):
                        all_paths.append(paths[j])
            aspl_variance = float(np.var(all_paths)) if all_paths else 0.0
    except:
        aspl_variance = 0.0

    # Articulation points & bridges
    art_pts = len(g.articulation_points())
    bridges = len(g.bridges())

    # Betweenness centralization - use approximation for large graphs
    bet = approximate_betweenness(g, sample_size=BETWEENNESS_SAMPLE_SIZE)
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
    
    # Global Transitivity (different from avg clustering)
    try:
        transitivity = g.transitivity_undirected()
    except:
        transitivity = 0.0

    return {
        # Connectivity metrics
        "λ₂": sanitize_value(lambda2, 0.0),
        "AvgNodeConn": sanitize_value(avg_node_conn, 0.0),
        "EdgeConn": sanitize_value(edge_conn, 0.0),
        "SpectralGap": sanitize_value(spectral_gap, 0.0),
        
        # Robustness metrics
        "GCC_5%": sanitize_value(gcc_5, 0.0),
        "GCC_10%": sanitize_value(gcc_10, 0.0),
        "AttackCurveAUC": sanitize_value(attack_auc, 0.0),
        "RobustnessCoeff": sanitize_value(robustness_coeff, 0.0),
        
        # Distance metrics
        "ASPL": sanitize_value(aspl, 0.0),
        "Diameter": sanitize_value(diam, 0.0),
        "Efficiency": sanitize_value(efficiency, 0.0),
        "ASPLVariance": sanitize_value(aspl_variance, 0.0),
        
        # Structure metrics
        "ArticulationPoints": int(art_pts) if not np.isnan(art_pts) else 0,
        "Bridges": int(bridges) if not np.isnan(bridges) else 0,
        "BetCentralization": sanitize_value(bet_central, 0.0),
        
        # Spectral metrics
        "NatConnectivity": sanitize_value(natconn, 0.0),
        "EffResistance": sanitize_value(eff_res, 1e9),
        "λ₂_λₙ_Ratio": sanitize_value(lambda2_lambdan_ratio, 0.0),
        
        # Topology metrics
        "Assortativity": sanitize_value(assort, 0.0),
        "AvgClustering": sanitize_value(avg_clust, 0.0),
        "Transitivity": sanitize_value(transitivity, 0.0),
    }

# ============================================================================
# MAIN EXPERIMENT LOOP
# ============================================================================
print(f"\nTesting reward functions on {len(GRAPH_FILES)} networks")
print(f"Total experiments: {len(GRAPH_FILES) * 4} ({len(GRAPH_FILES)} networks × 4 reward functions)")
print(f"{'='*80}\n")

results = []

# FIXED: Evaluation rollout trace - logs every attempt during policy evaluation
# Includes both valid and invalid actions, with WasValid and WasAdded flags
# This is NOT training-time logging - it's the final deterministic rollout
attempt_records = []

# Reward landscape: DISABLED to prevent memory issues on large networks
# all_reward_landscapes = []

network_times = []

metrics = [
    # Connectivity metrics
    "λ₂", "AvgNodeConn", "EdgeConn", "SpectralGap",
    # Robustness metrics
    "GCC_5%", "GCC_10%", "AttackCurveAUC", "RobustnessCoeff",
    # Distance metrics
    "ASPL", "Diameter", "Efficiency", "ASPLVariance",
    # Structure metrics
    "ArticulationPoints", "Bridges", "BetCentralization",
    # Spectral metrics
    "NatConnectivity", "EffResistance", "λ₂_λₙ_Ratio",
    # Topology metrics
    "Assortativity", "AvgClustering", "Transitivity",
]

# ============================================================================
# CHECKPOINT SYSTEM - Load previous progress if exists
# ============================================================================
# Allow output filenames to be overridden via environment variables
checkpoint_file = os.environ.get('CHECKPOINT_FILE', "./checkpoint_progress.csv")
output_metrics_file = os.environ.get('OUTPUT_METRICS_FILE', "./results_network_metrics.csv")
output_edges_all_file = os.environ.get('OUTPUT_EDGES_ALL_FILE', "./results_evaluation_attempts.csv")
output_edges_added_file = os.environ.get('OUTPUT_EDGES_ADDED_FILE', "./results_evaluation_successful.csv")
# output_landscape_file = "./results_reward_landscape.csv"  # DISABLED

processed_networks = set()

if os.path.exists(checkpoint_file):
    print(f"{'='*80}")
    print(f"CHECKPOINT FOUND - Loading previous progress")
    print(f"{'='*80}\n")
    
    try:
        # Load checkpoint
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_networks = set(checkpoint_df['Graph'].values)
        
        print(f"Found {len(processed_networks)} already processed networks:")
        for net in sorted(processed_networks):
            print(f"  - {net}")
        print(f"\nResuming from network {len(processed_networks) + 1}/{len(GRAPH_FILES)}\n")
        
        # Load existing results with error handling
        if os.path.exists(output_metrics_file):
            try:
                # Check file size first
                file_size = os.path.getsize(output_metrics_file)
                print(f"Checking {output_metrics_file} (size: {file_size} bytes)")
                
                if file_size == 0:
                    print(f"Warning: {output_metrics_file} is empty (0 bytes), starting fresh results")
                    results = []
                else:
                    df_temp = pd.read_csv(output_metrics_file)
                    if len(df_temp) == 0:
                        print(f"Warning: {output_metrics_file} has no data rows, starting fresh results")
                        results = []
                    elif len(df_temp.columns) == 0:
                        print(f"Warning: {output_metrics_file} has no columns, starting fresh results")
                        results = []
                    else:
                        results = df_temp.to_dict('records')
                        print(f"Loaded {len(results)} existing network results")
            except pd.errors.EmptyDataError:
                print(f"Warning: {output_metrics_file} is empty, starting fresh results")
                results = []
            except Exception as e:
                print(f"Warning: Could not load {output_metrics_file}: {e}")
                print(f"Starting fresh results")
                results = []
        else:
            print(f"{output_metrics_file} does not exist, will create new")
            results = []
        
        if os.path.exists(output_edges_all_file):
            try:
                file_size = os.path.getsize(output_edges_all_file)
                print(f"Checking {output_edges_all_file} (size: {file_size} bytes)")
                
                if file_size == 0:
                    print(f"Warning: {output_edges_all_file} is empty, starting fresh")
                    attempt_records = []
                else:
                    df_temp = pd.read_csv(output_edges_all_file)
                    if len(df_temp) > 0 and len(df_temp.columns) > 0:
                        attempt_records = df_temp.to_dict('records')
                        print(f"Loaded {len(attempt_records)} existing edge attempts")
                    else:
                        print(f"Warning: {output_edges_all_file} is empty, starting fresh")
                        attempt_records = []
            except pd.errors.EmptyDataError:
                print(f"Warning: {output_edges_all_file} is empty, starting fresh")
                attempt_records = []
            except Exception as e:
                print(f"Warning: Could not load {output_edges_all_file}: {e}")
                attempt_records = []
        else:
            print(f"{output_edges_all_file} does not exist, will create new")
            attempt_records = []
        
        # Reward landscape loading DISABLED (feature removed to prevent memory issues)
                
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Starting fresh")
        processed_networks = set()
        results = []
        attempt_records = []
else:
    print(f"No checkpoint found - Starting fresh\n")

for idx, path in enumerate(tqdm(GRAPH_FILES, desc="Overall Progress"), 1):
    network_start = time.time()
    name = os.path.basename(path).split(".")[0]
    
    # ============================================================================
    # CHECKPOINT: Skip if already processed
    # ============================================================================
    if name in processed_networks:
        print(f"\n{'='*80}")
        print(f"Network {idx}/{len(GRAPH_FILES)}: {name} - SKIPPING (already processed)")
        print(f"{'='*80}")
        continue
    
    try:
        orig_g = ig.Graph.Read_GraphML(path).as_undirected()
        n, m = orig_g.vcount(), orig_g.ecount()

        print(f"\n{'='*80}")
        print(f"Network {idx}/{len(GRAPH_FILES)}: {name} (N={n}, M={m})")
        if n > LARGE_GRAPH_THRESHOLD:
            print(f"  Note: Large graph detected - using approximations for efficiency")
        print(f"{'='*80}")

        # Original metrics
        orig_met = exact_metrics(orig_g)
        row = {"Graph": name, "N": n, "M": m}
        for k, v in orig_met.items():
            row[f"Orig_{k}"] = v
    
    except Exception as e:
        print(f"  ERROR loading or processing {name}: {e}")
        print(f"  Skipping this network...")
        continue

    # Budget
    B = edge_budget(n, m)
    row["BudgetEdges"] = B
    print(f"Budget: {B} edges")
    
    # Test all four reward functions
    for reward_idx, reward_type in enumerate(["pbr", "effres", "ivi", "nnsi"], 1):
        print(f"  [{reward_idx}/4] {reward_type.upper():10s} - Training...", end=" ", flush=True)
        
        try:
            env = ResilienceEnv(
                path,
                reward_type=reward_type,
                budget_edges=B,
                gamma=5.0,
                beta=1.0,
                delta=1.0,
                shortlist_size=min(512, max(128, 16 * B)),
                max_candidate_pool=5000,
            )
            
            policy_kwargs = dict(
                features_extractor_class=CleanGNNExtractor,
                features_extractor_kwargs=dict(env=env),
                net_arch=[256, 256],
            )
            
            model = MaskablePPO(
                "MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                clip_range=0.2,
                max_grad_norm=None,  # Disable gradient clipping to avoid PyTorch bug
                device=DEVICE,
                verbose=0,
                seed=SEED,
            )
            
            # Training with progress - 50k steps
            print("Training 50k steps...", end=" ", flush=True)
            model.learn(total_timesteps=50000)
            print("Evaluating...", end=" ", flush=True)
            
            # Evaluation with action masking
            obs, _ = env.reset()
            rollout_attempts = []
            done = False
            attempt_num = 0
            
            print(f"Starting rollout (budget={env.budget})...", end=" ", flush=True)
            
            while not done:
                # Get action masks for MaskablePPO
                action_masks = get_action_masks(env)
                
                # If no valid actions left, stop
                if not action_masks.any():
                    break
                
                # Get model's action with masking
                raw_action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
                raw_action = int(raw_action)
                
                # Check if action is valid (should always be true with masking)
                was_valid = bool(action_masks[raw_action])
                
                # Get edge info from shortlist
                if raw_action < len(env.current_shortlist):
                    edge_tuple = env.current_shortlist[raw_action]
                    if edge_tuple is not None:
                        u, v = edge_tuple
                    else:
                        u, v = -1, -1  # Empty slot
                else:
                    u, v = -1, -1  # Invalid index
                
                # Track if edge was actually added
                prev_added_count = len(env.added_edges)
                
                # Execute action
                obs, reward, done, _, _ = env.step(raw_action)
                
                # Check if edge was added
                was_added = len(env.added_edges) > prev_added_count
                
                attempt_num += 1
                
                # Log attempt with full information
                rollout_attempts.append({
                    "Graph": name,
                    "Reward": reward_type.upper(),
                    "Attempt": attempt_num,
                    "RawAction": raw_action,
                    "Node_U": int(u),
                    "Node_V": int(v),
                    "Edge": f"{u}-{v}",
                    "WasValid": was_valid,
                    "WasAdded": was_added,
                    "StepReward": float(reward),
                })
                
                # Safety limit
                if attempt_num >= env.budget * 10:
                    break
            
            # Store attempt trace for this rollout
            attempt_records.extend(rollout_attempts)
            
            # Evaluate final graph
            final_met = exact_metrics(env.g)
            prefix = reward_type.upper()
            for k, v in final_met.items():
                row[f"{prefix}_{k}"] = v
            
            print(f"✓ ({attempt_num} attempts, {len(env.added_edges)} edges added)")
            
            # CRITICAL: Delete model and env to free memory immediately
            del model
            del env
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            for k in metrics:
                row[f"{reward_type.upper()}_{k}"] = np.nan
            
            # Save checkpoint even on error
            print(f"  Saving checkpoint after error...", end=" ", flush=True)
            df_temp = pd.DataFrame(results + [row])
            df_temp.to_csv(output_metrics_file, index=False)
            print(f"Done")

    network_time = time.time() - network_start
    network_times.append(network_time)
    avg_time = np.mean(network_times)
    remaining = len(GRAPH_FILES) - idx
    eta_minutes = (avg_time * remaining) / 60
    
    print(f"  Network completed in {network_time/60:.1f} min | Avg: {avg_time/60:.1f} min | ETA: {eta_minutes:.1f} min ({eta_minutes/60:.1f}h)")
    results.append(row)
    
    # ============================================================================
    # CHECKPOINT: Save progress after each network
    # ============================================================================
    print(f"  Saving checkpoint...", end=" ", flush=True)
    
    # Save main results
    df_temp = pd.DataFrame(results)
    df_temp.to_csv(output_metrics_file, index=False)
    
    # Save edges
    if len(attempt_records) > 0:
        df_edges_temp = pd.DataFrame(attempt_records)
        df_edges_temp.to_csv(output_edges_all_file, index=False)
        
        # Save added-only edges
        df_edges_added_temp = df_edges_temp[df_edges_temp["WasAdded"] == True].copy()
        df_edges_added_temp.to_csv(output_edges_added_file, index=False)
    
    # Update checkpoint file
    processed_networks.add(name)
    checkpoint_df = pd.DataFrame({'Graph': list(processed_networks)})
    checkpoint_df.to_csv(checkpoint_file, index=False)
    
    # CRITICAL: Force memory cleanup after each network to prevent crashes
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Done ({len(processed_networks)}/{len(GRAPH_FILES)} networks)")
    
    # Force garbage collection to free memory
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

df = pd.DataFrame(results)

# ============================================================================
# COMPUTE IMPROVEMENTS WITH CORRECT SEMANTICS
# ============================================================================
# FIXED: Account for "lower is better" vs "higher is better" metrics
# Positive improvement % always means "better"

METRIC_DIRECTIONS = {
    # Higher is better
    'λ₂': 'higher',
    'AvgNodeConn': 'higher',
    'EdgeConn': 'higher',
    'SpectralGap': 'higher',
    'GCC_5%': 'higher',
    'GCC_10%': 'higher',
    'AttackCurveAUC': 'higher',
    'RobustnessCoeff': 'higher',
    'Efficiency': 'higher',
    'NatConnectivity': 'higher',
    'λ₂_λₙ_Ratio': 'higher',
    'AvgClustering': 'higher',
    'Transitivity': 'higher',
    
    # Lower is better
    'ASPL': 'lower',
    'Diameter': 'lower',
    'ASPLVariance': 'lower',
    'ArticulationPoints': 'lower',
    'Bridges': 'lower',
    'BetCentralization': 'lower',
    'EffResistance': 'lower',
    
    # Special: negative is better (onion structure)
    'Assortativity': 'negative',
}

for k in metrics:
    orig_col = f"Orig_{k}"
    if orig_col not in df.columns:
        continue
    
    direction = METRIC_DIRECTIONS.get(k, 'higher')
    
    for r in ["PBR", "EFFRES", "IVI", "NNSI"]:
        col = f"{r}_{k}"
        if col not in df.columns:
            continue
        
        if direction == 'higher':
            # Higher is better: positive % = improvement
            improvement = ((df[col] - df[orig_col]) / (df[orig_col].abs() + 1e-8) * 100)
        elif direction == 'lower':
            # Lower is better: positive % = improvement (flip sign)
            improvement = ((df[orig_col] - df[col]) / (df[orig_col].abs() + 1e-8) * 100)
        elif direction == 'negative':
            # More negative is better: positive % = more negative
            improvement = ((df[orig_col] - df[col]) / (df[orig_col].abs() + 1e-8) * 100)
        
        df[f"%Δ_{r}_vs_Orig_{k}"] = improvement.round(2)

# PBR vs EffRes comparison (keep as raw difference for now)
for k in metrics:
    if f"PBR_{k}" in df.columns and f"EFFRES_{k}" in df.columns:
        df[f"%Δ_PBR_vs_EFFRES_{k}"] = (
            (df[f"PBR_{k}"] - df[f"EFFRES_{k}"]) / (df[f"EFFRES_{k}"].abs() + 1e-8) * 100
        ).round(2)

# ============================================================================
# SAVE RESULTS (Final)
# ============================================================================
# File paths already defined in checkpoint section above

df.to_csv(output_metrics_file, index=False)

# Save evaluation attempt trace
print(f"\nSaving evaluation attempt records... ({len(attempt_records)} records)")
df_edges = pd.DataFrame(attempt_records)
print(f"Created DataFrame with {len(df_edges)} rows")
df_edges.to_csv(output_edges_all_file, index=False)
print(f"Saved to: {output_edges_all_file}")

# Save successful attempts only (edges that were actually added)
print(f"Filtering for successfully added edges...")
if len(df_edges) > 0 and "WasAdded" in df_edges.columns:
    df_edges_added = df_edges[df_edges["WasAdded"] == True].copy()
    print(f"Found {len(df_edges_added)} successfully added edges")
else:
    df_edges_added = pd.DataFrame()
    print(f"No edge attempts recorded (empty DataFrame)")
df_edges_added.to_csv(output_edges_added_file, index=False)
print(f"Saved to: {output_edges_added_file}")

print(f"\n{'='*80}")
print(f"FILES SAVED:")
print(f"{'='*80}")
print(f"Metrics: {output_metrics_file}")
print(f"All edge attempts: {output_edges_all_file}")
print(f"Successful edges: {output_edges_added_file}")
# print(f"Reward landscape: {output_landscape_file}")
print(f"{'='*80}")

# ============================================================================
# CLEANUP: Remove checkpoint file on successful completion
# ============================================================================
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print(f"\nCheckpoint file removed (all networks processed successfully)")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
end_time = time.time()
total_time = end_time - start_time
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)

print(f"\n{'='*80}")
print(f"EXPERIMENT COMPLETE")
print(f"{'='*80}")
print(f"\nResults saved to: {output_metrics_file}")
print(f"Edges saved to: {output_edges_all_file}")
print(f"Added-only edges saved to: {output_edges_added_file}")
print(f"Total time: {hours}h {minutes}m")
print(f"Networks: {len(df)}")
print(f"Experiments: {len(df) * 4}")
print(f"Total attempts tracked: {len(df_edges)}")
print(f"Total edges added tracked: {len(df_edges_added)}")

# Display sample results
print(f"\n{'='*80}")
print(f"SAMPLE RESULTS (first 5 networks)")
print(f"{'='*80}\n")
display_cols = ["Graph", "N", "M", "BudgetEdges"]
for k in ["λ₂", "AvgNodeConn", "GCC_5%", "NatConnectivity", "EffResistance"]:
    display_cols += [f"PBR_{k}", f"EFFRES_{k}", f"IVI_{k}", f"NNSI_{k}"]

print(df[display_cols].head().to_string(index=False))

print(f"\n{'='*80}")
print(f"Full results in: {output_metrics_file}")
print(f"{'='*80}")
print(f"\nEXPERIMENT FINISHED")
print(f"{'='*80}\n")

# Explicitly exit to prevent re-running
print("Exiting script")
print(f"Final PID: {os.getpid()}")
import sys
sys.exit(0)
