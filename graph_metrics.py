#!/usr/bin/env python3
"""
OUTPUtS:
    graph_metrics_summary.csv
    node_feature_summary.csv
"""

from __future__ import annotations

import os
import glob
import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import igraph as ig
except ImportError as e:
    raise SystemExit(
        "igraph is required. Install with: pip install igraph"
    ) from e


# ----------------------------
# Config
# ----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# If a metric can't be computed (e.g., disconnected for effective resistance), use these:
BIG_PENALTY = 1e9
NATCONN_N_MAX = 600  # skip natural connectivity if graph bigger than this (dense eig is heavy)
EFFRES_N_MAX = 800   # skip effective resistance if too large (dense eig is heavy)

# Average pairwise node connectivity:
# - exact for N <= EXACT_CONN_N_MAX (all pairs)
# - else approximate by sampling nodes and pairs
EXACT_CONN_N_MAX = 50
SAMPLED_NODES_MAX = 50
SAMPLED_PAIRS_MAX = 400  # number of sampled pairs for large graphs


# ----------------------------
# Helpers
# ----------------------------
def find_dataset_folder() -> str:
    """Find dataset folder, supporting both correct and typo names."""
    candidates = ["real_world_topologies", "real_worl_topologies"]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"Could not find dataset folder. Expected one of: {candidates}"
    )


def read_graph_graphml(path: str) -> ig.Graph:
    """Read GraphML and convert to simple undirected graph."""
    g = ig.Graph.Read_GraphML(path)

    # Convert to undirected (ITZ graphs are often undirected, but we enforce it)
    if g.is_directed():
        g = g.as_undirected(combine_edges=None)

    # Remove loops and multi-edges to make it "simple"
    if g.has_multiple():
        g.simplify(multiple=True, loops=False)
    else:
        # still remove loops if any
        g.simplify(multiple=False, loops=True)

    # Ensure vertex names exist (some GraphML use "id" etc.)
    # Not strictly needed, but nice for debugging
    if "name" not in g.vs.attributes():
        g.vs["name"] = [str(i) for i in range(g.vcount())]

    return g


def gcc_subgraph(g: ig.Graph) -> ig.Graph:
    """Return the giant connected component as an induced subgraph."""
    if g.vcount() == 0:
        return g
    comps = g.components(mode="weak")
    if len(comps) == 0:
        return g
    giant = max(comps, key=len)
    return g.induced_subgraph(giant)


def safe_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def degree_gini(deg: np.ndarray) -> float:
    """Gini coefficient for a non-negative array."""
    if deg.size == 0:
        return float("nan")
    x = np.sort(deg.astype(float))
    if np.allclose(x, 0):
        return 0.0
    n = x.size
    cumx = np.cumsum(x)
    # Gini = (n+1 - 2 * sum(cumx)/cumx[-1]) / n
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def normalized_laplacian_eigs(g: ig.Graph) -> Optional[np.ndarray]:
    """
    Compute eigenvalues of the normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}
    using a dense adjacency matrix. Returns eigvals sorted ascending, or None on failure.
    """
    n = g.vcount()
    if n == 0:
        return np.array([], dtype=float)

    try:
        A = np.array(g.get_adjacency().data, dtype=float)
        deg = A.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(inv_sqrt)
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        eigs = np.linalg.eigvalsh(L)
        eigs.sort()
        return eigs
    except Exception:
        return None


def algebraic_connectivity_lambda2(g: ig.Graph) -> float:
    """
    Exact lambda2 from normalized Laplacian eigenvalues.
    For disconnected graphs, lambda2 may be 0.
    """
    eigs = normalized_laplacian_eigs(g)
    if eigs is None or eigs.size < 2:
        return float("nan")
    return float(eigs[1])


def effective_resistance_kirchhoff(g: ig.Graph) -> float:
    """
    Effective graph resistance / Kirchhoff index using Laplacian eigenvalues:
        R_G = n * sum_{i=2..n} 1/lambda_i  (for connected graphs, lambda_1=0)
    For disconnected graphs, we return BIG_PENALTY (as in the thesis script style).
    Uses normalized Laplacian eigenvalues, consistent with lambda2 computation.
    """
    n = g.vcount()
    if n <= 1:
        return 0.0
    if not g.is_connected():
        return float(BIG_PENALTY)
    if n > EFFRES_N_MAX:
        return float("nan")  # too expensive dense eig for very large graphs

    eigs = normalized_laplacian_eigs(g)
    if eigs is None or eigs.size != n:
        return float("nan")

    # skip the first eigenvalue (0)
    lam = eigs[1:]
    # Avoid division by zero
    if np.any(lam <= 1e-12):
        return float(BIG_PENALTY)
    return float(n * np.sum(1.0 / lam))


def natural_connectivity(g: ig.Graph) -> float:
    """
    Natural connectivity:
        \bar{lambda} = log( (1/n) * sum_i exp(mu_i) )
    where mu_i are eigenvalues of adjacency matrix.
    """
    n = g.vcount()
    if n == 0:
        return float("nan")
    if n > NATCONN_N_MAX:
        return float("nan")

    try:
        A = np.array(g.get_adjacency().data, dtype=float)
        mu = np.linalg.eigvalsh(A)
        val = math.log(np.mean(np.exp(mu)))
        return float(val)
    except Exception:
        return float("nan")


def avg_pairwise_vertex_connectivity(g: ig.Graph) -> float:
    """
    Average pairwise vertex connectivity.
    - Exact over all unordered pairs if N <= EXACT_CONN_N_MAX
    - Else approximate via sampling nodes and pairs
    """
    n = g.vcount()
    if n <= 1:
        return 0.0

    # Work on GCC to avoid degenerate behavior on disconnected graphs
    gg = gcc_subgraph(g)
    n2 = gg.vcount()
    if n2 <= 1:
        return 0.0

    vertices = list(range(n2))

    try:
        if n2 <= EXACT_CONN_N_MAX:
            vals = []
            for i in range(n2):
                for j in range(i + 1, n2):
                    vals.append(gg.vertex_connectivity(i, j))
            return float(np.mean(vals)) if vals else 0.0

        # approximate
        sample_nodes = random.sample(vertices, k=min(SAMPLED_NODES_MAX, n2))
        pairs = []
        for _ in range(SAMPLED_PAIRS_MAX):
            u, v = random.sample(sample_nodes, 2)
            pairs.append((u, v))

        vals = [gg.vertex_connectivity(u, v) for (u, v) in pairs]
        return float(np.mean(vals)) if vals else 0.0

    except Exception:
        return float("nan")


def betweenness_centralization(g: ig.Graph) -> float:
    """
    Betweenness centralization (Freeman).
    igraph provides centralization with centralization_betweenness().
    """
    try:
        return float(g.centralization_betweenness(directed=False, normalized=True))
    except Exception:
        return float("nan")


def compute_node_features(g: ig.Graph) -> Dict[str, np.ndarray]:
    """Compute node features used as RL state in your thesis script."""
    deg = np.array(g.degree(), dtype=float)
    try:
        clo = np.array(g.closeness(), dtype=float)
    except Exception:
        clo = np.full(g.vcount(), np.nan)

    try:
        pr = np.array(g.pagerank(directed=False), dtype=float)
    except Exception:
        pr = np.full(g.vcount(), np.nan)

    try:
        core = np.array(g.coreness(), dtype=float)
    except Exception:
        core = np.full(g.vcount(), np.nan)

    try:
        clust = np.array(g.transitivity_local_undirected(mode="zero"), dtype=float)
    except Exception:
        clust = np.full(g.vcount(), np.nan)

    return {
        "degree": deg,
        "closeness": clo,
        "pagerank": pr,
        "coreness": core,
        "clustering": clust,
    }


def summarize_feature(x: np.ndarray) -> Dict[str, float]:
    """Return mean/std/max/p90 for an array, handling NaNs."""
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "p90": float("nan")}
    # ignore NaNs
    x2 = x[np.isfinite(x)]
    if x2.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(x2)),
        "std": float(np.std(x2)),
        "max": float(np.max(x2)),
        "p90": safe_percentile(x2, 90),
    }


# ----------------------------
# Main metric computation per graph
# ----------------------------
def compute_graph_metrics(g: ig.Graph) -> Dict[str, float]:
    n = g.vcount()
    m = g.ecount()
    avg_deg = (2.0 * m / n) if n > 0 else float("nan")
    density = g.density(loops=False) if n > 1 else float("nan")

    comps = g.components(mode="weak")
    n_components = len(comps)
    gcc = gcc_subgraph(g)
    gcc_frac = (gcc.vcount() / n) if n > 0 else float("nan")

    # Use GCC for path-based metrics to avoid errors on disconnected graphs
    aspl = float("nan")
    diameter = float("nan")
    if gcc.vcount() > 1:
        try:
            aspl = float(gcc.average_path_length(directed=False))
        except Exception:
            aspl = float("nan")
        try:
            diameter = float(gcc.diameter(directed=False))
        except Exception:
            diameter = float("nan")
    elif gcc.vcount() == 1:
        aspl = 0.0
        diameter = 0.0

    # Clustering + assortativity
    try:
        avg_clust = float(g.transitivity_undirected(mode="zero"))
    except Exception:
        avg_clust = float("nan")
    try:
        assort = float(g.assortativity_degree(directed=False))
    except Exception:
        assort = float("nan")

    # Centralization and inequality
    bet_cent = betweenness_centralization(g)
    deg = np.array(g.degree(), dtype=float)
    deg_gini = degree_gini(deg)

    # Spectral robustness
    lam2 = algebraic_connectivity_lambda2(g)
    eff_res = effective_resistance_kirchhoff(g)
    nat_conn = natural_connectivity(g)

    # Connectivity metric
    avg_node_conn = avg_pairwise_vertex_connectivity(g)

    # Bridges & articulation points
    try:
        bridges = len(g.bridges())
    except Exception:
        bridges = float("nan")
    try:
        art_points = len(g.articulation_points())
    except Exception:
        art_points = float("nan")

    return {
        "N": n,
        "M": m,
        "avg_degree": avg_deg,
        "density": density,
        "n_components": n_components,
        "gcc_fraction": gcc_frac,
        "aspl_gcc": aspl,
        "diameter_gcc": diameter,
        "avg_clustering": avg_clust,
        "assortativity_degree": assort,
        "betweenness_centralization": bet_cent,
        "degree_gini": deg_gini,
        "lambda2_normlap": lam2,
        "effective_resistance": eff_res,
        "natural_connectivity": nat_conn,
        "avg_pairwise_vertex_connectivity": avg_node_conn,
        "n_bridges": bridges,
        "n_articulation_points": art_points,
    }


def main() -> None:
    folder = find_dataset_folder()
    graphml_files = sorted(glob.glob(os.path.join(folder, "*.graphml")))

    if not graphml_files:
        raise FileNotFoundError(f"No .graphml files found in folder: {folder}")

    graph_rows: List[Dict[str, object]] = []
    feat_rows: List[Dict[str, object]] = []

    for path in graphml_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            g = read_graph_graphml(path)
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue

        # Graph-level metrics
        gm = compute_graph_metrics(g)
        gm_row = {"graph": name, "file": os.path.basename(path)}
        gm_row.update(gm)
        graph_rows.append(gm_row)

        # Node feature summaries
        feats = compute_node_features(g)
        feat_row: Dict[str, object] = {"graph": name, "file": os.path.basename(path), "N": g.vcount(), "M": g.ecount()}
        for k, arr in feats.items():
            s = summarize_feature(arr)
            feat_row[f"{k}_mean"] = s["mean"]
            feat_row[f"{k}_std"] = s["std"]
            feat_row[f"{k}_max"] = s["max"]
            feat_row[f"{k}_p90"] = s["p90"]
        feat_rows.append(feat_row)

        print(f"[OK] {name}: N={g.vcount()}, M={g.ecount()}")

    df_graph = pd.DataFrame(graph_rows).sort_values(by=["N", "M", "graph"], ascending=[True, True, True])
    df_feat = pd.DataFrame(feat_rows).sort_values(by=["N", "M", "graph"], ascending=[True, True, True])

    out_graph = "graph_metrics_summary.csv"
    out_feat = "node_feature_summary.csv"

    df_graph.to_csv(out_graph, index=False)
    df_feat.to_csv(out_feat, index=False)

    print("\nSaved:")
    print(f" - {out_graph} ({len(df_graph)} graphs)")
    print(f" - {out_feat} ({len(df_feat)} graphs)")
    print(f"Dataset folder used: {folder}")


if __name__ == "__main__":
    main()