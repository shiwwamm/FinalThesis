#!/usr/bin/env python3
"""
OUTPUT:

1) graph_metrics_summary.csv
2) node_feature_summary.csv

node_feature_summary.csv includes per-graph summaries for each node feature:
min, median, max, mean, std, p90
"""

from __future__ import annotations

import os
import math
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import igraph as ig
except ImportError as e:
    raise SystemExit("igraph is required. Install with: pip install igraph") from e


# ----------------------------
# Topology list (ONLY these)
# ----------------------------
topo: Dict[str, str] = {
    # Small networks (≤ 40 nodes) - 10 networks
    "Arpanet19706": "Arpanet19706.graphml",
    "Abilene": "Abilene.graphml",
    "Cesnet1997": "Cesnet1997.graphml",
    "Eenet": "Eenet.graphml",
    "Claranet": "Claranet.graphml",
    "Atmnet": "Atmnet.graphml",
    "Belnet2005": "Belnet2005.graphml",
    "Bbnplanet": "Bbnplanet.graphml",
    "Arpanet19728": "Arpanet19728.graphml",
    "Bics": "Bics.graphml",
    # Medium networks (41-93 nodes) - 10 networks
    "Cernet": "Cernet.graphml",
    "Chinanet": "Chinanet.graphml",
    "Cesnet200706": "Cesnet200706.graphml",
    "Surfnet": "Surfnet.graphml",
    "Garr200902": "Garr200902.graphml",
    "Dfn": "Dfn.graphml",
    "Internode": "Internode.graphml",
    "Esnet": "Esnet.graphml",
    "Tw": "Tw.graphml",
    "VtlWavenet2011": "VtlWavenet2011.graphml",
    # Large networks (> 93 nodes) - 10 networks
    "Interoute": "Interoute.graphml",
    "Deltacom": "Deltacom.graphml",
    "Ion": "Ion.graphml",
    "Pern": "Pern.graphml",
    "TataNld": "TataNld.graphml",
    "GtsCe": "GtsCe.graphml",
    "Colt": "Colt.graphml",
    "UsCarrier": "UsCarrier.graphml",
    "DialtelecomCz": "DialtelecomCz.graphml",
    "Cogentco": "Cogentco.graphml",
}

# ----------------------------
# Config
# ----------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BIG_PENALTY = 1e9
NATCONN_N_MAX = 600
EFFRES_N_MAX = 800

EXACT_CONN_N_MAX = 50
SAMPLED_NODES_MAX = 50
SAMPLED_PAIRS_MAX = 400


# ----------------------------
# Helpers
# ----------------------------
def find_dataset_folder() -> str:
    candidates = ["real_world_topologies", "real_worl_topologies"]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"Could not find dataset folder. Expected one of: {candidates}")


def read_graph_graphml(path: str) -> ig.Graph:
    g = ig.Graph.Read_GraphML(path)

    if g.is_directed():
        g = g.as_undirected(combine_edges=None)

    if g.has_multiple():
        g.simplify(multiple=True, loops=False)
    else:
        g.simplify(multiple=False, loops=True)

    if "name" not in g.vs.attributes():
        g.vs["name"] = [str(i) for i in range(g.vcount())]
    return g


def gcc_subgraph(g: ig.Graph) -> ig.Graph:
    if g.vcount() == 0:
        return g
    comps = g.components(mode="weak")
    if len(comps) == 0:
        return g
    giant = max(comps, key=len)
    return g.induced_subgraph(giant)


def degree_gini(deg: np.ndarray) -> float:
    if deg.size == 0:
        return float("nan")
    x = np.sort(deg.astype(float))
    if np.allclose(x, 0):
        return 0.0
    n = x.size
    cumx = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n)


def normalized_laplacian_eigs(g: ig.Graph) -> Optional[np.ndarray]:
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
    eigs = normalized_laplacian_eigs(g)
    if eigs is None or eigs.size < 2:
        return float("nan")
    return float(eigs[1])


def effective_resistance_kirchhoff(g: ig.Graph) -> float:
    n = g.vcount()
    if n <= 1:
        return 0.0
    if not g.is_connected():
        return float(BIG_PENALTY)
    if n > EFFRES_N_MAX:
        return float("nan")
    eigs = normalized_laplacian_eigs(g)
    if eigs is None or eigs.size != n:
        return float("nan")
    lam = eigs[1:]
    if np.any(lam <= 1e-12):
        return float(BIG_PENALTY)
    return float(n * np.sum(1.0 / lam))


def natural_connectivity(g: ig.Graph) -> float:
    n = g.vcount()
    if n == 0:
        return float("nan")
    if n > NATCONN_N_MAX:
        return float("nan")
    try:
        A = np.array(g.get_adjacency().data, dtype=float)
        mu = np.linalg.eigvalsh(A)
        return float(math.log(np.mean(np.exp(mu))))
    except Exception:
        return float("nan")


def avg_pairwise_vertex_connectivity(g: ig.Graph) -> float:
    n = g.vcount()
    if n <= 1:
        return 0.0

    gg = gcc_subgraph(g)
    n2 = gg.vcount()
    if n2 <= 1:
        return 0.0

    verts = list(range(n2))
    try:
        if n2 <= EXACT_CONN_N_MAX:
            vals = []
            for i in range(n2):
                for j in range(i + 1, n2):
                    vals.append(gg.vertex_connectivity(i, j))
            return float(np.mean(vals)) if vals else 0.0

        sample_nodes = random.sample(verts, k=min(SAMPLED_NODES_MAX, n2))
        vals = []
        for _ in range(SAMPLED_PAIRS_MAX):
            u, v = random.sample(sample_nodes, 2)
            vals.append(gg.vertex_connectivity(u, v))
        return float(np.mean(vals)) if vals else 0.0
    except Exception:
        return float("nan")


def betweenness_centralization(g: ig.Graph) -> float:
    try:
        return float(g.centralization_betweenness(directed=False, normalized=True))
    except Exception:
        return float("nan")


def compute_node_features(g: ig.Graph) -> Dict[str, np.ndarray]:
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
    """
    Return min/median/max/mean/std/p90 for an array, ignoring NaNs.
    """
    if x.size == 0:
        return {k: float("nan") for k in ["min", "median", "max", "mean", "std", "p90"]}
    x2 = x[np.isfinite(x)]
    if x2.size == 0:
        return {k: float("nan") for k in ["min", "median", "max", "mean", "std", "p90"]}

    return {
        "min": float(np.min(x2)),
        "median": float(np.median(x2)),
        "max": float(np.max(x2)),
        "mean": float(np.mean(x2)),
        "std": float(np.std(x2)),
        "p90": float(np.percentile(x2, 90)),
    }


def compute_graph_metrics(g: ig.Graph) -> Dict[str, float]:
    n = g.vcount()
    m = g.ecount()

    avg_deg = (2.0 * m / n) if n > 0 else float("nan")
    density = g.density(loops=False) if n > 1 else float("nan")

    comps = g.components(mode="weak")
    n_components = len(comps)
    gcc = gcc_subgraph(g)
    gcc_frac = (gcc.vcount() / n) if n > 0 else float("nan")

    # Path-based metrics on GCC
    if gcc.vcount() <= 1:
        aspl = 0.0 if gcc.vcount() == 1 else float("nan")
        diameter = 0.0 if gcc.vcount() == 1 else float("nan")
    else:
        try:
            aspl = float(gcc.average_path_length(directed=False))
        except Exception:
            aspl = float("nan")
        try:
            diameter = float(gcc.diameter(directed=False))
        except Exception:
            diameter = float("nan")

    # Global clustering + assortativity
    try:
        avg_clust = float(g.transitivity_undirected(mode="zero"))
    except Exception:
        avg_clust = float("nan")
    try:
        assort = float(g.assortativity_degree(directed=False))
    except Exception:
        assort = float("nan")

    # Centralization / inequality
    bet_cent = betweenness_centralization(g)
    deg = np.array(g.degree(), dtype=float)
    deg_gini = degree_gini(deg)

    # Spectral
    lam2 = algebraic_connectivity_lambda2(g)
    eff_res = effective_resistance_kirchhoff(g)
    nat_conn = natural_connectivity(g)

    # Connectivity
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

    graph_rows: List[Dict[str, object]] = []
    feat_rows: List[Dict[str, object]] = []

    missing = []
    for graph_name, filename in topo.items():
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            missing.append(filename)
            continue

        try:
            g = read_graph_graphml(path)
        except Exception as e:
            print(f"[WARN] Failed to read {filename}: {e}")
            continue

        gm = compute_graph_metrics(g)
        graph_rows.append({"graph": graph_name, "file": filename, **gm})

        feats = compute_node_features(g)
        feat_row: Dict[str, object] = {"graph": graph_name, "file": filename, "N": g.vcount(), "M": g.ecount()}
        for k, arr in feats.items():
            s = summarize_feature(arr)
            for stat, val in s.items():
                feat_row[f"{k}_{stat}"] = val
        feat_rows.append(feat_row)

        print(f"[OK] {graph_name}: N={g.vcount()}, M={g.ecount()}")

    if missing:
        print("\n[WARN] Missing files in dataset folder:")
        for f in missing:
            print(f" - {f}")
        print("The script will still write CSVs for files that were found.")

    df_graph = pd.DataFrame(graph_rows)
    df_feat = pd.DataFrame(feat_rows)

    if not df_graph.empty:
        df_graph = df_graph.sort_values(by=["N", "M", "graph"], ascending=[True, True, True])
    if not df_feat.empty:
        df_feat = df_feat.sort_values(by=["N", "M", "graph"], ascending=[True, True, True])

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
