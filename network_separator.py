import networkx as nx
import glob
import os
import numpy as np
from collections import defaultdict

def jenks_breaks(data, n_classes):
    """
    Jenks Natural Breaks algorithm - finds optimal breakpoints that minimize
    variance within groups and maximize variance between groups.
    """
    data = sorted(data)
    n = len(data)
    
    # Initialize matrices
    mat1 = np.zeros((n + 1, n_classes + 1))
    mat2 = np.zeros((n + 1, n_classes + 1))
    
    for i in range(1, n_classes + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, n + 1):
            mat2[j][i] = float('inf')
    
    v = 0.0
    for l in range(2, n + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, n_classes + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    
    k = n
    kclass = [0] * (n_classes + 1)
    kclass[n_classes] = float(data[n - 1])
    
    for j in range(n_classes, 1, -1):
        idx = int(mat1[k][j]) - 2
        kclass[j - 1] = data[idx]
        k = int(mat1[k][j] - 1)
    
    return kclass[1:-1]  # Return breakpoints (excluding min and max)


def stratified_jenks_binning_and_sampling(
    directory_path,
    pattern="*.graphml",
    n_classes=3,
    sample_per_stratum=None,
    random_seed=42
):
    """
    1. Loads all GraphML files
    2. Computes node counts
    3. Uses Jenks Natural Breaks to find optimal cutoffs
    4. Performs stratified sampling
    """
    np.random.seed(random_seed)
    
    file_list = sorted(glob.glob(os.path.join(directory_path, pattern)))
    if not file_list:
        print("No files found.")
        return None, None
    
    networks = []
    for fp in file_list:
        fn = os.path.basename(fp)
        try:
            G = nx.read_graphml(fp)
            networks.append({
                'filename': fn,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'directed': nx.is_directed(G)
            })
        except Exception as e:
            print(f"Skip {fn}: {e}")
    
    if not networks:
        return None, None
    
    # Extract node counts
    node_counts = np.array([net['nodes'] for net in networks])
    print(f"Loaded {len(networks)} networks. Node range: {node_counts.min()} – {node_counts.max()}")
    print(f"Median: {np.median(node_counts):.1f}   Mean: {node_counts.mean():.1f}\n")
    
    # Apply Jenks Natural Breaks
    print("Computing Jenks Natural Breaks (optimal breakpoints)...")
    breaks = jenks_breaks(node_counts.tolist(), n_classes)
    cutoff_small = int(breaks[0])
    cutoff_large = int(breaks[1]) if len(breaks) > 1 else int(breaks[0]) + 1
    
    print(f"Jenks breakpoints: {[int(b) for b in breaks]}")
    print(f"Optimal cutoffs: ≤ {cutoff_small}  |  {cutoff_small+1}–{cutoff_large}  |  > {cutoff_large}\n")
    
    # Assign strata
    strata_groups = {'Small': [], 'Medium': [], 'Large': []}
    for net in networks:
        if net['nodes'] <= cutoff_small:
            strata_groups['Small'].append(net)
        elif net['nodes'] <= cutoff_large:
            strata_groups['Medium'].append(net)
        else:
            strata_groups['Large'].append(net)
    
    # Report stratum sizes
    print("Stratum sizes:")
    for stratum_name in ['Small', 'Medium', 'Large']:
        count = len(strata_groups[stratum_name])
        if count > 0:
            nodes = [n['nodes'] for n in strata_groups[stratum_name]]
            print(f"  {stratum_name}: {count} networks (nodes: {min(nodes)}–{max(nodes)})")
        else:
            print(f"  {stratum_name}: 0 networks")
    
    # Decide sample sizes per stratum
    if sample_per_stratum is None:
        # Proportional sampling
        total_desired = 60
        counts = [len(strata_groups[s]) for s in ['Small', 'Medium', 'Large']]
        proportions = np.array(counts) / sum(counts)
        sample_sizes = np.round(proportions * total_desired).astype(int)
        sample_sizes[-1] += total_desired - sample_sizes.sum()
    else:
        sample_sizes = sample_per_stratum
    
    print(f"\nSampling plan: Small={sample_sizes[0]}, Medium={sample_sizes[1]}, Large={sample_sizes[2]}")
    
    # Perform stratified sampling
    sampled_networks = []
    for i, stratum_name in enumerate(['Small', 'Medium', 'Large']):
        group = strata_groups[stratum_name]
        if len(group) == 0:
            print(f"Warning: {stratum_name} stratum empty")
            continue
        n_take = min(sample_sizes[i], len(group))
        selected = list(np.random.choice(group, size=n_take, replace=False))
        sampled_networks.extend(selected)
    
    # Summary of sampled nodes
    sampled_nodes = [n['nodes'] for n in sampled_networks]
    print("\nSampled node counts stats:")
    print(f"  Count: {len(sampled_nodes)}")
    print(f"  Range: {min(sampled_nodes)} – {max(sampled_nodes)}")
    print(f"  Median: {np.median(sampled_nodes):.1f}")
    
    # Sort sampled by size
    sampled_networks.sort(key=lambda x: x['nodes'])
    
    return sampled_networks, strata_groups


def stratified_manual_binning_and_sampling(
    directory_path,
    pattern="*.graphml",
    cutoff_small=20,
    cutoff_large=85,
    sample_per_stratum=None,
    random_seed=42
):
    """
    1. Loads all GraphML files
    2. Computes node counts
    3. Uses manual cutoffs: Small (≤cutoff_small), Medium (cutoff_small+1 to cutoff_large), Large (>cutoff_large)
    4. Performs stratified sampling
    """
    np.random.seed(random_seed)
    
    file_list = sorted(glob.glob(os.path.join(directory_path, pattern)))
    if not file_list:
        print("No files found.")
        return None, None
    
    networks = []
    for fp in file_list:
        fn = os.path.basename(fp)
        try:
            G = nx.read_graphml(fp)
            networks.append({
                'filename': fn,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'directed': nx.is_directed(G)
            })
        except Exception as e:
            print(f"Skip {fn}: {e}")
    
    if not networks:
        return None, None
    
    # Extract node counts
    node_counts = np.array([net['nodes'] for net in networks])
    print(f"Loaded {len(networks)} networks. Node range: {node_counts.min()} – {node_counts.max()}")
    print(f"Median: {np.median(node_counts):.1f}   Mean: {node_counts.mean():.1f}\n")
    
    print(f"Manual cutoffs: ≤ {cutoff_small}  |  {cutoff_small+1}–{cutoff_large}  |  > {cutoff_large}\n")
    
    # Assign strata
    strata_groups = {'Small': [], 'Medium': [], 'Large': []}
    for net in networks:
        if net['nodes'] <= cutoff_small:
            strata_groups['Small'].append(net)
        elif net['nodes'] <= cutoff_large:
            strata_groups['Medium'].append(net)
        else:
            strata_groups['Large'].append(net)
    
    # Report stratum sizes
    print("Stratum sizes:")
    for stratum_name in ['Small', 'Medium', 'Large']:
        count = len(strata_groups[stratum_name])
        if count > 0:
            nodes = [n['nodes'] for n in strata_groups[stratum_name]]
            print(f"  {stratum_name}: {count} networks (nodes: {min(nodes)}–{max(nodes)})")
        else:
            print(f"  {stratum_name}: 0 networks")
    
    # Decide sample sizes per stratum
    if sample_per_stratum is None:
        # Proportional sampling
        total_desired = 60
        counts = [len(strata_groups[s]) for s in ['Small', 'Medium', 'Large']]
        proportions = np.array(counts) / sum(counts)
        sample_sizes = np.round(proportions * total_desired).astype(int)
        sample_sizes[-1] += total_desired - sample_sizes.sum()
    else:
        sample_sizes = sample_per_stratum
    
    print(f"\nSampling plan: Small={sample_sizes[0]}, Medium={sample_sizes[1]}, Large={sample_sizes[2]}")
    
    # Perform stratified sampling
    sampled_networks = []
    for i, stratum_name in enumerate(['Small', 'Medium', 'Large']):
        group = strata_groups[stratum_name]
        if len(group) == 0:
            print(f"Warning: {stratum_name} stratum empty")
            continue
        n_take = min(sample_sizes[i], len(group))
        selected = list(np.random.choice(group, size=n_take, replace=False))
        sampled_networks.extend(selected)
    
    # Summary of sampled nodes
    sampled_nodes = [n['nodes'] for n in sampled_networks]
    print("\nSampled node counts stats:")
    print(f"  Count: {len(sampled_nodes)}")
    print(f"  Range: {min(sampled_nodes)} – {max(sampled_nodes)}")
    print(f"  Median: {np.median(sampled_nodes):.1f}")
    
    # Sort sampled by size
    sampled_networks.sort(key=lambda x: x['nodes'])
    
    return sampled_networks, strata_groups


def stratified_log_binning_and_sampling(
    directory_path,
    pattern="*.graphml",
    n_strata=3,
    sample_per_stratum=None,          # e.g. [10, 15, 5] or None → proportional
    random_seed=42
):
    """
    1. Loads all GraphML files
    2. Computes node counts
    3. Applies log10 transform
    4. Creates equal-width bins in log-space → strata
    5. Performs stratified sampling
    """
    np.random.seed(random_seed)
    
    file_list = sorted(glob.glob(os.path.join(directory_path, pattern)))
    if not file_list:
        print("No files found.")
        return None, None
    
    networks = []
    for fp in file_list:
        fn = os.path.basename(fp)
        try:
            G = nx.read_graphml(fp)
            networks.append({
                'filename': fn,
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'directed': nx.is_directed(G)
            })
        except Exception as e:
            print(f"Skip {fn}: {e}")
    
    if not networks:
        return None, None
    
    # Extract node counts
    node_counts = np.array([net['nodes'] for net in networks])
    print(f"Loaded {len(networks)} networks. Node range: {node_counts.min()} – {node_counts.max()}")
    print(f"Median: {np.median(node_counts):.1f}   Mean: {node_counts.mean():.1f}\n")
    
    # Log transform (base 10)
    log_nodes = np.log10(node_counts)
    log_min, log_max = log_nodes.min(), log_nodes.max()
    
    # Equal-width bins in log space
    bin_edges = np.linspace(log_min, log_max, n_strata + 1)
    cutoffs = 10 ** bin_edges[1:-1]   # internal cut points
    
    print("Log-space bin edges:", [f"{x:.3f}" for x in bin_edges])
    print(f"Approximate node cutoffs: ≤ {cutoffs[0]:.0f}  |  {cutoffs[0]:.0f}+ to {cutoffs[1]:.0f}  |  > {cutoffs[1]:.0f}")
    
    # Assign strata (0 = small, 1 = medium, ..., n_strata-1 = large)
    strata_labels = np.digitize(log_nodes, bin_edges[:-1])
    
    # Group networks by stratum
    strata_groups = defaultdict(list)
    for i, net in enumerate(networks):
        strata_groups[strata_labels[i]].append(net)
    
    # Report stratum sizes
    print("\nStratum sizes:")
    for s in range(n_strata):
        print(f"  Stratum {s} ({s==0 and 'Small' or s==1 and 'Medium' or 'Large'}): "
              f"{len(strata_groups[s])} networks")
    
    # Decide sample sizes per stratum
    total_desired = None
    if sample_per_stratum is None:
        # Proportional sampling example: total 60 networks
        total_desired = 60
        proportions = np.array([len(strata_groups[s]) for s in range(n_strata)])
        proportions = proportions / proportions.sum()
        sample_sizes = np.round(proportions * total_desired).astype(int)
        # Adjust to exact total
        sample_sizes[-1] += total_desired - sample_sizes.sum()
    else:
        sample_sizes = np.array(sample_per_stratum)
        total_desired = sample_sizes.sum()
    
    print(f"\nSampling plan ({total_desired} total): {sample_sizes.tolist()}")
    
    # Perform stratified sampling
    sampled_networks = []
    for s in range(n_strata):
        group = strata_groups[s]
        if len(group) == 0:
            print(f"Warning: stratum {s} empty")
            continue
        n_take = min(sample_sizes[s], len(group))
        selected = np.random.choice(group, size=n_take, replace=False)
        sampled_networks.extend(selected)
    
    # Summary of sampled nodes
    sampled_nodes = [n['nodes'] for n in sampled_networks]
    print("\nSampled node counts stats:")
    print(f"  Count: {len(sampled_nodes)}")
    print(f"  Range: {min(sampled_nodes)} – {max(sampled_nodes)}")
    print(f"  Median: {np.median(sampled_nodes):.1f}")
    
    # Optional: sort sampled by size for easier inspection
    sampled_networks.sort(key=lambda x: x['nodes'])
    
    return sampled_networks, strata_groups

# ────────────────────────────────────────────────
# Usage example — change the path!
# folder = r"C:\your\path\to\graphml\files"
# sampled, all_groups = stratified_log_binning_and_sampling(
#     folder,
#     sample_per_stratum=[15, 20, 10],   # e.g. more emphasis on medium
#     random_seed=123
# )


if __name__ == "__main__":
    # Run the analysis on the real_world_topologies directory
    topology_dir = "real_world_topologies"
    
    if os.path.exists(topology_dir):
        print("Running stratified Jenks Natural Breaks binning and sampling...\n")
        sampled, all_groups = stratified_jenks_binning_and_sampling(
            topology_dir,
            n_classes=3,
            sample_per_stratum=[15, 20, 10],  # Small, Medium, Large
            random_seed=42
        )
        
        # Print detailed results for each stratum
        for stratum_name in ['Small', 'Medium', 'Large']:
            if stratum_name in all_groups and all_groups[stratum_name]:
                print(f"\n{'='*65}")
                print(f"{stratum_name} Networks ({len(all_groups[stratum_name])} networks)")
                print(f"{'='*65}")
                print(f"{'Filename':<35} {'Nodes':>6} {'Edges':>6} {'Directed?':>10}")
                print("-" * 65)
                for net in all_groups[stratum_name]:
                    print(f"{net['filename']:<35} {net['nodes']:>6} {net['edges']:>6} {str(net['directed']):>10}")
        
        print(f"\n{'='*65}")
        print(f"SAMPLED Networks ({len(sampled)} total)")
        print(f"{'='*65}")
        print(f"{'Filename':<35} {'Nodes':>6} {'Edges':>6} {'Directed?':>10}")
        print("-" * 65)
        for net in sampled:
            print(f"{net['filename']:<35} {net['nodes']:>6} {net['edges']:>6} {str(net['directed']):>10}")
    else:
        print(f"Directory '{topology_dir}' not found in current directory.")
        print("Please provide the correct path to your topology files.")
