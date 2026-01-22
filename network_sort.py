import networkx as nx
import os
import glob

def summarize_itz_graphml(directory_path, pattern="*.graphml"):
    """
    Scans a directory for GraphML files (default: all .graphml files),
    loads each one using NetworkX, and prints/collects:
    - Filename
    - Number of nodes
    - Number of edges
    
    Returns a list of dictionaries with the results.
    
    Example usage:
    results = summarize_itz_graphml("/path/to/zoo/graphml/folder")
    """
    # Find all matching files
    file_list = sorted(glob.glob(os.path.join(directory_path, pattern)))
    
    if not file_list:
        print(f"No GraphML files found in {directory_path} with pattern '{pattern}'")
        return []
    
    results = []
    
    print(f"Found {len(file_list)} GraphML files. Processing...\n")
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        try:
            # Read the graph (handles both directed and undirected)
            G = nx.read_graphml(filepath)
            
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            is_directed = nx.is_directed(G)
            
            # Collect result
            results.append({
                'filename': filename,
                'nodes': num_nodes,
                'edges': num_edges,
                'directed': is_directed
            })
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
    
    # Sort results by number of nodes (ascending)
    results.sort(key=lambda x: x['nodes'])
    
    # Print sorted results
    print(f"{'Filename':<35} {'Nodes':>6} {'Edges':>6} {'Directed?':>10}")
    print("-" * 65)
    for result in results:
        print(f"{result['filename']:<35} {result['nodes']:>6} {result['edges']:>6} {str(result['directed']):>10}")
    
    print("\nDone.")
    return results


if __name__ == "__main__":
    # Run the analysis on the real_world_topologies directory
    topology_dir = "real_world_topologies"
    
    if os.path.exists(topology_dir):
        results = summarize_itz_graphml(topology_dir)
        print(f"\nTotal networks analyzed: {len(results)}")
    else:
        print(f"Directory '{topology_dir}' not found in current directory.")
        print("Please provide the correct path to your topology files.")