"""
preprocess.py
-------------
Load graph datasets (CA-GrQc, com-dblp), extract node features,
generate synthetic labels via community detection, and partition
nodes into federated clients (IID and non-IID).
"""

import os
import time
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────
# 1.  Graph Loading
# ──────────────────────────────────────────────

def load_graph(filepath: str) -> nx.Graph:
    """Load an undirected graph from an edge-list text file.

    Lines beginning with '#' are treated as comments and skipped.
    Nodes are relabelled to consecutive integers starting at 0.
    """
    print(f"[preprocess] Loading graph from: {filepath}")
    t0 = time.time()
    G_raw = nx.read_edgelist(filepath, comments="#", nodetype=int)
    # Relabel to 0-based consecutive integers
    mapping = {n: i for i, n in enumerate(sorted(G_raw.nodes()))}
    G = nx.relabel_nodes(G_raw, mapping)
    G = G.to_undirected()
    # Remove self-loops (required by nx.core_number and clustering coeff)
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        G.remove_edges_from(self_loops)
        print(f"[preprocess]  → removed {len(self_loops)} self-loop(s)")
    elapsed = time.time() - t0
    print(f"[preprocess]  → {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges  ({elapsed:.2f}s)")
    return G


# ──────────────────────────────────────────────
# 2.  Feature Extraction
# ──────────────────────────────────────────────

def extract_features(G: nx.Graph, sample_size: int = None) -> tuple:
    """Extract structural node features.

    Features per node:
        0  degree
        1  log(degree + 1)
        2  local clustering coefficient
        3  average degree of neighbours
        4  core number (k-core decomposition)

    Parameters
    ----------
    G           : input graph
    sample_size : if set, only compute clustering coeff for a random
                  subset of nodes (speeds up large graphs).

    Returns
    -------
    nodes   : sorted list of node ids
    X       : float32 feature matrix  [N, 5]
    """
    print("[preprocess] Extracting node features …")
    t0 = time.time()

    nodes = sorted(G.nodes())
    N = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    degrees = dict(G.degree())
    core_numbers = nx.core_number(G)

    # Clustering coefficient — expensive on large graphs
    if sample_size and N > sample_size:
        sample_nodes = random.sample(nodes, sample_size)
        cc_partial = nx.clustering(G, nodes=sample_nodes)
        cc = defaultdict(float, cc_partial)
    else:
        cc = nx.clustering(G)

    X = np.zeros((N, 5), dtype=np.float32)
    for n in nodes:
        i = node_idx[n]
        d = degrees[n]
        nbr_degrees = [degrees[nb] for nb in G.neighbors(n)] if d > 0 else [0]
        X[i, 0] = d
        X[i, 1] = np.log1p(d)
        X[i, 2] = cc[n]
        X[i, 3] = np.mean(nbr_degrees)
        X[i, 4] = core_numbers[n]

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    elapsed = time.time() - t0
    print(f"[preprocess]  → feature matrix shape: {X.shape}  ({elapsed:.2f}s)")
    return nodes, X


# ──────────────────────────────────────────────
# 3.  Label Generation via Community Detection
# ──────────────────────────────────────────────

def generate_labels(G: nx.Graph, nodes: list,
                    n_classes: int = 5,
                    method: str = "cores") -> np.ndarray:
    """Generate synthetic multi-class labels.

    Two strategies
    ──────────────
    'cores'   : Assign each node to a class bucket based on its
                k-core number (fast, deterministic, works on any graph).
    'greedy'  : Greedy modularity communities then merge to n_classes.
                Better quality but slower.

    Returns
    -------
    y  : int32 array of shape [N]  with values in [0, n_classes)
    """
    print(f"[preprocess] Generating labels (method={method}, "
          f"n_classes={n_classes}) …")
    t0 = time.time()

    node_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    if method == "cores":
        core_numbers = nx.core_number(G)
        raw = np.array([core_numbers[n] for n in nodes], dtype=np.float32)
        # Quantise into n_classes equal-frequency buckets
        percentiles = np.percentile(raw, np.linspace(0, 100, n_classes + 1))
        percentiles[-1] += 1e-6          # ensure max value falls in last bucket
        y = np.digitize(raw, percentiles[1:-1]).astype(np.int32)

    elif method == "greedy":
        if not nx.is_connected(G):
            G_sub = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        else:
            G_sub = G
        communities = nx.algorithms.community.greedy_modularity_communities(G_sub)
        # Sort by size descending; keep top n_classes, merge rest into last class
        communities = sorted(communities, key=len, reverse=True)
        y = np.zeros(N, dtype=np.int32) + (n_classes - 1)
        for cls_id, comm in enumerate(communities[:n_classes]):
            for n in comm:
                if n in node_idx:
                    y[node_idx[n]] = cls_id

    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0
    counts = np.bincount(y, minlength=n_classes)
    print(f"[preprocess]  → label distribution: {counts}  ({elapsed:.2f}s)")
    return y


# ──────────────────────────────────────────────
# 4.  Client Partitioning
# ──────────────────────────────────────────────

def partition_iid(nodes: list, n_clients: int,
                  seed: int = 42) -> list:
    """Randomly shuffle and split node indices equally across clients."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(nodes))
    rng.shuffle(idx)
    return [arr.tolist() for arr in np.array_split(idx, n_clients)]


def partition_noniid(nodes: list, y: np.ndarray,
                     n_clients: int, alpha: float = 0.5,
                     seed: int = 42) -> list:
    """Dirichlet-based non-IID partition.

    Each client receives data drawn from a Dirichlet(alpha) distribution
    over classes.  Lower alpha → more heterogeneous.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1
    N = len(nodes)

    # Group indices by class
    class_indices = [np.where(y == c)[0].tolist() for c in range(n_classes)]
    for ci in class_indices:
        rng.shuffle(ci)          # in-place shuffle (modifies the list)

    client_indices = [[] for _ in range(n_clients)]
    for c_indices in class_indices:
        proportions = rng.dirichlet(np.repeat(alpha, n_clients))
        proportions = (np.cumsum(proportions) * len(c_indices)).astype(int)
        splits = np.split(c_indices, proportions[:-1])
        for k, split in enumerate(splits):
            client_indices[k].extend(split.tolist())

    return client_indices


# ──────────────────────────────────────────────
# 5.  High-level Convenience Function
# ──────────────────────────────────────────────

def prepare_dataset(filepath: str,
                    n_classes: int = 5,
                    n_clients: int = 10,
                    partition: str = "iid",
                    label_method: str = "cores",
                    sample_size: int = None,
                    seed: int = 42):
    """End-to-end pipeline: load → features → labels → partition.

    Returns
    -------
    G               : NetworkX graph
    nodes           : sorted node list
    X               : feature matrix [N, 5]
    y               : label array [N]
    client_indices  : list of n_clients lists of node indices
    """
    G = load_graph(filepath)
    nodes, X = extract_features(G, sample_size=sample_size)
    y = generate_labels(G, nodes, n_classes=n_classes, method=label_method)

    if partition == "iid":
        client_indices = partition_iid(nodes, n_clients, seed=seed)
    elif partition == "noniid":
        client_indices = partition_noniid(nodes, y, n_clients, seed=seed)
    else:
        raise ValueError(f"Unknown partition: {partition}")

    print(f"[preprocess] Done — {n_clients} clients "
          f"({partition} partition), ~{len(nodes)//n_clients} nodes each.\n")
    return G, nodes, X, y, client_indices


# ──────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../data/CA-GrQc.txt"
    G, nodes, X, y, clients = prepare_dataset(path, n_classes=5, n_clients=10)
    print(f"Feature matrix: {X.shape}")
    print(f"Labels:         {y.shape}, classes: {np.unique(y)}")
    print(f"Client sizes:   {[len(c) for c in clients]}")
