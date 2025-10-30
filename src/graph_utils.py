import numpy as np
import torch
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import pdist, squareform

def haversine_matrix(coords):
    """Calculate pairwise haversine distances"""
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lat[:, None])*np.cos(lat[None, :])*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.minimum(1, np.sqrt(a)))
    return 6371.0 * c

def create_graph_structure(coords_df, data_df, feature_cols, station_list, test_start_date=None, k=7, alpha=0.3, verbose=True):
    """
    Creates the graph structure (edge_index, edge_attr) for the GNN layers.
    Combines spatial distance (kNN) with feature similarity.
    """
    coords_df = coords_df.set_index('Station').reindex(station_list).reset_index()
    coords = coords_df[['latitude', 'longitude']].values
    num_stations = len(station_list)
    dist_mat = haversine_matrix(coords)

    # 1. Adjacency Matrix
    binary_adj = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
    binary_adj = binary_adj.maximum(binary_adj.T)
    adj_coo = binary_adj.tocoo()

    # 2. Similarity
    if test_start_date is not None:
        train_data = data_df[data_df['timestamp'] < test_start_date]
    else:
        train_data = data_df

    feature_matrix = train_data.groupby('Station')[feature_cols].mean().reindex(station_list).fillna(0).values
    feat_sim = 1 - squareform(pdist(feature_matrix, 'cosine'))
    np.fill_diagonal(feat_sim, 0)

    dist_sim = np.exp(-dist_mat / (dist_mat.std() + 1e-8))
    np.fill_diagonal(dist_sim, 0)

    combined = alpha * dist_sim + (1 - alpha) * feat_sim

    # 3. Weights
    row, col = adj_coo.row, adj_coo.col
    weights = combined[row, col]
    weights = np.clip(weights, 0, None)

    # 4. Find "Best Buddy"
    guaranteed_edges = set()
    for i in range(num_stations):
        similarities = combined[i].copy()
        similarities[i] = -np.inf 
        best_neighbor_idx = np.argmax(similarities)
        guaranteed_edges.add(tuple(sorted((i, best_neighbor_idx)))) 
    
    # 5. Create Masks
    mask_threshold = weights >= np.percentile(weights, 25)
    
    mask_guaranteed = np.zeros_like(mask_threshold, dtype=bool)
    for idx, (r, c) in enumerate(zip(row, col)):
        if tuple(sorted((r, c))) in guaranteed_edges:
            mask_guaranteed[idx] = True

    # 6. Combine Masks
    final_mask = mask_threshold | mask_guaranteed

    # 7. Apply final_mask
    row, col, weights = row[final_mask], col[final_mask], weights[final_mask]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    
    if verbose:
        print(f"\n🕸️ GNN Graph Structure (k_gnn={k}, alpha={alpha}):")
        print(f"  Total edges: {edge_index.shape[1]}")

    return edge_index, edge_attr

def create_pm25_neighbor_dict(coords_df, station_list, k_pm25, verbose=True):
    """
    Creates a neighbor dictionary for PM2.5 features using kNN (distance-only)
    and ensuring symmetry.
    """
    coords_df = coords_df.set_index('Station').reindex(station_list).reset_index()
    coords = coords_df[['latitude', 'longitude']].values
    
    # 1. kNN (distance)
    adj = kneighbors_graph(coords, n_neighbors=k_pm25, mode='connectivity', include_self=False)
    # 2. Make symmetric
    adj = adj.maximum(adj.T)
    adj_coo = adj.tocoo()
    
    neighbor_dict = {i: [] for i in range(len(station_list))}
    for r, c in zip(adj_coo.row, adj_coo.col):
        if c not in neighbor_dict[r]:
            neighbor_dict[r].append(c)
            
    if verbose:
        num_neighbors = [len(v) for v in neighbor_dict.values()]
        print(f"\n🤝 PM2.5 Neighbor Structure (k_pm25={k_pm25}, distance-only):")
        print(f"  Neighbors per station: min={min(num_neighbors)}, max={max(num_neighbors)}, mean={np.mean(num_neighbors):.1f}")
        print(f"  Stations with 0 neighbors: {sum(1 for n in num_neighbors if n == 0)}")
        
    return neighbor_dict