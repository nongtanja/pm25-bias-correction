import torch
from torch import nn

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self._adj_cache = {}

    def _compute_normalized_adj(self, edge_index, edge_attr, num_nodes, device):
        # This function computes D^-0.5 * (A+I) * D^-0.5
        edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        
        # Add self-loops
        loop_index = torch.arange(num_nodes, dtype=torch.long, device=device)
        loop_index = torch.stack([loop_index, loop_index], dim=0)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        
        if edge_attr is not None:
            loop_attr = torch.ones((num_nodes, edge_attr.size(1)), device=device)
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        
        row, col = edge_index
        edge_weight = edge_attr.squeeze() if edge_attr is not None else torch.ones(row.size(0), device=device)
        
        # Normalize
        deg = torch.zeros(num_nodes, device=device).scatter_add_(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        adj = torch.sparse_coo_tensor(edge_index, norm, (num_nodes, num_nodes), device=device)
        return adj.coalesce()

    def forward(self, x_batch, edge_index, edge_attr=None):
        B, N, F_in = x_batch.shape
        device = x_batch.device
        cache_key = (str(device), int(N))
        
        if cache_key not in self._adj_cache:
            adj = self._compute_normalized_adj(edge_index, edge_attr, N, device)
            self._adj_cache[cache_key] = adj
        adj = self._adj_cache[cache_key]
        
        x_lin = self.lin(x_batch)
        x_permuted = x_lin.permute(1, 0, 2).reshape(N, B * x_lin.shape[-1])
        out_flat = torch.sparse.mm(adj, x_permuted)
        out = out_flat.reshape(N, B, x_lin.shape[-1]).permute(1, 0, 2)
        out = self.act(out)
        return self.dropout(out)
    
class STGNNModel(nn.Module):
    def __init__(self, num_features, num_stations, hidden_dim=64, num_gnn_layers=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stations = num_stations
        
        self.station_emb = nn.Embedding(num_stations, hidden_dim)
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        self.temporal = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=2,
            batch_first=True, 
            dropout=dropout,
            bidirectional=False
        )
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr=None):
        batch, seq_len, num_nodes, num_features = x.size()
        device = x.device
        
        station_idx = torch.arange(num_nodes, device=device)
        station_emb = self.station_emb(station_idx) 
        
        spatial_seq = []
        for t in range(seq_len):
            xt = x[:, t] 
            h = self.input_proj(xt)
            h = h + station_emb.unsqueeze(0) 
            
            for gnn, norm in zip(self.gnn_layers, self.layer_norms):
                h_new = gnn(h, edge_index, edge_attr)
                h = norm(h + h_new) 
            
            spatial_seq.append(h.unsqueeze(1)) 
        
        spatial_seq = torch.cat(spatial_seq, dim=1) 
        
        B, T, N, H = spatial_seq.shape
        spatial_seq = spatial_seq.permute(0, 2, 1, 3).reshape(B*N, T, H)
        
        lstm_out, _ = self.temporal(spatial_seq) 
        
        attn_weights = self.temporal_attention(lstm_out) 
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        h = (lstm_out * attn_weights).sum(dim=1)
        
        pred = self.fc(h).view(B, N, 1)
        
        return pred