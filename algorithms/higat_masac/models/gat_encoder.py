import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATEncoder(nn.Module):
    def __init__(self, in_channels, config):
        super(GATEncoder, self).__init__()
        self.config = config
        self.heads = config['rl']['gat_heads']
        self.embed_dim = config['rl']['gat_embed_dim']
        self.num_layers = config['rl']['gat_layers']
        
        # We need in_channels to embed_dim mapping
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            GATConv(in_channels, self.embed_dim // self.heads, heads=self.heads, dropout=0.0)
        )
        
        # Middle layers
        for _ in range(self.num_layers - 2):
            self.convs.append(
                GATConv(self.embed_dim, self.embed_dim // self.heads, heads=self.heads, dropout=0.0)
            )
            
        # Final layer (no concatenation if we just want embed_dim out, or concat=False)
        if self.num_layers > 1:
            self.convs.append(
                GATConv(self.embed_dim, self.embed_dim, heads=1, concat=False, dropout=0.0)
            )
            
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, num_edge_features] -> GATConv can take edge_attr in newer PyG
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                
        # If batch is provided, it means we are dealing with multiple graphs and might want mean pooling
        if batch is not None:
            # Macro agent pools all nodes to get a graph-level embedding
            graph_embed = global_mean_pool(x, batch)
            return graph_embed
        
        # Micro agent returns node embeddings individually
        return x
