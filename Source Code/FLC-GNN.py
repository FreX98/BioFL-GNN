import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                           recall_score, roc_auc_score, confusion_matrix)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.metrics import pairwise_distances
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool
import os
from tabulate import tabulate
import time
from datetime import timedelta

class FastLocalConv(MessagePassing):
    """Fast Local Convolutional Layer"""
    def __init__(self, in_channels, out_channels):
        super(FastLocalConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = nn.Linear(in_channels, out_channels)
        self.att = nn.Parameter(torch.Tensor(1, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform node features
        x = self.lin(x)
        
        # Compute attention weights
        row, col = edge_index
        alpha = (x[row] * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Manual softmax implementation
        # Create a dictionary to group attention weights by source node
        node_alpha = {}
        for i, node in enumerate(row.tolist()):
            if node not in node_alpha:
                node_alpha[node] = []
            node_alpha[node].append(alpha[i].item())
        
        # Compute softmax for each group
        softmax_alpha = torch.zeros_like(alpha)
        for node, alphas in node_alpha.items():
            alphas = torch.tensor(alphas, device=x.device)
            softmax_alphas = F.softmax(alphas, dim=0)
            indices = [i for i, n in enumerate(row.tolist()) if n == node]
            for idx, val in zip(indices, softmax_alphas):
                softmax_alpha[idx] = val
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, alpha=softmax_alpha)

    def message(self, x_j, alpha):
        # Message passing with attention weights
        return alpha.view(-1, 1) * x_j
        
class GNN_Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, gnn_type,
                 n_layers=3, dropout=0.2, heads=3, use_fast_local=False):
        super().__init__()
        self.gnn_type = gnn_type
        self.use_fast_local = use_fast_local
        self.layers = nn.ModuleList()
        
        # Adjust hid_dim for GAT
        if gnn_type == 'GAT':
            # Ensure hid_dim is divisible by heads and >= heads
            hid_dim = max((hid_dim // heads) * heads, heads)
            print(f"Adjusted hid_dim to {hid_dim} for GAT with {heads} heads")
        
        # Fast Local Convolutional Layer (first layer)
        if use_fast_local:
            self.fast_local = FastLocalConv(in_dim, hid_dim)
            current_dim = hid_dim
        else:
            current_dim = in_dim

        # First GNN layer
        if gnn_type == 'GCN':
            self.layers.append(GCNConv(current_dim, hid_dim))
        elif gnn_type == 'GAT':
            # For GAT, we'll use a more robust initialization
            self.layers.append(self._create_gat_layer(current_dim, hid_dim, heads))
        elif gnn_type == 'GIN':
            nn1 = nn.Sequential(
                nn.Linear(current_dim, hid_dim),
                nn.ReLU(),
                nn.Linear(hid_dim, hid_dim)
            )
            self.layers.append(GINConv(nn1))
        elif gnn_type == 'GraphSAGE':
            self.layers.append(SAGEConv(current_dim, hid_dim))

        # Hidden layers
        for _ in range(n_layers-2):
            if gnn_type == 'GCN':
                self.layers.append(GCNConv(hid_dim, hid_dim))
            elif gnn_type == 'GAT':
                self.layers.append(self._create_gat_layer(hid_dim, hid_dim, heads))
            elif gnn_type == 'GIN':
                nn_h = nn.Sequential(
                    nn.Linear(hid_dim, hid_dim),
                    nn.ReLU(),
                    nn.Linear(hid_dim, hid_dim)
                )
                self.layers.append(GINConv(nn_h))
            elif gnn_type == 'GraphSAGE':
                self.layers.append(SAGEConv(hid_dim, hid_dim))

        # Output layer
        if gnn_type == 'GAT':
            # GAT concatenates head outputs by default
            out_in_dim = hid_dim
        else:
            out_in_dim = hid_dim
            
        self.out = nn.Linear(out_in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def _create_gat_layer(self, in_dim, hid_dim, heads):
        """Create a GAT layer with proper initialization"""
        print(f"[DEBUG] type(in_dim): {type(in_dim)}, value: {in_dim}")
        # Calculate output dimension per head
        out_dim = hid_dim // heads
        
        # Ensure out_dim is at least 1
        if out_dim < 1:
            out_dim = 1
            hid_dim = heads * out_dim  # Adjust hid_dim to maintain heads*out_dim
        
        return GATConv(
            in_channels=int(in_dim),  # Ensure in_dim is an integer
            out_channels=out_dim,
            heads=heads,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            bias=True
        )

    def forward(self, x, edge_index):
        # Apply fast local convolution if enabled
        if self.use_fast_local:
            x = self.fast_local(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        # Apply GNN layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.activation(x)
            x = self.dropout(x)

        # Final output
        x = self.out(x)
        return F.log_softmax(x, dim=1)
