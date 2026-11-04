"""Small GCN regressor in plain PyTorch (no torch-geometric).

Each layer is a GCN-style message pass with symmetric degree normalization;
messages are gated by a learned scalar from the bond features, so bond type
actually influences the pass instead of being featurized and ignored.
Readout is global mean pooling over each molecule's atoms, then an MLP head.
"""

from __future__ import annotations

import torch
from torch import nn

from src.gnn.featurize import ATOM_DIM, BOND_DIM


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bond_dim: int = BOND_DIM):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.edge_gate = nn.Linear(bond_dim, 1)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        n = x.size(0)
        device = x.device
        # Add self-loops (zero bond features -> gate bias decides their weight).
        loop = torch.arange(n, device=device)
        loop_index = torch.stack([loop, loop], dim=0)
        loop_attr = torch.zeros(n, edge_attr.size(1), device=device)
        src = torch.cat([edge_index[0], loop_index[0]])
        dst = torch.cat([edge_index[1], loop_index[1]])
        attr = torch.cat([edge_attr, loop_attr], dim=0)
        # Symmetric normalization: D^-1/2 (A + I) D^-1/2.
        deg = torch.zeros(n, device=device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        gate = torch.sigmoid(self.edge_gate(attr)).squeeze(-1)
        h = self.lin(x)
        msg = h[src] * (norm * gate).unsqueeze(-1)
        out = torch.zeros_like(h).index_add_(0, dst, msg)
        return torch.relu(out)


class GCNRegressor(nn.Module):
    def __init__(
        self,
        atom_dim: int = ATOM_DIM,
        bond_dim: int = BOND_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        layers = [GCNLayer(atom_dim, hidden_dim, bond_dim)]
        layers += [GCNLayer(hidden_dim, hidden_dim, bond_dim) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.drop(layer(h, edge_index, edge_attr))
        # Global mean pooling per molecule.
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        pooled = torch.zeros(n_graphs, h.size(1), device=h.device)
        pooled.index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(
            0, batch, torch.ones_like(batch, dtype=torch.float32)
        )
        pooled = pooled / counts.unsqueeze(-1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)
