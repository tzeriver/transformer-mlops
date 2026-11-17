"""Small GCN regressor in plain PyTorch (no torch-geometric).

Each layer is a GCN-style message pass with symmetric degree normalization;
messages are gated by a learned scalar from the bond features, so bond type
actually influences the pass instead of being featurized and ignored.
Readout is global mean pooling over each molecule's atoms, then an MLP head.

Optional (defaults reproduce the original model exactly):
- pool="mean_sum_max": concat mean, scaled sum and max pooling, so the head
  sees molecule size and the most extreme atom, not only the average atom.
- residual=True: skip connections on layers after the first.
- desc_dim>0: concat standardized whole-molecule descriptors (featurize.py
  DESC_DIM) to the pooled vector. Standardization stats live in buffers, so
  they are saved in the state_dict with the weights.
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


POOLS = ("mean", "mean_sum_max")
# Sum pooling grows with atom count (~10-30 heavy atoms in ESOL); dividing by
# 10 keeps it on roughly the same scale as the mean and max features.
_SUM_SCALE = 10.0


class GCNRegressor(nn.Module):
    def __init__(
        self,
        atom_dim: int = ATOM_DIM,
        bond_dim: int = BOND_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        pool: str = "mean",
        residual: bool = False,
        desc_dim: int = 0,
    ):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool
        self.residual = residual
        self.desc_dim = desc_dim
        layers = [GCNLayer(atom_dim, hidden_dim, bond_dim)]
        layers += [GCNLayer(hidden_dim, hidden_dim, bond_dim) for _ in range(num_layers - 1)]
        self.layers = nn.ModuleList(layers)
        self.drop = nn.Dropout(dropout)
        if desc_dim:
            # Only registered when used, so default models keep the original
            # state_dict keys and older artifacts still load strictly.
            self.register_buffer("desc_mean", torch.zeros(desc_dim))
            self.register_buffer("desc_std", torch.ones(desc_dim))
        head_in = hidden_dim * (3 if pool == "mean_sum_max" else 1) + desc_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def set_descriptor_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store training-set descriptor mean/std used to standardize inputs."""
        self.desc_mean.copy_(mean)
        self.desc_std.copy_(std.clamp(min=1e-6))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        desc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            out = self.drop(layer(h, edge_index, edge_attr))
            # First layer changes width (atom_dim -> hidden), so no skip there.
            h = h + out if self.residual and i > 0 else out
        # Global pooling per molecule (mean, plus sum/max if enabled).
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        summed = torch.zeros(n_graphs, h.size(1), device=h.device)
        summed.index_add_(0, batch, h)
        counts = torch.zeros(n_graphs, device=h.device).index_add_(
            0, batch, torch.ones_like(batch, dtype=torch.float32)
        )
        pooled = summed / counts.unsqueeze(-1).clamp(min=1.0)
        if self.pool == "mean_sum_max":
            maxed = torch.zeros(n_graphs, h.size(1), device=h.device).scatter_reduce(
                0, batch.unsqueeze(-1).expand_as(h), h, reduce="amax", include_self=False
            )
            pooled = torch.cat([pooled, summed / _SUM_SCALE, maxed], dim=1)
        if self.desc_dim:
            if desc is None:
                raise ValueError("model was built with desc_dim > 0 but no descriptors were passed")
            pooled = torch.cat([pooled, (desc - self.desc_mean) / self.desc_std], dim=1)
        return self.head(pooled).squeeze(-1)
