# Graph Convolutional Network (GCN)
# Author: Shengning Wang

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

torch.sparse.check_sparse_tensor_invariants.disable()


def sparse_graph_aggregate(adj_indices: Tensor, adj_values: Tensor, x: Tensor) -> Tensor:
    """
    Apply a fixed sparse graph operator to node features.

    Args:
        adj_indices (Tensor): Sparse adjacency indices. (2, E).
        adj_values (Tensor): Sparse adjacency values. (E,).
        x (Tensor): Node features. (B, N, C).

    Returns:
        Tensor: Aggregated node features. (B, N, C).
    """
    B, N, _ = x.shape
    adj = torch.sparse_coo_tensor(adj_indices, adj_values.float(), size=(N, N), device=x.device).coalesce()
    y = [torch.sparse.mm(adj, x[b].float()) for b in range(B)]
    return torch.stack(y, dim=0).to(dtype=x.dtype)


class GraphConvolution(nn.Module):
    """
    Kipf-Welling graph convolution on a fixed normalized adjacency.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize graph convolution.

        Args:
            in_channels (int): Input feature width.
            out_channels (int): Output feature width.
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, adj_indices: Tensor, adj_values: Tensor) -> Tensor:
        """
        Apply graph convolution.

        Args:
            x (Tensor): Node features. (B, N, C_IN).
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).

        Returns:
            Tensor: Output node features. (B, N, C_OUT).
        """
        return self.linear(sparse_graph_aggregate(adj_indices, adj_values, x))


class GCN(nn.Module):
    """
    GCN baseline using a fixed kNN graph built from mesh coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        adj_indices: Tensor,
        adj_values: Tensor,
        width: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize GCN.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).
            width (int): Hidden channel width.
            depth (int): Number of graph convolution layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.register_buffer("adj_indices", adj_indices.long(), persistent=False)
        self.register_buffer("adj_values", adj_values.float(), persistent=False)
        self.input = GraphConvolution(in_channels + spatial_dim, width)
        self.layers = nn.ModuleList([GraphConvolution(width, width) for _ in range(depth - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(width, out_channels)

    def forward(self, inputs: Tensor, coords: Tensor, t_norm: Optional[Tensor] = None) -> Tensor:
        """
        Predict the next state on the mesh.

        Args:
            inputs (Tensor): Current node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Ignored normalized rollout time. (B,).

        Returns:
            Tensor: Predicted next state. (B, N, C_OUT).
        """
        x = torch.cat([coords, inputs], dim=-1)
        x = self.dropout(F.gelu(self.norms[0](self.input(x, self.adj_indices, self.adj_values))))
        for idx, layer in enumerate(self.layers, start=1):
            x = x + self.dropout(F.gelu(self.norms[idx](layer(x, self.adj_indices, self.adj_values))))
        return self.proj(x)

    def predict(self, inputs: Tensor, coords: Tensor, steps: int, bc: Optional[object] = None) -> Tensor:
        """
        Autoregressively predict a full trajectory from one initial state.

        Args:
            inputs (Tensor): Initial state. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            steps (int): Number of future frames to predict.
            bc (Optional[object]): Boundary condition with an enforce method.

        Returns:
            Tensor: Predicted sequence including the initial state. (B, steps + 1, N, C_OUT).
        """
        states = [inputs]
        state = inputs
        with torch.no_grad():
            for _ in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                state = self.forward(state, coords)
                if bc is not None:
                    state = bc.enforce(state)
                states.append(state)
        return torch.stack(states, dim=1)
