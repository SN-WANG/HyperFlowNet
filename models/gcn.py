# Graph Convolutional Network (GCN)
# Author: Shengning Wang

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

torch.sparse.check_sparse_tensor_invariants.disable()


def build_local_graph(coords: Tensor, k: int, sigma_scale: float = 1.5) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build one fixed row-normalized local graph from normalized mesh coordinates.

    Args:
        coords (Tensor): Node coordinates. (N, D).
        k (int): Number of nearest neighbors.
        sigma_scale (float): Distance scale multiplier.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            Sparse adjacency indices. (2, E_ADJ).
            Sparse adjacency values. (E_ADJ,).
            Undirected edge list. (2, E_EDGE).
    """
    N = coords.shape[0]
    dist = torch.cdist(coords, coords)
    dist.fill_diagonal_(float("inf"))

    knn = torch.topk(dist, k=k, largest=False).indices
    src = torch.arange(N, device=coords.device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = knn.reshape(-1)

    graph_src = torch.cat([src, dst], dim=0)
    graph_dst = torch.cat([dst, src], dim=0)
    pairs = torch.stack([graph_src, graph_dst], dim=1).tolist()
    unique_pairs = []
    seen = set()
    for src_idx, dst_idx in pairs:
        pair = (src_idx, dst_idx)
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    graph_edges = torch.tensor(unique_pairs, device=coords.device, dtype=torch.long)
    graph_src = graph_edges[:, 0]
    graph_dst = graph_edges[:, 1]
    graph_dist = torch.norm(coords[graph_dst] - coords[graph_src], dim=-1)

    sigma = (graph_dist.median() * sigma_scale).clamp_min(1e-6)
    graph_weight = torch.exp(-graph_dist / sigma)

    self_idx = torch.arange(N, device=coords.device)
    all_src = torch.cat([graph_src, self_idx], dim=0)
    all_dst = torch.cat([graph_dst, self_idx], dim=0)
    all_weight = torch.cat([graph_weight, torch.ones(N, device=coords.device, dtype=graph_weight.dtype)], dim=0)

    degree = torch.zeros(N, device=coords.device, dtype=all_weight.dtype)
    degree.index_add_(0, all_src, all_weight)
    adj_values = all_weight / degree[all_src].clamp_min(1e-8)
    adj_indices = torch.stack([all_src, all_dst], dim=0)

    edge_index = torch.stack([graph_src, graph_dst], dim=0)
    edge_index = edge_index[:, graph_src < graph_dst].contiguous()
    return adj_indices.long(), adj_values.float(), edge_index.long()


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
        graph_k: int = 12,
        graph_sigma_scale: float = 1.5,
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
            graph_k (int): Number of nearest neighbors in the local graph.
            graph_sigma_scale (float): Distance scale multiplier of graph weights.
            width (int): Hidden channel width.
            depth (int): Number of graph convolution layers.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.graph_k = graph_k
        self.graph_sigma_scale = graph_sigma_scale
        self.register_buffer("adj_indices", None, persistent=False)
        self.register_buffer("adj_values", None, persistent=False)
        self.register_buffer("graph_coords", None, persistent=False)
        self.input = GraphConvolution(in_channels + spatial_dim, width)
        self.layers = nn.ModuleList([GraphConvolution(width, width) for _ in range(depth - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(width, out_channels)

    def _graph(self, coords: Tensor) -> Tuple[Tensor, Tensor]:
        coords_ref = coords[0].detach()
        if self.graph_coords is None or not torch.equal(self.graph_coords, coords_ref):
            adj_indices, adj_values, _ = build_local_graph(
                coords=coords_ref,
                k=self.graph_k,
                sigma_scale=self.graph_sigma_scale,
            )
            self.adj_indices = adj_indices
            self.adj_values = adj_values
            self.graph_coords = coords_ref.clone()
        return self.adj_indices, self.adj_values

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
        adj_indices, adj_values = self._graph(coords)
        x = torch.cat([coords, inputs], dim=-1)
        x = self.dropout(F.gelu(self.norms[0](self.input(x, adj_indices, adj_values))))
        for idx, layer in enumerate(self.layers, start=1):
            x = x + self.dropout(F.gelu(self.norms[idx](layer(x, adj_indices, adj_values))))
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
