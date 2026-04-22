# HyperFlowNet: A Spatio-Temporal Neural Operator for Shock-Wave Flow Simulation
# Author: Shengning Wang

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

torch.sparse.check_sparse_tensor_invariants.enable()


def build_local_graph(coords: Tensor, k: int, sigma_scale: float = 1.5) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build the fixed local graph operators from one normalized mesh.

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

    dir_src = torch.cat([src, dst], dim=0)
    dir_dst = torch.cat([dst, src], dim=0)
    dir_dist = torch.norm(coords[dir_dst] - coords[dir_src], dim=-1)

    sigma = (dir_dist.median() * sigma_scale).clamp_min(1e-6)
    dir_weight = torch.exp(-dir_dist / sigma)

    self_idx = torch.arange(N, device=coords.device)
    all_src = torch.cat([dir_src, self_idx], dim=0)
    all_dst = torch.cat([dir_dst, self_idx], dim=0)
    all_weight = torch.cat([dir_weight, torch.ones(N, device=coords.device, dtype=dir_weight.dtype)], dim=0)

    degree = torch.zeros(N, device=coords.device, dtype=all_weight.dtype)
    degree.index_add_(0, all_src, all_weight)
    adj_weight = all_weight / degree[all_src].clamp_min(1e-8)
    adj_indices = torch.stack([all_src, all_dst], dim=0)

    edge_pairs = torch.stack([torch.minimum(src, dst), torch.maximum(src, dst)], dim=1)
    edge_index = torch.unique(edge_pairs, dim=0).transpose(0, 1).contiguous()
    return adj_indices.long(), adj_weight.float(), edge_index.long()


def sparse_graph_aggregate(adj_indices: Tensor, adj_values: Tensor, x: Tensor) -> Tensor:
    """
    Aggregate node features with one fixed sparse local operator.

    Args:
        adj_indices (Tensor): Sparse adjacency indices. (2, E).
        adj_values (Tensor): Sparse adjacency values. (E,).
        x (Tensor): Node features. (B, N, C).

    Returns:
        Tensor: Local aggregated features. (B, N, C).
    """
    B, N, _ = x.shape
    adj = torch.sparse_coo_tensor(adj_indices, adj_values.float(), size=(N, N), device=x.device).coalesce()
    x32 = x.float()
    y32 = [torch.sparse.mm(adj, x32[b]) for b in range(B)]
    return torch.stack(y32, dim=0).to(dtype=x.dtype)


# ============================================================
# Encoding Blocks
# ============================================================


class SpatialEncoder(nn.Module):
    """
    Learnable Fourier encoder for irregular mesh coordinates.
    """

    def __init__(self, spatial_dim: int, coord_features: int = 8) -> None:
        """
        Initialize the spatial encoder.

        Args:
            spatial_dim (int): Spatial dimensionality.
            coord_features (int): Half-dimension of the encoded coordinates.
        """
        super().__init__()
        self.freq_matrix = nn.Parameter(torch.randn(spatial_dim, coord_features))

    def forward(self, coords: Tensor) -> Tensor:
        """
        Encode physical coordinates with learnable Fourier features.

        Args:
            coords (Tensor): Node coordinates. (B, N, D).

        Returns:
            Tensor: Encoded coordinates. (B, N, 2 * C_COORD).
        """
        proj = (2.0 * torch.pi) * (coords @ self.freq_matrix)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TemporalEncoder(nn.Module):
    """
    Sinusoidal encoder for normalized rollout time.
    """

    def __init__(self, time_features: int = 4, freq_base: int = 1000) -> None:
        """
        Initialize the temporal encoder.

        Args:
            time_features (int): Half-dimension of the temporal embedding.
            freq_base (int): Base for exponentially decaying frequencies.
        """
        super().__init__()
        indices = torch.arange(time_features, dtype=torch.float32)
        omega = freq_base ** (-indices / max(time_features, 1))
        self.freq_base = freq_base
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, t_norm: Tensor, num_nodes: int) -> Tensor:
        """
        Encode normalized time and broadcast it to all nodes.

        Args:
            t_norm (Tensor): Normalized frame times. (B,).
            num_nodes (int): Number of mesh nodes.

        Returns:
            Tensor: Temporal embedding. (B, N, 2 * C_TIME).
        """
        t_scaled = t_norm.float() * self.freq_base
        angles = self.omega.unsqueeze(0) * t_scaled.unsqueeze(1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.unsqueeze(1).expand(-1, num_nodes, -1)


# ============================================================
# Frontier Blocks
# ============================================================


class FrontierAttention(nn.Module):
    """
    Frontier-aware slice attention on irregular meshes.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int, frontier_beta: float = 1.0) -> None:
        """
        Initialize the frontier-aware slice attention.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            frontier_beta (float): Frontier contrast strength.
        """
        super().__init__()
        self.frontier_beta = frontier_beta
        self.norm = nn.LayerNorm(width)
        self.slice_proj = nn.Linear(width, num_slices)
        self.frontier_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=num_heads, batch_first=True)

    def forward(self, x: Tensor, adj_indices: Tensor, adj_values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply frontier-aware slice attention on node features.

        Args:
            x (Tensor): Node tokens. (B, N, C).
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).

        Returns:
            Tuple[Tensor, Tensor]:
                Node update. (B, N, C).
                Slice assignments. (B, N, S).
        """
        h = self.norm(x)
        h_local = sparse_graph_aggregate(adj_indices, adj_values, h)
        h_frontier = h - h_local

        logits = self.slice_proj(h) + self.frontier_beta * self.frontier_proj(h_frontier)
        weights = F.softmax(logits, dim=-1)

        weight_sum = weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-8)
        slices = torch.bmm(weights.transpose(1, 2), h) / weight_sum
        slices_out, _ = self.attn(slices, slices, slices, need_weights=False)
        return torch.bmm(weights, slices_out), weights


class HyperFlowBlock(nn.Module):
    """
    One residual HyperFlowNet block with FrontierAttention.
    """

    def __init__(
        self,
        width: int,
        num_slices: int,
        num_heads: int,
        ffn_dim: int,
        frontier_beta: float,
    ) -> None:
        """
        Initialize one HyperFlowNet block.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of attention heads.
            ffn_dim (int): Hidden width of the feed-forward block.
            frontier_beta (float): Frontier contrast strength.
        """
        super().__init__()
        self.frontier_attn = FrontierAttention(width, num_slices, num_heads, frontier_beta=frontier_beta)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, width),
        )

    def forward(self, x: Tensor, adj_indices: Tensor, adj_values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Apply one residual HyperFlowNet update.

        Args:
            x (Tensor): Node tokens. (B, N, C).
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).

        Returns:
            Tuple[Tensor, Tensor]:
                Updated node tokens. (B, N, C).
                Slice assignments. (B, N, S).
        """
        x_update, weights = self.frontier_attn(x, adj_indices, adj_values)
        x = x + x_update
        x = x + self.ffn(self.norm(x))
        return x, weights


# ============================================================
# HyperFlowNet
# ============================================================


class HyperFlowNet(nn.Module):
    """
    HyperFlowNet: A Spatio-Temporal Neural Operator for Shock-Wave Flow Simulation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        adj_indices: Tensor,
        adj_values: Tensor,
        edge_index: Tensor,
        width: int = 256,
        depth: int = 8,
        num_slices: int = 32,
        num_heads: int = 8,
        frontier_beta: float = 1.0,
        ffn_dim: Optional[int] = None,
        coord_features: int = 8,
        time_features: int = 4,
        freq_base: int = 1000,
    ) -> None:
        """
        Initialize HyperFlowNet for shock-wave flow simulation.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).
            edge_index (Tensor): Undirected local edge list. (2, E_EDGE).
            width (int): Hidden channel width.
            depth (int): Number of HyperFlowNet blocks.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            frontier_beta (float): Frontier contrast strength.
            ffn_dim (Optional[int]): Hidden width of the feed-forward block.
            coord_features (int): Half-dimension of the Fourier spatial encoding.
            time_features (int): Half-dimension of the temporal encoding.
            freq_base (int): Base for temporal frequencies.
        """
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * width

        self.register_buffer("adj_indices", adj_indices.long(), persistent=False)
        self.register_buffer("adj_values", adj_values.float(), persistent=False)
        self.register_buffer("edge_index", edge_index.long(), persistent=False)

        self.spatial_encoder = SpatialEncoder(spatial_dim=spatial_dim, coord_features=coord_features)
        self.time_encoder = TemporalEncoder(time_features=time_features, freq_base=freq_base)
        coord_dim = 2 * coord_features
        time_dim = 2 * time_features

        self.embed = nn.Linear(in_channels + coord_dim + time_dim, width)
        self.blocks = nn.ModuleList([
            HyperFlowBlock(
                width=width,
                num_slices=num_slices,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                frontier_beta=frontier_beta,
            )
            for _ in range(depth)
        ])
        self.proj = nn.Linear(width, out_channels)

    def forward(self, inputs: Tensor, coords: Tensor, t_norm: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        """
        Predict the next state on the mesh.

        Args:
            inputs (Tensor): Current node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized rollout time. (B,).

        Returns:
            Tuple[Tensor, List[Tensor]]:
                Predicted next state. (B, N, C_OUT).
                Slice assignments from all blocks.
        """
        B, N, _ = coords.shape
        if t_norm is None:
            t_norm = torch.zeros(B, device=coords.device, dtype=coords.dtype)
        else:
            t_norm = t_norm.to(device=coords.device, dtype=coords.dtype)

        comps = [
            inputs,
            self.spatial_encoder(coords),
            self.time_encoder(t_norm, N).to(dtype=coords.dtype),
        ]

        x = self.embed(torch.cat(comps, dim=-1))
        weight_bank: List[Tensor] = []
        for block in self.blocks:
            x, weights = block(x, self.adj_indices, self.adj_values)
            weight_bank.append(weights)
        return self.proj(x), weight_bank

    def predict(self, inputs: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Run autoregressive rollout prediction.

        Args:
            inputs (Tensor): Initial rollout state. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            steps (int): Number of rollout steps.

        Returns:
            Tensor: Predicted sequence with the initial state. (B, T + 1, N, C_OUT).
        """
        device = next(self.parameters()).device
        x = inputs.to(device)
        coords = coords.to(device)

        seq: List[Tensor] = [x.cpu()]
        with torch.no_grad():
            iterator = tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True)
            for step_idx in iterator:
                t_norm = torch.full((x.shape[0],), step_idx / max(steps, 1), device=device, dtype=x.dtype)
                x, _ = self.forward(x, coords, t_norm=t_norm)
                seq.append(x.cpu())
        return torch.stack(seq, dim=1)
