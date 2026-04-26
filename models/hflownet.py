# HyperFlowNet graph attention operators for shock-wave flow prediction
# Author: Shengning Wang

import math
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
    Apply a fixed sparse local graph operator to node features.

    Args:
        adj_indices (Tensor): Sparse adjacency indices. (2, E).
        adj_values (Tensor): Sparse adjacency values. (E,).
        x (Tensor): Node features. (B, N, C).

    Returns:
        Tensor: Locally aggregated node features. (B, N, C).
    """
    B, N, _ = x.shape
    adj = torch.sparse_coo_tensor(adj_indices, adj_values.float(), size=(N, N), device=x.device).coalesce()
    y = [torch.sparse.mm(adj, x[b].float()) for b in range(B)]
    return torch.stack(y, dim=0).to(dtype=x.dtype)


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
# Graph Attention Blocks
# ============================================================


class GraphBiasAttention(nn.Module):
    """
    Slice attention with graph structure injected into attention logits.
    """

    def __init__(
        self,
        width: int,
        num_slices: int,
        num_heads: int,
        graph_beta_init: float = 0.13,
        graph_bias_eps: float = 1e-6,
    ) -> None:
        """
        Initialize graph-biased slice attention.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            graph_beta_init (float): Initial graph bias strength.
            graph_bias_eps (float): Small bias stabilizer.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.graph_bias_eps = graph_bias_eps

        self.slice_proj = nn.Linear(width, num_slices)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(graph_beta_init)), dtype=torch.float32))

    def forward(self, x: Tensor, adj_indices: Tensor, adj_values: Tensor) -> Tensor:
        """
        Apply graph-biased slice attention.

        Args:
            x (Tensor): Node tokens. (B, N, C).
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).

        Returns:
            Tensor: Node update. (B, N, C).
        """
        B, _, C = x.shape
        H, D = self.num_heads, self.head_dim

        weights = F.softmax(self.slice_proj(x), dim=-1)
        weight_sum = weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(self.graph_bias_eps)
        slices = torch.bmm(weights.transpose(1, 2), x) / weight_sum

        graph_weights = sparse_graph_aggregate(adj_indices, adj_values, weights)
        graph_bias = torch.bmm(weights.transpose(1, 2), graph_weights)
        graph_bias = 0.5 * (graph_bias + graph_bias.transpose(1, 2))
        graph_bias = graph_bias / graph_bias.sum(dim=-1, keepdim=True).clamp_min(self.graph_bias_eps)
        graph_bias = torch.log(graph_bias.clamp_min(self.graph_bias_eps))

        q = self.q_proj(slices).view(B, -1, H, D).transpose(1, 2)
        k = self.k_proj(slices).view(B, -1, H, D).transpose(1, 2)
        v = self.v_proj(slices).view(B, -1, H, D).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        logits = logits + F.softplus(self.beta_raw) * graph_bias.unsqueeze(1)

        attn = torch.softmax(logits, dim=-1)
        slices_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, C)
        slices_out = self.out_proj(slices_out)
        return torch.bmm(weights, slices_out)


class GraphAssignAttention(nn.Module):
    """
    Slice attention with graph structure injected into node-to-slice assignment.
    """

    def __init__(self, width: int, num_slices: int, num_heads: int) -> None:
        """
        Initialize graph-aware assignment attention.

        Args:
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
        """
        super().__init__()
        self.assign_self = nn.Linear(width, width)
        self.assign_graph = nn.Linear(width, width, bias=False)
        self.assign_norm = nn.LayerNorm(width)
        self.slice_proj = nn.Linear(width, num_slices)
        self.attn = nn.MultiheadAttention(embed_dim=width, num_heads=num_heads, batch_first=True)

    def forward(self, x: Tensor, adj_indices: Tensor, adj_values: Tensor) -> Tensor:
        """
        Apply graph-aware assignment attention.

        Args:
            x (Tensor): Node tokens. (B, N, C).
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).

        Returns:
            Tensor: Node update. (B, N, C).
        """
        x_graph = sparse_graph_aggregate(adj_indices, adj_values, x)
        assign = F.gelu(self.assign_norm(self.assign_self(x) + self.assign_graph(x_graph)))
        weights = F.softmax(self.slice_proj(assign), dim=-1)

        weight_sum = weights.sum(dim=1, keepdim=True).transpose(1, 2).clamp_min(1e-8)
        slices = torch.bmm(weights.transpose(1, 2), x) / weight_sum
        slices_out, _ = self.attn(slices, slices, slices, need_weights=False)
        return torch.bmm(weights, slices_out)


class GraphFlowBlock(nn.Module):
    """
    One residual HyperFlowNet block with graph-injected slice attention.
    """

    def __init__(
        self,
        graph_mode: str,
        width: int,
        num_slices: int,
        num_heads: int,
        ffn_dim: int,
        graph_beta_init: float,
        graph_bias_eps: float,
    ) -> None:
        """
        Initialize one graph attention block.

        Args:
            graph_mode (str): Graph injection mode, either bias or assign.
            width (int): Node token width.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of attention heads.
            ffn_dim (int): Hidden width of the feed-forward block.
            graph_beta_init (float): Initial graph bias strength.
            graph_bias_eps (float): Small graph bias stabilizer.
        """
        super().__init__()
        if graph_mode == "bias":
            self.graph_attn = GraphBiasAttention(
                width=width,
                num_slices=num_slices,
                num_heads=num_heads,
                graph_beta_init=graph_beta_init,
                graph_bias_eps=graph_bias_eps,
            )
        elif graph_mode == "assign":
            self.graph_attn = GraphAssignAttention(width=width, num_slices=num_slices, num_heads=num_heads)
        else:
            raise ValueError("graph_mode must be either 'bias' or 'assign'")

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, width),
        )

    def forward(self, x: Tensor, adj_indices: Tensor, adj_values: Tensor) -> Tensor:
        """
        Apply one residual graph attention update.

        Args:
            x (Tensor): Node tokens. (B, N, C).
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).

        Returns:
            Tensor: Updated node tokens. (B, N, C).
        """
        x = x + self.graph_attn(self.norm1(x), adj_indices, adj_values)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# HyperFlowNet
# ============================================================


class HyperFlowNet(nn.Module):
    """
    HyperFlowNet graph-injection comparison model for shock-wave flow prediction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        adj_indices: Tensor,
        adj_values: Tensor,
        edge_index: Tensor,
        graph_mode: str = "bias",
        width: int = 128,
        depth: int = 4,
        num_slices: int = 32,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        coord_features: int = 8,
        time_features: int = 4,
        freq_base: int = 1000,
        graph_beta_init: float = 0.13,
        graph_bias_eps: float = 1e-6,
    ) -> None:
        """
        Initialize HyperFlowNet.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            adj_indices (Tensor): Sparse adjacency indices. (2, E).
            adj_values (Tensor): Sparse adjacency values. (E,).
            edge_index (Tensor): Undirected local edge list. (2, E_EDGE).
            graph_mode (str): Graph injection mode, either bias or assign.
            width (int): Hidden channel width.
            depth (int): Number of HyperFlowNet blocks.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            ffn_dim (Optional[int]): Hidden width of the feed-forward block.
            coord_features (int): Half-dimension of the Fourier spatial encoding.
            time_features (int): Half-dimension of the temporal encoding.
            freq_base (int): Base for temporal frequencies.
            graph_beta_init (float): Initial graph bias strength.
            graph_bias_eps (float): Small graph bias stabilizer.
        """
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * width

        self.graph_mode = graph_mode
        self.register_buffer("adj_indices", adj_indices.long(), persistent=False)
        self.register_buffer("adj_values", adj_values.float(), persistent=False)
        self.register_buffer("edge_index", edge_index.long(), persistent=False)

        self.spatial_encoder = SpatialEncoder(spatial_dim=spatial_dim, coord_features=coord_features)
        self.time_encoder = TemporalEncoder(time_features=time_features, freq_base=freq_base)
        coord_dim = 2 * coord_features
        time_dim = 2 * time_features

        self.embed = nn.Linear(in_channels + coord_dim + time_dim, width)
        self.blocks = nn.ModuleList([
            GraphFlowBlock(
                graph_mode=graph_mode,
                width=width,
                num_slices=num_slices,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                graph_beta_init=graph_beta_init,
                graph_bias_eps=graph_bias_eps,
            )
            for _ in range(depth)
        ])
        self.proj = nn.Linear(width, out_channels)

    def forward(self, inputs: Tensor, coords: Tensor, t_norm: Optional[Tensor] = None) -> Tensor:
        """
        Predict the next state on the mesh.

        Args:
            inputs (Tensor): Current node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized rollout time. (B,).

        Returns:
            Tensor: Predicted next state. (B, N, C_OUT).
        """
        B, N, _ = coords.shape
        if t_norm is None:
            t_norm = torch.zeros(B, device=coords.device, dtype=coords.dtype)
        else:
            t_norm = t_norm.to(device=coords.device, dtype=coords.dtype)

        x = torch.cat([
            inputs,
            self.spatial_encoder(coords),
            self.time_encoder(t_norm, N).to(dtype=coords.dtype),
        ], dim=-1)

        x = self.embed(x)
        for block in self.blocks:
            x = block(x, self.adj_indices, self.adj_values)
        return self.proj(x)

    def predict(self, inputs: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Autoregressively predict a full trajectory from one initial state.

        Args:
            inputs (Tensor): Initial state. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            steps (int): Number of future frames to predict.

        Returns:
            Tensor: Predicted sequence including the initial state. (B, steps + 1, N, C_OUT).
        """
        states = [inputs]
        state = inputs
        with torch.no_grad():
            for step_idx in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                t_norm = torch.full((inputs.shape[0],), step_idx / max(steps, 1), device=inputs.device, dtype=inputs.dtype)
                state = self.forward(state, coords, t_norm=t_norm)
                states.append(state)
        return torch.stack(states, dim=1)
