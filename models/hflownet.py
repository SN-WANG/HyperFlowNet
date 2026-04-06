# HyperFlowNet for spatio-temporal irregular-mesh flow prediction
# Author: Shengning Wang

from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm


# ============================================================
# Encoding Blocks
# ============================================================


class SpatialEncoder(nn.Module):
    """
    Lightweight coordinate encoder with learned low-frequency and Fourier features.
    """

    def __init__(
        self,
        spatial_dim: int,
        coords_features: int = 8,
        learned_scale: float = 1.0,
    ) -> None:
        """
        Initialize the spatial encoder.

        Args:
            spatial_dim (int): Spatial coordinate dimension.
            coords_features (int): Number of learned low-frequency and Fourier features.
            learned_scale (float): Initialization scale of the learned frequency matrix.
        """
        super().__init__()

        self.spatial_dim = spatial_dim
        self.coords_features = coords_features

        if coords_features > 0:
            self.low_freq_proj = nn.Sequential(
                nn.Linear(spatial_dim, coords_features),
                nn.GELU(),
                nn.Linear(coords_features, coords_features),
            )
            self.freq_matrix = nn.Parameter(learned_scale * torch.randn(spatial_dim, coords_features))
            self.out_dim = spatial_dim + 3 * coords_features
        else:
            self.low_freq_proj = None
            self.freq_matrix = None
            self.out_dim = spatial_dim

    def forward(self, coords: Tensor) -> Tensor:
        """
        Encode mesh coordinates into lightweight geometry features.

        Args:
            coords (Tensor): Node coordinates. (B, N, D).

        Returns:
            Tensor: Encoded spatial features. (B, N, C).
        """
        if self.coords_features <= 0:
            return coords

        coords = coords.to(dtype=self.freq_matrix.dtype)
        low_freq = self.low_freq_proj(coords)
        phases = (2.0 * torch.pi) * (coords @ self.freq_matrix)
        return torch.cat([coords, low_freq, torch.sin(phases), torch.cos(phases)], dim=-1)


class TemporalEncoder(nn.Module):
    """
    Sinusoidal temporal encoder for normalized rollout time.
    """

    def __init__(self, time_features: int = 4, freq_base: int = 1000) -> None:
        """
        Initialize the temporal encoder.

        Args:
            time_features (int): Number of sinusoidal frequency pairs.
            freq_base (int): Reference time scale used to distribute frequencies.
        """
        super().__init__()
        self.time_features = time_features
        self.freq_base = freq_base

        indices = torch.arange(time_features, dtype=torch.float32)
        omega = freq_base ** (-indices / max(time_features, 1))
        self.register_buffer("omega", omega, persistent=False)
        self.out_dim = 2 * time_features

    def forward(self, t_norm: Tensor, num_nodes: int) -> Tensor:
        """
        Encode normalized time and broadcast it to all nodes.

        Args:
            t_norm (Tensor): Normalized time indices. (B,).
            num_nodes (int): Number of mesh nodes.

        Returns:
            Tensor: Time features. (B, N, C).
        """
        t_scaled = t_norm.float() * self.freq_base
        angles = self.omega.unsqueeze(0) * t_scaled.unsqueeze(1)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return embedding.unsqueeze(1).expand(-1, num_nodes, -1)


# ============================================================
# Token Blocks
# ============================================================


class SliceAttention(nn.Module):
    """
    Shared-assignment slice linear attention for irregular mesh node tokens.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int) -> None:
        """
        Initialize the mesh slice attention module.

        Args:
            hidden_dim (int): Hidden token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of slice tokens used to compress the node set.
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_slices = num_slices
        self.head_dim = hidden_dim // num_heads

        self.assignment_proj = nn.Linear(hidden_dim, num_slices)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        nn.init.orthogonal_(self.assignment_proj.weight)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, num_tokens, _ = x.shape
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, num_tokens, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, num_tokens, self.hidden_dim)

    def _linear_attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Apply linear attention in the compressed slice space.

        Args:
            query (Tensor): Slice queries. (B, H, S, D).
            key (Tensor): Slice keys. (B, H, S, D).
            value (Tensor): Slice values. (B, H, S, D).

        Returns:
            Tensor: Updated slice tokens. (B, H, S, D).
        """
        query = F.elu(query, alpha=1.0) + 1.0
        key = F.elu(key, alpha=1.0) + 1.0

        key_value = torch.einsum("bhsd,bhse->bhde", key, value)
        key_sum = key.sum(dim=2)
        denom = torch.einsum("bhsd,bhd->bhs", query, key_sum).unsqueeze(-1).clamp_min(1e-6)
        numer = torch.einsum("bhsd,bhde->bhse", query, key_value)
        return numer / denom

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply slice attention with shared slice assignment and linear slice mixing.

        Args:
            x (Tensor): Input node tokens. (B, N, H).

        Returns:
            Tensor: Updated node tokens. (B, N, H).
        """
        slice_weights = torch.softmax(self.assignment_proj(x), dim=-1)
        slice_tokens = torch.einsum("bns,bnc->bsc", slice_weights, x)
        slice_norm = slice_weights.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        slice_tokens = slice_tokens / slice_norm

        query = self._reshape_heads(self.query_proj(slice_tokens))
        key = self._reshape_heads(self.key_proj(slice_tokens))
        value = self._reshape_heads(self.value_proj(slice_tokens))

        out_slice = self._merge_heads(self._linear_attention(query, key, value))
        out = torch.einsum("bns,bsc->bnc", slice_weights, out_slice)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Token-wise feed-forward network used inside HyperFlow blocks.
    """

    def __init__(self, hidden_dim: int, ffn_ratio: float = 4.0) -> None:
        """
        Initialize the feed-forward network.

        Args:
            hidden_dim (int): Hidden token width.
            ffn_ratio (float): Expansion ratio of the intermediate hidden layer.
        """
        super().__init__()
        inner_dim = max(1, int(round(hidden_dim * ffn_ratio)))
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the feed-forward network on token features.

        Args:
            x (Tensor): Input tokens. (B, L, H).

        Returns:
            Tensor: Output tokens. (B, L, H).
        """
        return self.net(x)


class HyperFlowBlock(nn.Module):
    """
    Pre-norm slice-attention block for irregular mesh dynamics.
    """

    def __init__(self, hidden_dim: int, num_heads: int, num_slices: int, ffn_ratio: float) -> None:
        """
        Initialize one HyperFlow block.

        Args:
            hidden_dim (int): Hidden token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of slice tokens.
            ffn_ratio (float): Expansion ratio of the feed-forward network.
        """
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = SliceAttention(hidden_dim, num_heads, num_slices)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_ratio=ffn_ratio)

    def forward(self, node_tokens: Tensor) -> Tensor:
        """
        Update node tokens with slice attention and feed-forward residual blocks.

        Args:
            node_tokens (Tensor): Node tokens. (B, N, H).

        Returns:
            Tensor: Updated node tokens. (B, N, H).
        """
        node_tokens = node_tokens + self.attn(self.attn_norm(node_tokens))
        node_tokens = node_tokens + self.ffn(self.ffn_norm(node_tokens))
        return node_tokens


# ============================================================
# HyperFlowNet
# ============================================================


class HyperFlowNet(nn.Module):
    """
    Spatio-temporal slice operator for autoregressive flow prediction on irregular meshes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 128,
        depth: int = 4,
        num_slices: int = 32,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        use_spatial_encoding: bool = True,
        use_temporal_encoding: bool = True,
        coords_features: int = 8,
        time_features: int = 4,
        freq_base: int = 1000,
    ) -> None:
        """
        Initialize the HyperFlowNet architecture.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden token width.
            depth (int): Number of stacked HyperFlow blocks.
            num_slices (int): Number of slice tokens used by the slice operator.
            num_heads (int): Number of attention heads.
            ffn_ratio (float): Expansion ratio of the feed-forward network.
            use_spatial_encoding (bool): Whether to encode coordinates before concatenation.
            use_temporal_encoding (bool): Whether to append a sinusoidal time embedding.
            coords_features (int): Number of learned spatial encoding features.
            time_features (int): Number of temporal sinusoidal frequency pairs.
            freq_base (int): Reference time scale used by the temporal encoder.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dim = spatial_dim
        self.width = width
        self.num_slices = num_slices

        if use_spatial_encoding:
            self.spatial_encoder = SpatialEncoder(
                spatial_dim=spatial_dim,
                coords_features=coords_features,
            )
            spatial_width = self.spatial_encoder.out_dim
        else:
            self.spatial_encoder = None
            spatial_width = spatial_dim

        if use_temporal_encoding and time_features > 0:
            self.time_encoder = TemporalEncoder(time_features=time_features, freq_base=freq_base)
            time_width = self.time_encoder.out_dim
        else:
            self.time_encoder = None
            time_width = 0

        embed_in = in_channels + spatial_width + time_width
        self.input_embed = nn.Linear(embed_in, width)
        self.input_norm = nn.LayerNorm(width)

        self.blocks = nn.ModuleList([
            HyperFlowBlock(width, num_heads, num_slices, ffn_ratio=ffn_ratio)
            for _ in range(depth)
        ])

        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, out_channels)

    def forward(self, inputs: Tensor, coords: Tensor, t_norm: Optional[Tensor] = None) -> Tensor:
        """
        Predict the next state from the current flow field.

        Args:
            inputs (Tensor): Current node features. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized rollout time. (B,).

        Returns:
            Tensor: Predicted next state. (B, N, C_OUT).
        """
        batch_size, num_nodes, _ = coords.shape
        hidden_dtype = self.input_embed.weight.dtype

        inputs = inputs.to(dtype=hidden_dtype)
        coords = coords.to(dtype=hidden_dtype)

        components = [inputs]
        if self.spatial_encoder is not None:
            components.append(self.spatial_encoder(coords).to(dtype=hidden_dtype))
        else:
            components.append(coords)

        if self.time_encoder is not None:
            if t_norm is None:
                t_norm = torch.zeros(batch_size, device=coords.device, dtype=hidden_dtype)
            else:
                t_norm = t_norm.to(device=coords.device, dtype=hidden_dtype)
            components.append(self.time_encoder(t_norm, num_nodes).to(dtype=hidden_dtype))

        node_tokens = self.input_norm(self.input_embed(torch.cat(components, dim=-1)))

        for block in self.blocks:
            node_tokens = block(node_tokens)

        return self.output_head(self.output_norm(node_tokens))

    def predict(
        self,
        inputs: Tensor,
        coords: Tensor,
        steps: int,
        t0_norm: Optional[Tensor] = None,
        dt_norm: Optional[Tensor] = None,
        boundary_condition=None,
    ) -> Tensor:
        """
        Run autoregressive rollout for inference.

        Args:
            inputs (Tensor): Initial rollout state. (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            steps (int): Number of rollout steps.
            t0_norm (Optional[Tensor]): Initial normalized time index. (B,).
            dt_norm (Optional[Tensor]): Normalized time increment per rollout step. (B,).
            boundary_condition: Optional boundary-condition object exposing an `enforce` method.

        Returns:
            Tensor: Rollout sequence including the initial state. (B, T + 1, N, C_OUT).
        """
        device = next(self.parameters()).device
        input_state = inputs.to(device)
        coords = coords.to(device)

        if t0_norm is None:
            t0_norm = torch.zeros(input_state.shape[0], device=device, dtype=input_state.dtype)
        else:
            t0_norm = t0_norm.to(device=device, dtype=input_state.dtype)

        if dt_norm is None:
            dt_norm = torch.full((input_state.shape[0],), 1.0 / max(steps, 1), device=device, dtype=input_state.dtype)
        else:
            dt_norm = dt_norm.to(device=device, dtype=input_state.dtype)

        preds: List[Tensor] = [inputs.cpu()]

        with torch.no_grad():
            for step_idx in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                step_t_norm = t0_norm + step_idx * dt_norm
                next_state = self(input_state, coords, t_norm=step_t_norm)

                if boundary_condition is not None:
                    next_state = boundary_condition.enforce(next_state)

                preds.append(next_state.cpu())
                input_state = next_state

        return torch.stack(preds, dim=1)
