# Annealed sliding-history HyperFlowNet for flow prediction
# Author: Shengning Wang

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm


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
            t_norm (Tensor): Normalized frame times. (B, L).
            num_nodes (int): Number of mesh nodes.

        Returns:
            Tensor: Temporal embedding. (B, L, N, 2 * C_TIME).
        """
        t_scaled = t_norm.float() * self.freq_base
        angles = t_scaled.unsqueeze(-1) * self.omega.view(1, 1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.unsqueeze(2).expand(-1, -1, num_nodes, -1)


class LagEncoder(nn.Module):
    """
    Sinusoidal encoder for relative history lag indices.
    """

    def __init__(self, lag_features: int = 4, freq_base: int = 1000) -> None:
        """
        Initialize the lag encoder.

        Args:
            lag_features (int): Half-dimension of the lag embedding.
            freq_base (int): Base for exponentially decaying frequencies.
        """
        super().__init__()
        indices = torch.arange(lag_features, dtype=torch.float32)
        omega = freq_base ** (-indices / max(lag_features, 1))
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, lags: Tensor, batch_size: int, num_nodes: int) -> Tensor:
        """
        Encode lag indices and broadcast them to all nodes.

        Args:
            lags (Tensor): Lag indices. (L,).
            batch_size (int): Batch size.
            num_nodes (int): Number of mesh nodes.

        Returns:
            Tensor: Lag embedding. (B, L, N, 2 * C_LAG).
        """
        angles = lags.float().unsqueeze(-1) * self.omega.view(1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.view(1, -1, 1, emb.shape[-1]).expand(batch_size, -1, num_nodes, -1)


# ============================================================
# Sliding-History Kernel Blocks
# ============================================================


class HistoryAttention(nn.Module):
    """
    History-aware slice attention with annealed lag bias, spatial bias, and token gates.
    """

    def __init__(
        self,
        width: int,
        spatial_dim: int,
        num_slices: int,
        num_heads: int,
        use_bias: bool = True,
        use_gating: bool = True,
        bias_beta_init: float = 1.0,
        gate_beta_init: float = 1.0,
        space_tau_init: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize history attention.

        Args:
            width (int): Node token width.
            spatial_dim (int): Spatial coordinate dimension.
            num_slices (int): Number of slice tokens per frame.
            num_heads (int): Number of attention heads.
            use_bias (bool): Whether to add joint spatial-history log bias.
            use_gating (bool): Whether to apply slice token gating.
            bias_beta_init (float): Initial structural bias strength.
            gate_beta_init (float): Initial gate-logit strength.
            space_tau_init (float): Initial spatial Gaussian bandwidth.
            eps (float): Small stabilizer.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")

        self.spatial_dim = spatial_dim
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.use_bias = use_bias
        self.use_gating = use_gating
        self.eps = eps

        self.slice_proj = nn.Linear(width, num_slices)
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

        self.bias_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(bias_beta_init)), dtype=torch.float32))
        self.gate_beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(gate_beta_init)), dtype=torch.float32))
        self.space_tau_raw = nn.Parameter(torch.tensor(math.log(math.expm1(space_tau_init)), dtype=torch.float32))
        self.gate_mlp = nn.Sequential(
            nn.Linear(width + 2, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def _slice_tokens(self, x: Tensor, coords: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        weights = F.softmax(self.slice_proj(x), dim=-1)
        pop = weights.sum(dim=2)
        denom = pop.clamp_min(self.eps).unsqueeze(-1)
        slices = torch.einsum("blng,blnc->blgc", weights, x) / denom

        coords = coords.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        centers = torch.einsum("blng,blnd->blgd", weights, coords) / denom
        return weights, slices, centers, pop

    def _joint_log_bias(self, centers: Tensor, history_weights: Tensor) -> Tensor:
        center_current = centers[:, 0]
        delta = center_current[:, :, None, None, :] - centers[:, None, :, :, :]
        dist2 = delta.square().sum(dim=-1)
        tau2 = F.softplus(self.space_tau_raw).square().clamp_min(self.eps)
        space_bias = -dist2 / tau2

        history_bias = torch.log(history_weights.clamp_min(self.eps)).view(history_weights.shape[0], 1, -1, 1)
        bias = space_bias + history_bias
        bias = bias.reshape(centers.shape[0], self.num_slices, -1)
        return bias - torch.logsumexp(bias, dim=-1, keepdim=True)

    def forward(self, x: Tensor, coords: Tensor, history_weights: Tensor) -> Tensor:
        """
        Apply current-to-history slice attention.

        Args:
            x (Tensor): History node tokens. (B, L, N, C).
            coords (Tensor): Node coordinates. (B, N, D).
            history_weights (Tensor): Normalized lag weights. (B, L).

        Returns:
            Tensor: Current-frame node update. (B, N, C).
        """
        B, L, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        weights, slices, centers, pop = self._slice_tokens(x, coords)
        query = self.q_proj(slices[:, 0]).view(B, self.num_slices, H, D).transpose(1, 2)
        key = self.k_proj(slices).view(B, L * self.num_slices, H, D).transpose(1, 2)
        value = self.v_proj(slices).view(B, L * self.num_slices, C)

        if self.use_gating:
            pop_log = torch.log(pop.clamp_min(self.eps)).unsqueeze(-1)
            pi = history_weights[:, :, None, None].expand(-1, -1, self.num_slices, 1)
            gate = 2.0 * torch.sigmoid(self.gate_mlp(torch.cat([slices, pop_log, pi], dim=-1)))
            gate_flat = gate.reshape(B, L * self.num_slices, 1)
            value = value * gate_flat

        value = value.view(B, L * self.num_slices, H, D).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(D)

        if self.use_bias:
            bias = self._joint_log_bias(centers, history_weights)
            logits = logits + F.softplus(self.bias_beta_raw) * bias.unsqueeze(1)

        if self.use_gating:
            gate_bias = torch.log(gate_flat.squeeze(-1).clamp_min(self.eps))
            logits = logits + F.softplus(self.gate_beta_raw) * gate_bias[:, None, None, :]

        attn = torch.softmax(logits, dim=-1)
        out_slices = torch.matmul(attn, value).transpose(1, 2).contiguous().view(B, self.num_slices, C)
        out_slices = self.out_proj(out_slices)
        return torch.bmm(weights[:, 0], out_slices)


class HyperFlowBlock(nn.Module):
    """
    One pre-norm sliding-history HyperFlowNet block.
    """

    def __init__(
        self,
        width: int,
        spatial_dim: int,
        num_slices: int,
        num_heads: int,
        ffn_dim: int,
        use_bias: bool,
        use_gating: bool,
        bias_beta_init: float,
        gate_beta_init: float,
        space_tau_init: float,
    ) -> None:
        """
        Initialize one HyperFlowNet block.

        Args:
            width (int): Node token width.
            spatial_dim (int): Spatial coordinate dimension.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of attention heads.
            ffn_dim (int): Hidden width of the feed-forward block.
            use_bias (bool): Whether to use joint log bias.
            use_gating (bool): Whether to use slice token gating.
            bias_beta_init (float): Initial structural bias strength.
            gate_beta_init (float): Initial gate-logit strength.
            space_tau_init (float): Initial spatial Gaussian bandwidth.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.history_attn = HistoryAttention(
            width=width,
            spatial_dim=spatial_dim,
            num_slices=num_slices,
            num_heads=num_heads,
            use_bias=use_bias,
            use_gating=use_gating,
            bias_beta_init=bias_beta_init,
            gate_beta_init=gate_beta_init,
            space_tau_init=space_tau_init,
        )
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, width),
        )

    def forward(self, history: Tensor, coords: Tensor, history_weights: Tensor) -> Tensor:
        """
        Apply one residual current-frame update.

        Args:
            history (Tensor): History node tokens. (B, L, N, C).
            coords (Tensor): Node coordinates. (B, N, D).
            history_weights (Tensor): Normalized lag weights. (B, L).

        Returns:
            Tensor: Updated history tokens. (B, L, N, C).
        """
        current = history[:, 0] + self.history_attn(self.norm1(history), coords, history_weights)
        current = current + self.ffn(self.norm2(current))
        return torch.cat([current.unsqueeze(1), history[:, 1:]], dim=1)


# ============================================================
# HyperFlowNet
# ============================================================


class HyperFlowNet(nn.Module):
    """
    Annealed sliding-history kernel operator for flow prediction on fixed meshes.
    """

    uses_history = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 128,
        depth: int = 4,
        num_slices: int = 32,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        use_spatial_encoding: bool = True,
        use_time_encoding: bool = True,
        use_bias: bool = True,
        use_gating: bool = True,
        coord_features: int = 8,
        time_features: int = 4,
        lag_features: int = 4,
        freq_base: int = 1000,
        bias_beta_init: float = 1.0,
        gate_beta_init: float = 1.0,
        space_tau_init: float = 1.0,
    ) -> None:
        """
        Initialize HyperFlowNet.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden channel width.
            depth (int): Number of HyperFlowNet blocks.
            num_slices (int): Number of slice tokens.
            num_heads (int): Number of slice-space attention heads.
            ffn_dim (Optional[int]): Hidden width of the feed-forward block.
            use_spatial_encoding (bool): Whether to use Fourier spatial encoding.
            use_time_encoding (bool): Whether to use absolute time and lag encodings.
            use_bias (bool): Whether to use joint spatial-history log bias.
            use_gating (bool): Whether to use slice token gating.
            coord_features (int): Half-dimension of the Fourier spatial encoding.
            time_features (int): Half-dimension of the temporal encoding.
            lag_features (int): Half-dimension of the relative lag encoding.
            freq_base (int): Base for temporal frequencies.
            bias_beta_init (float): Initial structural bias strength.
            gate_beta_init (float): Initial gate-logit strength.
            space_tau_init (float): Initial spatial Gaussian bandwidth.
        """
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * width

        self.use_spatial_encoding = use_spatial_encoding and coord_features > 0
        self.use_time_encoding = use_time_encoding and time_features > 0
        self.use_lag_encoding = use_time_encoding and lag_features > 0

        if self.use_spatial_encoding:
            self.spatial_encoder = SpatialEncoder(spatial_dim=spatial_dim, coord_features=coord_features)
            coord_dim = 2 * coord_features
        else:
            self.spatial_encoder = None
            coord_dim = 0

        if self.use_time_encoding:
            self.time_encoder = TemporalEncoder(time_features=time_features, freq_base=freq_base)
            time_dim = 2 * time_features
        else:
            self.time_encoder = None
            time_dim = 0

        if self.use_lag_encoding:
            self.lag_encoder = LagEncoder(lag_features=lag_features, freq_base=freq_base)
            lag_dim = 2 * lag_features
        else:
            self.lag_encoder = None
            lag_dim = 0

        self.embed = nn.Linear(in_channels + coord_dim + time_dim + lag_dim, width)
        self.blocks = nn.ModuleList([
            HyperFlowBlock(
                width=width,
                spatial_dim=spatial_dim,
                num_slices=num_slices,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                use_bias=use_bias,
                use_gating=use_gating,
                bias_beta_init=bias_beta_init,
                gate_beta_init=gate_beta_init,
                space_tau_init=space_tau_init,
            )
            for _ in range(depth)
        ])
        self.proj = nn.Linear(width, out_channels)

    def _prepare_t_norm(self, t_norm: Optional[Tensor], batch_size: int, history_len: int, coords: Tensor) -> Tensor:
        if t_norm is None:
            return torch.zeros(batch_size, history_len, device=coords.device, dtype=coords.dtype)
        t_norm = t_norm.to(device=coords.device, dtype=coords.dtype)
        if t_norm.dim() == 1:
            t_norm = t_norm[:, None].expand(-1, history_len)
        return t_norm

    def _prepare_history_weights(self, history_weights: Optional[Tensor], batch_size: int, history_len: int, x: Tensor) -> Tensor:
        if history_weights is None:
            weights = x.new_ones(batch_size, history_len)
        else:
            weights = history_weights.to(device=x.device, dtype=x.dtype)
            if weights.dim() == 1:
                weights = weights.unsqueeze(0).expand(batch_size, -1)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        inputs: Tensor,
        coords: Tensor,
        t_norm: Optional[Tensor] = None,
        history_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Predict the next state on the mesh.

        Args:
            inputs (Tensor): History or current node features. (B, L, N, C_IN) or (B, N, C_IN).
            coords (Tensor): Node coordinates. (B, N, D).
            t_norm (Optional[Tensor]): Normalized history frame times. (B, L) or (B,).
            history_weights (Optional[Tensor]): Normalized lag weights. (L,) or (B, L).

        Returns:
            Tensor: Predicted next state. (B, N, C_OUT).
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)

        B, L, N, _ = inputs.shape
        components = [inputs]

        if self.spatial_encoder is not None:
            spatial = self.spatial_encoder(coords).unsqueeze(1).expand(-1, L, -1, -1)
            components.append(spatial)

        if self.time_encoder is not None:
            times = self._prepare_t_norm(t_norm, B, L, coords)
            components.append(self.time_encoder(times, N).to(dtype=inputs.dtype))

        if self.lag_encoder is not None:
            lags = torch.arange(L, device=inputs.device, dtype=inputs.dtype)
            components.append(self.lag_encoder(lags, B, N).to(dtype=inputs.dtype))

        x = self.embed(torch.cat(components, dim=-1))
        weights = self._prepare_history_weights(history_weights, B, L, x)

        for block in self.blocks:
            x = block(x, coords, weights)
        return self.proj(x[:, 0])

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
            for step_idx in tqdm(range(steps), desc="Predicting", leave=False, dynamic_ncols=True):
                t_norm = torch.full(
                    (inputs.shape[0],), step_idx / max(steps, 1), device=inputs.device, dtype=inputs.dtype
                )
                state = self.forward(state, coords, t_norm=t_norm)
                if bc is not None:
                    state = bc.enforce(state)
                states.append(state)
        return torch.stack(states, dim=1)
