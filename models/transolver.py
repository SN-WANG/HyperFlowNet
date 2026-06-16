# Transolver
# Author: Shengning Wang

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


ACTIVATIONS = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


def _build_activation(name: str) -> nn.Module:
    key = name.lower()
    factory = ACTIVATIONS[key]
    return factory() if isinstance(factory, type) else factory()


def _trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    with torch.no_grad():
        tensor.normal_(0.0, std)
        while True:
            mask = tensor.abs() > 2 * std
            if not mask.any():
                break
            tensor[mask] = torch.empty_like(tensor[mask]).normal_(0.0, std)
    return tensor


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    """
    Build sinusoidal timestep embeddings.

    Args:
        timesteps (Tensor): Time indices. (B,).
        dim (int): Embedding dimension.
        max_period (int): Minimum-frequency control.

    Returns:
        Tensor: Time embeddings. (B, C).
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        / max(half_dim, 1)
    )
    angles = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TransolverMLP(nn.Module):
    """
    Token-wise MLP used by Transolver.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 0,
        act: str = "gelu",
        res: bool = False,
    ) -> None:
        """
        Initialize the token-wise MLP.

        Args:
            in_channels (int): Input token width.
            hidden_channels (int): Hidden token width.
            out_channels (int): Output token width.
            num_layers (int): Number of hidden residual layers after the input lift.
            act (str): Activation name.
            res (bool): Whether to use residual hidden updates.
        """
        super().__init__()
        self.num_layers = num_layers
        self.res = res
        self.input_proj = nn.Sequential(nn.Linear(in_channels, hidden_channels), _build_activation(act))
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_channels, hidden_channels), _build_activation(act))
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the token-wise MLP.

        Args:
            x (Tensor): Input tokens. (B, N, C_IN).

        Returns:
            Tensor: Output tokens. (B, N, C_OUT).
        """
        x = self.input_proj(x)
        for layer in self.hidden_layers:
            x = layer(x) + x if self.res else layer(x)
        return self.output_proj(x)


class PhysicsAttention(nn.Module):
    """
    Irregular-mesh Physics Attention from Transolver.
    """

    def __init__(self, width: int, num_heads: int, num_slices: int, dropout: float = 0.0) -> None:
        """
        Initialize physics attention.

        Args:
            width (int): Token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of learned physical slices.
            dropout (float): Attention dropout.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        inner_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        self.in_project_x = nn.Linear(width, inner_dim)
        self.in_project_fx = nn.Linear(width, inner_dim)
        self.in_project_slice = nn.Linear(self.head_dim, num_slices)
        nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, width), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply physics attention on node tokens.

        Args:
            x (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Node token update. (B, N, C).
        """
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim

        fx_mid = self.in_project_fx(x).view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        x_mid = self.in_project_x(x).view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        slice_weights = torch.softmax(self.in_project_slice(x_mid) / self.temperature, dim=-1)
        slice_norm = slice_weights.sum(dim=2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) / slice_norm.clamp_min(1e-5).unsqueeze(-1)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out_slice = torch.matmul(self.dropout(attn), v)
        out = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out = out.transpose(1, 2).contiguous().view(B, N, H * D)
        return self.to_out(out)


class TransolverBlock(nn.Module):
    """
    One Transolver block with physics attention and feed-forward update.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        num_slices: int,
        out_channels: int,
        dropout: float = 0.0,
        act: str = "gelu",
        mlp_ratio: int = 1,
        last_layer: bool = False,
    ) -> None:
        """
        Initialize one Transolver block.

        Args:
            width (int): Token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of physical slices.
            out_channels (int): Output channel width.
            dropout (float): Dropout rate.
            act (str): Activation name.
            mlp_ratio (int): Expansion ratio in the block MLP.
            last_layer (bool): Whether this block returns predictions.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = PhysicsAttention(width=width, num_heads=num_heads, num_slices=num_slices, dropout=dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = TransolverMLP(width, width * mlp_ratio, width, num_layers=0, act=act, res=False)
        self.last_layer = last_layer
        if self.last_layer:
            self.norm3 = nn.LayerNorm(width)
            self.out_proj = nn.Linear(width, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply one Transolver block.

        Args:
            x (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Updated node tokens or predictions. (B, N, C) or (B, N, C_OUT).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        if self.last_layer:
            return self.out_proj(self.norm3(x))
        return x


class Transolver(nn.Module):
    """
    Transolver baseline adapted to the HyperFlowNet one-step interface.
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
        dropout: float = 0.0,
        act: str = "gelu",
        mlp_ratio: int = 1,
        use_time_input: bool = True,
        unified_pos: bool = True,
        ref: int = 8,
        ref_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize Transolver.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden channel width.
            depth (int): Number of Transolver blocks.
            num_slices (int): Number of physical slice tokens.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            act (str): Activation name.
            mlp_ratio (int): Expansion ratio in block MLPs.
            use_time_input (bool): Whether to add sinusoidal time embeddings.
            unified_pos (bool): Whether to use distance-to-reference-grid PE.
            ref (int): Reference grid resolution per spatial axis.
            ref_bounds (Optional[Sequence[Tuple[float, float]]]): Coordinate bounds for PE.
        """
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("width must be divisible by num_heads")

        self.spatial_dim = spatial_dim
        self.width = width
        self.use_time_input = use_time_input
        self.unified_pos = unified_pos
        self.ref = ref
        self.ref_bounds = self._normalize_ref_bounds(ref_bounds)

        coord_dim = ref ** spatial_dim if unified_pos else spatial_dim
        self.preprocess = TransolverMLP(in_channels + coord_dim, width * 2, width, num_layers=0, act=act, res=False)
        self.time_fc = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)) if use_time_input else None
        self.placeholder = nn.Parameter((1.0 / width) * torch.rand(width, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            TransolverBlock(
                width=width,
                num_heads=num_heads,
                num_slices=num_slices,
                out_channels=out_channels,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                last_layer=(layer_idx == depth - 1),
            )
            for layer_idx in range(depth)
        ])
        self._initialize_weights()

    def _normalize_ref_bounds(
        self,
        ref_bounds: Optional[Sequence[Tuple[float, float]]],
    ) -> Tuple[Tuple[float, float], ...]:
        if ref_bounds is None:
            return tuple((-1.0, 1.0) for _ in range(self.spatial_dim))
        return tuple((float(lower), float(upper)) for lower, upper in ref_bounds)

    def _initialize_weights(self) -> None:
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            _trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def _build_reference_points(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        axes = [torch.linspace(lower, upper, self.ref, device=device, dtype=dtype) for lower, upper in self.ref_bounds]
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(mesh, dim=-1).reshape(-1, self.spatial_dim)

    def _coordinate_features(self, coords: Tensor) -> Tensor:
        if not self.unified_pos:
            return coords
        ref_points = self._build_reference_points(coords.device, coords.dtype)
        return torch.sqrt(torch.sum((coords[:, :, None, :] - ref_points[None, None, :, :]) ** 2, dim=-1))

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
        coord_features = self._coordinate_features(coords)
        x = self.preprocess(torch.cat([coord_features, inputs], dim=-1))
        x = x + self.placeholder.view(1, 1, -1)

        if self.time_fc is not None and t_norm is not None:
            time_emb = timestep_embedding(t_norm, self.width).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = x + self.time_fc(time_emb)

        for block in self.blocks:
            x = block(x)
        return x

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
                t_norm = None
                if self.time_fc is not None:
                    t_norm = torch.full(
                        (inputs.shape[0],), step_idx / max(steps, 1), device=inputs.device, dtype=inputs.dtype
                    )
                state = self.forward(state, coords, t_norm=t_norm)
                if bc is not None:
                    state = bc.enforce(state)
                states.append(state)
        return torch.stack(states, dim=1)
