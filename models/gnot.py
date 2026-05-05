# Graph Neural Operator Transformer (GNOT)
# Author: Shengning Wang

from typing import Optional

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


class MLP(nn.Module):
    """
    Compact MLP for GNOT token projection.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, depth: int = 1) -> None:
        """
        Initialize the MLP.

        Args:
            in_channels (int): Input feature width.
            hidden_channels (int): Hidden feature width.
            out_channels (int): Output feature width.
            depth (int): Number of hidden layers.
        """
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_channels), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_channels, hidden_channels), nn.GELU()]
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the MLP.

        Args:
            x (Tensor): Input features. (..., C_IN).

        Returns:
            Tensor: Output features. (..., C_OUT).
        """
        return self.net(x)


class LinearAttention(nn.Module):
    """
    Linear normalized attention used by GNOT.
    """

    def __init__(self, width: int, num_heads: int, dropout: float = 0.0) -> None:
        """
        Initialize linear attention.

        Args:
            width (int): Token width.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Apply linear attention.

        Args:
            x (Tensor): Query tokens. (B, N, C).
            y (Optional[Tensor]): Key-value tokens. (B, M, C).

        Returns:
            Tensor: Attended tokens. (B, N, C).
        """
        if y is None:
            y = x
        B, N, C = x.shape
        M = y.shape[1]
        H, D = self.num_heads, self.head_dim

        q = self.query(x).view(B, N, H, D).transpose(1, 2).softmax(dim=-1)
        k = self.key(y).view(B, M, H, D).transpose(1, 2).softmax(dim=-1)
        v = self.value(y).view(B, M, H, D).transpose(1, 2)
        denom = (q * k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out = (q @ (k.transpose(-2, -1) @ v)) / denom + q
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.proj(self.dropout(out))


class GNOTBlock(nn.Module):
    """
    GNOT block with branch-to-trunk cross attention, self attention, and geometric gating.
    """

    def __init__(self, width: int, spatial_dim: int, num_heads: int, num_experts: int, dropout: float = 0.0) -> None:
        """
        Initialize one GNOT block.

        Args:
            width (int): Token width.
            spatial_dim (int): Spatial coordinate dimension.
            num_heads (int): Number of attention heads.
            num_experts (int): Number of geometric experts.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.cross_norm = nn.LayerNorm(width)
        self.branch_norm = nn.LayerNorm(width)
        self.self_norm = nn.LayerNorm(width)
        self.cross_attn = LinearAttention(width=width, num_heads=num_heads, dropout=dropout)
        self.self_attn = LinearAttention(width=width, num_heads=num_heads, dropout=dropout)
        self.gate = MLP(spatial_dim, 2 * width, num_experts, depth=2)
        self.mlp1 = nn.ModuleList([MLP(width, 4 * width, width, depth=1) for _ in range(num_experts)])
        self.mlp2 = nn.ModuleList([MLP(width, 4 * width, width, depth=1) for _ in range(num_experts)])

    def _moe(self, x: Tensor, coords: Tensor, experts: nn.ModuleList) -> Tensor:
        gates = torch.softmax(self.gate(coords), dim=-1).unsqueeze(-2)
        values = torch.stack([expert(x) for expert in experts], dim=-1)
        return (values * gates).sum(dim=-1)

    def forward(self, x: Tensor, branch: Tensor, coords: Tensor) -> Tensor:
        """
        Apply one GNOT block.

        Args:
            x (Tensor): Trunk tokens. (B, N, C).
            branch (Tensor): Branch tokens. (B, M, C).
            coords (Tensor): Node coordinates. (B, N, D).

        Returns:
            Tensor: Updated trunk tokens. (B, N, C).
        """
        x = x + self.cross_attn(self.cross_norm(x), self.branch_norm(branch))
        x = x + self._moe(x, coords, self.mlp1)
        x = x + self.self_attn(self.self_norm(x))
        x = x + self._moe(x, coords, self.mlp2)
        return x


class GNOT(nn.Module):
    """
    GNOT baseline adapted to batched node tensors without DGL.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        num_experts: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize GNOT.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden channel width.
            depth (int): Number of GNOT blocks.
            num_heads (int): Number of attention heads.
            num_experts (int): Number of geometric experts.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.trunk = MLP(in_channels + spatial_dim, width, width, depth=2)
        self.branch = MLP(in_channels + spatial_dim, width, width, depth=2)
        self.blocks = nn.ModuleList([
            GNOTBlock(width=width, spatial_dim=spatial_dim, num_heads=num_heads, num_experts=num_experts, dropout=dropout)
            for _ in range(depth)
        ])
        self.proj = MLP(width, width, out_channels, depth=1)

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
        features = torch.cat([coords, inputs], dim=-1)
        x = self.trunk(features)
        branch = self.branch(features)
        for block in self.blocks:
            x = block(x, branch, coords)
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
