# Transolver
# Author: Shengning Wang

from typing import Optional

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


class MLP(nn.Module):
    """
    Compact MLP for token projection.
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


class PhysicsAttention(nn.Module):
    """
    Physics-attention layer from Transolver for irregular meshes.
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
        slice_weights = torch.softmax(self.in_project_slice(x_mid) / self.temperature.clamp_min(0.1), dim=-1)
        slice_norm = slice_weights.sum(dim=2).clamp_min(1e-5)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights) / slice_norm.unsqueeze(-1)

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

    def __init__(self, width: int, num_heads: int, num_slices: int, dropout: float = 0.0) -> None:
        """
        Initialize one Transolver block.

        Args:
            width (int): Token width.
            num_heads (int): Number of attention heads.
            num_slices (int): Number of physical slices.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = PhysicsAttention(width=width, num_heads=num_heads, num_slices=num_slices, dropout=dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = MLP(width, 4 * width, width, depth=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply one Transolver block.

        Args:
            x (Tensor): Node tokens. (B, N, C).

        Returns:
            Tensor: Updated node tokens. (B, N, C).
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
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
        """
        super().__init__()
        self.embed = nn.Linear(in_channels + spatial_dim, width)
        self.blocks = nn.ModuleList([
            TransolverBlock(width=width, num_heads=num_heads, num_slices=num_slices, dropout=dropout)
            for _ in range(depth)
        ])
        self.proj = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_channels))

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
        x = self.embed(torch.cat([coords, inputs], dim=-1))
        for block in self.blocks:
            x = block(x)
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
