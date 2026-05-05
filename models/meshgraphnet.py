# MeshGraphNets
# Author: Shengning Wang

from typing import Optional

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


class MLP(nn.Module):
    """
    Compact MLP for MeshGraphNet message functions.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, depth: int = 2) -> None:
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


class MeshGraphNetBlock(nn.Module):
    """
    Edge-to-node message passing block from MeshGraphNet.
    """

    def __init__(self, width: int, edge_dim: int) -> None:
        """
        Initialize one MeshGraphNet block.

        Args:
            width (int): Node latent width.
            edge_dim (int): Edge latent width.
        """
        super().__init__()
        self.edge_mlp = MLP(2 * width + edge_dim, width, width, depth=2)
        self.node_mlp = MLP(2 * width, width, width, depth=2)
        self.edge_norm = nn.LayerNorm(width)
        self.node_norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, edge_attr: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """
        Apply one message passing block.

        Args:
            x (Tensor): Node tokens. (B, N, C).
            edge_attr (Tensor): Edge tokens. (B, E, C_EDGE).
            edge_index (Tensor): Directed edge list. (2, E).

        Returns:
            tuple[Tensor, Tensor]: Updated node and edge tokens.
        """
        src, dst = edge_index
        B, N, C = x.shape
        edge_input = torch.cat([x[:, src], x[:, dst], edge_attr], dim=-1)
        edge_update = self.edge_norm(self.edge_mlp(edge_input))

        agg = x.new_zeros(B, N, C)
        agg.index_add_(1, dst, edge_update)
        degree = x.new_zeros(N, 1)
        degree.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=x.device, dtype=x.dtype))
        agg = agg / degree.clamp_min(1.0).unsqueeze(0)

        node_update = self.node_norm(self.node_mlp(torch.cat([x, agg], dim=-1)))
        return x + node_update, edge_update


class MeshGraphNet(nn.Module):
    """
    MeshGraphNet baseline using kNN edges inferred from mesh coordinates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        edge_index: Tensor,
        width: int = 128,
        depth: int = 4,
    ) -> None:
        """
        Initialize MeshGraphNet.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            edge_index (Tensor): Undirected edge list. (2, E_EDGE).
            width (int): Hidden channel width.
            depth (int): Number of message passing blocks.
        """
        super().__init__()
        rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        self.register_buffer("edge_index", torch.cat([edge_index, rev_edge_index], dim=1).long(), persistent=False)
        self.node_encoder = MLP(in_channels + spatial_dim, width, width, depth=2)
        self.edge_encoder = MLP(spatial_dim + 1, width, width, depth=2)
        self.blocks = nn.ModuleList([MeshGraphNetBlock(width=width, edge_dim=width) for _ in range(depth)])
        self.proj = MLP(width, width, out_channels, depth=1)

    def _edge_features(self, coords: Tensor) -> Tensor:
        src, dst = self.edge_index
        rel = coords[:, dst] - coords[:, src]
        dist = torch.norm(rel, dim=-1, keepdim=True)
        return torch.cat([rel, dist], dim=-1)

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
        x = self.node_encoder(torch.cat([coords, inputs], dim=-1))
        edge_attr = self.edge_encoder(self._edge_features(coords))
        for block in self.blocks:
            x, edge_attr = block(x, edge_attr, self.edge_index)
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
