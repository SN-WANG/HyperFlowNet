# Geometry-Informed Neural Operator (GINO)
# Author: Shengning Wang

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm


class MLP(nn.Module):
    """
    Compact MLP for local integral kernels.
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


class SpectralConv2d(nn.Module):
    """
    Two-dimensional Fourier spectral convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: list[int]) -> None:
        """
        Initialize the spectral convolution.

        Args:
            in_channels (int): Input channel width.
            out_channels (int): Output channel width.
            modes (list[int]): Retained Fourier modes along both dimensions.
        """
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes[0], modes[1], dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes[0], modes[1], dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spectral convolution.

        Args:
            x (Tensor): Latent grid features. (B, C_IN, H, W).

        Returns:
            Tensor: Updated grid features. (B, C_OUT, H, W).
        """
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, s=(H, W))
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes[0], :self.modes[1]] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes[0], :self.modes[1]], self.weights1
        )
        out_ft[:, :, -self.modes[0]:, :self.modes[1]] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes[0]:, :self.modes[1]], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(H, W))


class SpectralConv3d(nn.Module):
    """
    Three-dimensional Fourier spectral convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, modes: list[int]) -> None:
        """
        Initialize the spectral convolution.

        Args:
            in_channels (int): Input channel width.
            out_channels (int): Output channel width.
            modes (list[int]): Retained Fourier modes along all dimensions.
        """
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, *modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, *modes, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, *modes, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, *modes, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spectral convolution.

        Args:
            x (Tensor): Latent grid features. (B, C_IN, H, W, L).

        Returns:
            Tensor: Updated grid features. (B, C_OUT, H, W, L).
        """
        B, _, H, W, L = x.shape
        m1, m2, m3 = self.modes
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(B, self.out_channels, H, W, L // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -m1:, -m2:, :m3], self.weights4)
        return torch.fft.irfftn(out_ft, s=(H, W, L))


class FNOBlock(nn.Module):
    """
    One FNO block in latent grid space.
    """

    def __init__(self, width: int, modes: list[int]) -> None:
        """
        Initialize the FNO block.

        Args:
            width (int): Hidden channel width.
            modes (list[int]): Retained Fourier modes per spatial dimension.
        """
        super().__init__()
        if len(modes) == 2:
            self.spectral = SpectralConv2d(width, width, modes)
            self.pointwise = nn.Conv2d(width, width, 1)
            self.norm = nn.InstanceNorm2d(width)
        else:
            self.spectral = SpectralConv3d(width, width, modes)
            self.pointwise = nn.Conv3d(width, width, 1)
            self.norm = nn.InstanceNorm3d(width)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply one FNO block.

        Args:
            x (Tensor): Latent grid features. (B, C, ...).

        Returns:
            Tensor: Updated grid features. (B, C, ...).
        """
        return F.gelu(self.norm(self.spectral(x) + self.pointwise(x)))


class LocalIntegral(nn.Module):
    """
    Local graph neural operator integral between source and query coordinates.
    """

    def __init__(self, feature_dim: int, spatial_dim: int, width: int, neighbors: int = 16) -> None:
        """
        Initialize the local integral layer.

        Args:
            feature_dim (int): Source feature width.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Output feature width.
            neighbors (int): Number of nearest source points per query point.
        """
        super().__init__()
        self.neighbors = neighbors
        self.kernel = MLP(feature_dim + 3 * spatial_dim + 1, width, width, depth=2)

    def forward(self, source_coords: Tensor, query_coords: Tensor, source_features: Tensor) -> Tensor:
        """
        Integrate source features onto query coordinates.

        Args:
            source_coords (Tensor): Source coordinates. (B, N, D).
            query_coords (Tensor): Query coordinates. (B, M, D).
            source_features (Tensor): Source features. (B, N, C).

        Returns:
            Tensor: Query features. (B, M, C_OUT).
        """
        B, _, D = source_coords.shape
        M = query_coords.shape[1]
        k = min(self.neighbors, source_coords.shape[1])
        dist = torch.cdist(query_coords, source_coords)
        nn_dist, nn_idx = torch.topk(dist, k=k, dim=-1, largest=False)

        batch_idx = torch.arange(B, device=source_coords.device).view(B, 1, 1)
        src_c = source_coords[batch_idx, nn_idx]
        src_f = source_features[batch_idx, nn_idx]
        query = query_coords.unsqueeze(2).expand(-1, -1, k, -1)
        rel = query - src_c
        edge_input = torch.cat([query, src_c, rel, nn_dist.unsqueeze(-1), src_f], dim=-1)

        sigma = nn_dist[..., -1:].median().clamp_min(1e-6)
        weights = torch.exp(-nn_dist.unsqueeze(-1) / sigma)
        values = self.kernel(edge_input)
        return (weights * values).sum(dim=2) / weights.sum(dim=2).clamp_min(1e-6)


class GINO(nn.Module):
    """
    Geometry-informed neural operator with GNO-FNO-GNO structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dim: int,
        width: int = 128,
        depth: int = 4,
        modes: Optional[list[int]] = None,
        grid_size: Optional[list[int]] = None,
        neighbors: int = 16,
    ) -> None:
        """
        Initialize GINO.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden channel width.
            depth (int): Number of latent FNO blocks.
            modes (Optional[list[int]]): Retained Fourier modes per spatial dimension.
            grid_size (Optional[list[int]]): Latent grid resolution per spatial dimension.
            neighbors (int): Number of local integral neighbors.
        """
        super().__init__()
        if modes is None:
            modes = [12] * spatial_dim
        if grid_size is None:
            grid_size = [64] * spatial_dim

        self.spatial_dim = spatial_dim
        self.width = width
        self.grid_size = grid_size
        self.input_lift = MLP(in_channels, width, width, depth=1)
        self.input_gno = LocalIntegral(width, spatial_dim, width, neighbors=neighbors)
        self.blocks = nn.ModuleList([FNOBlock(width=width, modes=modes) for _ in range(depth)])
        self.output_gno = LocalIntegral(width, spatial_dim, width, neighbors=neighbors)
        self.proj = MLP(width, width, out_channels, depth=1)

    def _latent_grid(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        axes = [torch.linspace(-1.0, 1.0, steps=size, device=device, dtype=dtype) for size in self.grid_size]
        mesh = torch.meshgrid(*axes, indexing="ij")
        grid = torch.stack(mesh, dim=-1).reshape(1, -1, self.spatial_dim)
        return grid.expand(batch_size, -1, -1)

    def _grid_features(self, features: Tensor) -> Tensor:
        B = features.shape[0]
        grid = features.view(B, *self.grid_size, self.width)
        return grid.permute(0, self.spatial_dim + 1, *range(1, self.spatial_dim + 1)).contiguous()

    def _point_features(self, features: Tensor) -> Tensor:
        B = features.shape[0]
        return features.permute(0, *range(2, self.spatial_dim + 2), 1).reshape(B, -1, self.width)

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
        latent_coords = self._latent_grid(inputs.shape[0], inputs.device, inputs.dtype)
        node_features = self.input_lift(inputs)
        latent_features = self.input_gno(coords, latent_coords, node_features)
        grid = self._grid_features(latent_features)
        for block in self.blocks:
            grid = block(grid)
        latent_features = self._point_features(grid)
        output_features = self.output_gno(latent_coords, coords, latent_features)
        return self.proj(output_features)

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
