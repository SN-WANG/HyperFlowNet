# Geometry-Aware Fourier Neural Operator (Geo-FNO)
# Author: Shengning Wang

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm


class DeformationNet(nn.Module):
    """
    Learnable coordinate deformation for Geo-FNO.
    """

    def __init__(self, spatial_dim: int, hidden_dim: int = 32, depth: int = 3) -> None:
        """
        Initialize the deformation network.

        Args:
            spatial_dim (int): Spatial coordinate dimension.
            hidden_dim (int): Hidden feature width.
            depth (int): Number of MLP layers.
        """
        super().__init__()
        layers = []
        in_dim = spatial_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, spatial_dim))
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coords: Tensor) -> Tensor:
        """
        Map physical coordinates to latent coordinates.

        Args:
            coords (Tensor): Normalized physical coordinates. (B, N, D).

        Returns:
            Tensor: Latent coordinates. (B, N, D).
        """
        return coords + self.net(coords)


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


class GeoFNO(nn.Module):
    """
    Geometry-aware Fourier Neural Operator for irregular node data.
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
    ) -> None:
        """
        Initialize Geo-FNO.

        Args:
            in_channels (int): Number of node input channels.
            out_channels (int): Number of node output channels.
            spatial_dim (int): Spatial coordinate dimension.
            width (int): Hidden channel width.
            depth (int): Number of FNO blocks.
            modes (Optional[list[int]]): Retained Fourier modes per spatial dimension.
            grid_size (Optional[list[int]]): Latent grid resolution per spatial dimension.
        """
        super().__init__()
        if modes is None:
            modes = [12] * spatial_dim
        if grid_size is None:
            grid_size = [64] * spatial_dim

        self.spatial_dim = spatial_dim
        self.width = width
        self.grid_size = grid_size
        self.deformation = DeformationNet(spatial_dim=spatial_dim)
        self.lift = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([FNOBlock(width=width, modes=modes) for _ in range(depth)])
        self.proj1 = nn.Linear(width, width)
        self.proj2 = nn.Linear(width, out_channels)

    def _latent_coords(self, coords: Tensor) -> Tensor:
        return self.deformation(coords).clamp(-1.0, 1.0)

    def _flat_grid_index(self, latent_coords: Tensor) -> Tensor:
        B, N, _ = latent_coords.shape
        dims = torch.tensor(self.grid_size, device=latent_coords.device).view(1, 1, -1)
        index = ((latent_coords + 1.0) * 0.5 * (dims - 1)).round().long()
        for dim_idx, dim_size in enumerate(self.grid_size):
            index[..., dim_idx] = index[..., dim_idx].clamp(0, dim_size - 1)

        strides = []
        stride = 1
        for dim_size in reversed(self.grid_size):
            strides.append(stride)
            stride *= dim_size
        strides = torch.tensor(strides[::-1], device=latent_coords.device).view(1, 1, -1)
        batch_idx = torch.arange(B, device=latent_coords.device).view(B, 1).expand(B, N)
        return (batch_idx * stride + (index * strides).sum(dim=-1)).reshape(-1)

    def _p2g(self, features: Tensor, latent_coords: Tensor) -> Tensor:
        B, _, _ = features.shape
        flat_index = self._flat_grid_index(latent_coords)
        total_grid_size = 1
        for dim_size in self.grid_size:
            total_grid_size *= dim_size

        grid = features.new_zeros(B * total_grid_size, self.width)
        count = features.new_zeros(B * total_grid_size, 1)
        grid.index_add_(0, flat_index, features.reshape(-1, self.width))
        count.index_add_(0, flat_index, torch.ones(flat_index.shape[0], 1, device=features.device, dtype=features.dtype))
        grid = grid / count.clamp_min(1.0)
        grid = grid.view(B, *self.grid_size, self.width)
        return grid.permute(0, self.spatial_dim + 1, *range(1, self.spatial_dim + 1)).contiguous()

    def _g2p(self, grid: Tensor, latent_coords: Tensor) -> Tensor:
        B, N, _ = latent_coords.shape
        if self.spatial_dim == 2:
            sample_coords = latent_coords[..., [1, 0]].view(B, N, 1, 2)
            sampled = F.grid_sample(grid, sample_coords, align_corners=True, padding_mode="border")
            return sampled.squeeze(-1).permute(0, 2, 1)

        sample_coords = latent_coords[..., [2, 1, 0]].view(B, N, 1, 1, 3)
        sampled = F.grid_sample(grid, sample_coords, align_corners=True, padding_mode="border")
        return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

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
        latent_coords = self._latent_coords(coords)
        grid = self._p2g(self.lift(inputs), latent_coords)
        for block in self.blocks:
            grid = block(grid)
        x = self._g2p(grid, latent_coords)
        return self.proj2(F.gelu(self.proj1(x)))

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
