# Shared 1D/2D building blocks for neural operators
# Author: Shengning Wang

import math

import torch
import torch.nn.functional as F
from torch import nn


def conv(ndim: int, c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Module:
    """1D or 2D convolution with same padding."""
    if ndim == 1:
        return nn.Conv1d(c_in, c_out, k, stride, padding=k // 2)
    return nn.Conv2d(c_in, c_out, k, stride, padding=k // 2)


def group_norm(channels: int) -> nn.Module:
    """GroupNorm with at most 8 groups, valid for 1D and 2D inputs."""
    return nn.GroupNorm(min(8, channels), channels)


def pool2(x: torch.Tensor) -> torch.Tensor:
    """Average pool by a factor of 2 on all spatial axes."""
    if x.ndim == 3:
        return 0.5 * (x[..., 0::2] + x[..., 1::2])
    return 0.25 * (x[..., 0::2, 0::2] + x[..., 0::2, 1::2] + x[..., 1::2, 0::2] + x[..., 1::2, 1::2])


def up2(x: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor upsample by a factor of 2 on all spatial axes."""
    if x.ndim == 3:
        return torch.repeat_interleave(x, 2, dim=-1)
    return torch.repeat_interleave(torch.repeat_interleave(x, 2, dim=-1), 2, dim=-2)


def coords(x: torch.Tensor) -> torch.Tensor:
    """Normalized coordinate grid appended as extra channels. (B, C, *S) -> (B, C+2, *S)."""
    grid = torch.meshgrid(
        *[torch.linspace(0.0, 1.0, s, device=x.device, dtype=x.dtype) for s in x.shape[2:]], indexing="ij"
    )
    extra = torch.stack([g.expand_as(x[:, :1]) for g in grid], dim=1)
    return torch.cat([x, extra], dim=1)


class ConvBlock(nn.Module):
    """Two convolutions with GELU and group norm. (B, C, *S) -> (B, C_OUT, *S)."""

    def __init__(self, ndim: int, c_in: int, c_out: int, k: int = 3) -> None:
        super().__init__()
        self.c1 = conv(ndim, c_in, c_out, k)
        self.n1 = group_norm(c_out)
        self.c2 = conv(ndim, c_out, c_out, k)
        self.n2 = group_norm(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(h)))


class SpectralConv1d(nn.Module):
    """Fourier spectral convolution for 1D fields. (B, C, N) -> (B, C, N)."""

    def __init__(self, c_in: int, c_out: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.weight = nn.Parameter(scale * torch.rand(c_in, c_out, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, self.weight.shape[1], N // 2 + 1, device=x.device, dtype=torch.complex64)
        w = torch.view_as_complex(self.weight)
        out_ft[:, :, : self.modes] = torch.einsum("bcn,com->bom", x_ft[:, :, : self.modes], w)
        return torch.fft.irfft(out_ft, n=N)


class SpectralConv2d(nn.Module):
    """Fourier spectral convolution for 2D fields. (B, C, H, W) -> (B, C, H, W)."""

    def __init__(self, c_in: int, c_out: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.weight = nn.Parameter(scale * torch.rand(c_in, c_out, modes, modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.weight.shape[1], H, W // 2 + 1, device=x.device, dtype=torch.complex64)
        w = torch.view_as_complex(self.weight)
        out_ft[:, :, : self.modes, : self.modes] = torch.einsum("bcnm,comn->bomn", x_ft[:, :, : self.modes, : self.modes], w)
        return torch.fft.irfft2(out_ft, s=(H, W))


class PatchEmbed(nn.Module):
    """Patch embedding by strided convolution without padding. (B, C, *S) -> (B, D, *S')."""

    def __init__(self, ndim: int, c_in: int, dim: int, patch: int) -> None:
        super().__init__()
        if ndim == 1:
            self.proj = nn.Conv1d(c_in, dim, patch, stride=patch)
        else:
            self.proj = nn.Conv2d(c_in, dim, patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Pre-LN transformer block with self-attention. (B, N, D) -> (B, N, D)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        return x + self.mlp(self.norm2(x))


def haar_1d(x: torch.Tensor) -> torch.Tensor:
    """1D Haar decomposition: (B, C, N) -> (B, C, N) with low half then high half."""
    even, odd = x[..., 0::2], x[..., 1::2]
    lo = (even + odd) / math.sqrt(2.0)
    hi = (even - odd) / math.sqrt(2.0)
    return torch.cat([lo, hi], dim=-1)


def ihaar_1d(y: torch.Tensor) -> torch.Tensor:
    """1D Haar reconstruction of haar_1d output. (B, C, N) -> (B, C, N)."""
    n = y.shape[-1] // 2
    lo, hi = y[..., :n], y[..., n:]
    even = (lo + hi) / math.sqrt(2.0)
    odd = (lo - hi) / math.sqrt(2.0)
    out = torch.empty_like(y)
    out[..., 0::2] = even
    out[..., 1::2] = odd
    return out


def haar_2d(x: torch.Tensor) -> torch.Tensor:
    """2D Haar decomposition: (B, C, H, W) -> (B, 4C, H/2, W/2)."""
    h = haar_1d(x)
    h = haar_1d(h.transpose(-1, -2)).transpose(-1, -2)
    B, C, H, W = h.shape
    hh = H // 2
    ww = W // 2
    return torch.cat(
        [h[:, :, :hh, :ww], h[:, :, :hh, ww:], h[:, :, hh:, :ww], h[:, :, hh:, ww:]], dim=1
    )


def ihaar_2d(y: torch.Tensor) -> torch.Tensor:
    """2D Haar reconstruction of haar_2d output. (B, 4C, H, W) -> (B, C, 2H, 2W)."""
    B, C4, H, W = y.shape
    C = C4 // 4
    ll, lh, hl, hh = y[:, :C], y[:, C : 2 * C], y[:, 2 * C : 3 * C], y[:, 3 * C :]
    w_lo = torch.cat([ll, hl], dim=-2)
    w_hi = torch.cat([lh, hh], dim=-2)
    h = torch.cat([w_lo, w_hi], dim=-1)
    h = ihaar_1d(h.transpose(-1, -2)).transpose(-1, -2)
    return ihaar_1d(h)
