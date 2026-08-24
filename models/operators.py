# Baseline neural operators for 1D and 2D fields
# Author: Shengning Wang

import torch
import torch.nn.functional as F
from torch import nn

from models.blocks import (
    ConvBlock,
    PatchEmbed,
    SpectralConv1d,
    SpectralConv2d,
    TransformerBlock,
    conv,
    group_norm,
    haar_1d,
    haar_2d,
    ihaar_1d,
    ihaar_2d,
    pool2,
    up2,
)


def _width(cfg: dict | None, default: int = 32) -> int:
    return int((cfg or {}).get("width", default))


def _depth(cfg: dict | None, default: int = 4) -> int:
    return int((cfg or {}).get("depth", default))


def _modes(cfg: dict | None, default: int = 16) -> int:
    return int((cfg or {}).get("modes", default))


def _heads(cfg: dict | None, default: int = 4) -> int:
    return int((cfg or {}).get("heads", default))


def _dim(cfg: dict | None, default: int = 64) -> int:
    return int((cfg or {}).get("dim", default))


def _patch(cfg: dict | None, default: int = 8) -> int:
    return int((cfg or {}).get("patch_size", default))


def _spectral(ndim: int, c_in: int, c_out: int, modes: int) -> nn.Module:
    return SpectralConv1d(c_in, c_out, modes) if ndim == 1 else SpectralConv2d(c_in, c_out, modes)


def _residual_skip(x: torch.Tensor, out: torch.Tensor, c_in: int, c_out: int, scale: float = 0.1) -> torch.Tensor:
    return x + scale * out if c_in == c_out else scale * out


class FNO(nn.Module):
    """Fourier Neural Operator with spectral convolutions and a residual skip. (B, C, *S) -> (B, C_OUT, *S)."""

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        super().__init__()
        w, d, m = _width(cfg), _depth(cfg), _modes(cfg)
        self.c_in = c_in
        self.c_out = c_in if c_out is None else c_out
        self.ndim = ndim
        self.lift = conv(ndim, c_in, w, k=1)
        self.spectrals = nn.ModuleList([_spectral(ndim, w, w, m) for _ in range(d)])
        self.norms = nn.ModuleList([group_norm(w) for _ in range(d)])
        self.project = conv(ndim, w, self.c_out, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for s, n in zip(self.spectrals, self.norms):
            h = F.gelu(n(s(h)))
        return _residual_skip(x, self.project(h), self.c_in, self.c_out)


class UNet(nn.Module):
    """Encoder-decoder network with skip connections. (B, C, *S) -> (B, C_OUT, *S)."""

    def __init__(
        self,
        c_in: int,
        ndim: int,
        cfg: dict | None = None,
        c_out: int | None = None,
        width: int | None = None,
        depth: int | None = None,
    ) -> None:
        super().__init__()
        w = width or _width(cfg)
        d = depth or _depth(cfg)
        self.ndim = ndim
        self.c_out = c_in if c_out is None else c_out
        self.enc = nn.ModuleList()
        cin = c_in
        for _ in range(d):
            self.enc.append(ConvBlock(ndim, cin, w))
            cin = w
        self.bottleneck = ConvBlock(ndim, cin, w)
        self.dec = nn.ModuleList([ConvBlock(ndim, 2 * w, w) for _ in range(d)])
        self.head = conv(ndim, w, self.c_out, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for e in self.enc:
            h = e(h)
            skips.append(h)
            h = pool2(h)
        h = self.bottleneck(h)
        for dblk, s in zip(self.dec, reversed(skips)):
            h = up2(h)
            h = dblk(torch.cat([h, s], dim=1))
        return self.head(h)


class ViT(nn.Module):
    """Vision-transformer surrogate with learned positional embedding. (B, C, *S) -> (B, C_OUT, *S)."""

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        super().__init__()
        dim, heads, patch, d = _dim(cfg), _heads(cfg), _patch(cfg), _depth(cfg, 4)
        self.ndim = ndim
        self.patch = patch
        self.c_in = c_in
        self.c_out = c_in if c_out is None else c_out
        self.embed = PatchEmbed(ndim, c_in, dim, patch)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(d)])
        self.norm = nn.LayerNorm(dim)
        self.head = conv(ndim, dim, self.c_out, k=1)
        self._pos = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        s_patch = tuple(s // self.patch for s in x.shape[2:])
        n_tokens = 1
        for s in s_patch:
            n_tokens *= s
        if self._pos is None or self._pos.shape[0] != n_tokens or self._pos.device != x.device:
            self._pos = nn.Parameter(torch.zeros(n_tokens, self.embed.proj.out_channels), requires_grad=False)
            nn.init.normal_(self._pos, std=0.02)
        pos = self._pos.to(x.device)
        h = self.embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        h = h + pos[None]
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).transpose(1, 2).reshape(B, -1, *s_patch)
        h = F.interpolate(h, size=x.shape[2:], mode="nearest")
        return _residual_skip(x, self.head(h), self.c_in, self.c_out)


class WNO(nn.Module):
    """Wavelet Neural Operator: spectral convolution on one-level Haar coefficients."""

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        super().__init__()
        w, m = _width(cfg), _modes(cfg)
        mult = 1 if ndim == 1 else 4
        self.ndim = ndim
        self.c_in = c_in
        self.c_out = c_in if c_out is None else c_out
        self.lift = conv(ndim, c_in, w, k=1)
        self.spectral = _spectral(ndim, w * mult, w * mult, m)
        self.norm = group_norm(w * mult)
        self.project = conv(ndim, w, self.c_out, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        c = haar_1d(h) if self.ndim == 1 else haar_2d(h)
        c = F.gelu(self.norm(self.spectral(c)))
        h = ihaar_1d(c) if self.ndim == 1 else ihaar_2d(c)
        return _residual_skip(x, self.project(h), self.c_in, self.c_out)


class UWNO(nn.Module):
    """U-shaped wavelet operator: Haar decomposition, U-Net refinement, reconstruction."""

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        super().__init__()
        w = _width(cfg)
        mult = 1 if ndim == 1 else 4
        self.ndim = ndim
        self.c_in = c_in
        self.c_out = c_in if c_out is None else c_out
        self.lift = conv(ndim, c_in, w, k=1)
        self.unet = UNet(w * mult, ndim, cfg, c_out=w * mult, width=w, depth=_depth(cfg, 2))
        self.project = conv(ndim, w, self.c_out, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        c = haar_1d(h) if self.ndim == 1 else haar_2d(h)
        c = self.unet(c)
        h = ihaar_1d(c) if self.ndim == 1 else ihaar_2d(c)
        return _residual_skip(x, self.project(h), self.c_in, self.c_out)


class DeepONet(nn.Module):
    """DeepONet surrogate: CNN branch over the field, MLP trunk over coordinates."""

    def __init__(self, c_in: int, ndim: int, cfg: dict | None = None, c_out: int | None = None) -> None:
        super().__init__()
        p = int((cfg or {}).get("basis", 32))
        self.ndim = ndim
        self.c_out = c_in if c_out is None else c_out
        self.p = p
        self.branch = nn.Sequential(
            conv(ndim, c_in, 32, k=3), nn.GELU(), conv(ndim, 32, 32), nn.GELU(),
            conv(ndim, 32, 16, k=1), nn.GELU(),
        )
        self.branch_head = nn.Linear(16, self.c_out * p)
        self.trunk = nn.Sequential(nn.Linear(ndim, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, p))
        self.bias = nn.Parameter(torch.zeros(self.c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.branch(x)
        h = F.adaptive_avg_pool1d(h, 1).flatten(1) if self.ndim == 1 else F.adaptive_avg_pool2d(h, 1).flatten(1)
        branch = self.branch_head(h).view(B, self.c_out, self.p)  # (B, C_OUT, P)
        coords = torch.stack(
            torch.meshgrid(
                *[torch.linspace(0.0, 1.0, s, device=x.device, dtype=x.dtype) for s in x.shape[2:]], indexing="ij"
            ),
            dim=-1,
        ).reshape(-1, self.ndim)  # (N_PTS, ndim)
        trunk = self.trunk(coords)  # (N_PTS, P)
        out = torch.einsum("bcp,np->bcn", branch, trunk) + self.bias[None, :, None]
        return out.reshape(B, self.c_out, *x.shape[2:])
